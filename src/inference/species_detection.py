# species_detection.py

import cv2
import torch
from collections import defaultdict
from camera import Camera  # 导入自定义相机类
from yolo_find import YOLODetector  # 导入YOLO检测类
from ultralytics import YOLO

class SpeciesDetector:
    def __init__(self, model_path):
        """加载YOLO模型"""
        self.device = "0"
        self.model = YOLO(model_path)
        print("模型加载成功")
        self.species_count = defaultdict(int)  # 用于计数每种物种
        self.confidences = defaultdict(list)  # 存储每个物种的置信度

    def detect_and_display(self, frame):
        """检测并显示物种标签、置信度和计数"""
        # 对输入帧进行目标检测
        results = self.model.predict(source=frame, show=False)
        annotated_frame = results[0].plot()  # 获取标注后的帧

        # 提取标签和置信度
        species_count = defaultdict(int)
        confidences = defaultdict(list)

        for box in results[0].boxes:
            label = int(box.cls)  # 获取标签（物种ID）
            confidence = box.conf  # 获取置信度

            # 假设你的标签已经映射到物种名称
            species_name = self.model.names[label]
            species_count[species_name] += 1
            confidences[species_name].append(confidence.item())

        # 计算并显示物种的计数与置信度
        species_info = ""
        for species_name, count in species_count.items():
            avg_confidence = sum(confidences[species_name]) / len(confidences[species_name]) if confidences[species_name] else 0
            species_info += f"{species_name}: {count} ({avg_confidence:.2f})  "

        # 在画面上显示信息
        cv2.putText(annotated_frame, species_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        return annotated_frame

def main():
    model_path = "/home/cc/yolo_v8_and_environment/src/runs/detect/train/weights/best.pt"
    detector = SpeciesDetector(model_path)
    camera = Camera(4)  # 使用相机类实例化摄像头

    print("开始实时检测，按 'q' 退出")
    try:
        while True:
            frame = camera.get_frame()  # 获取摄像头帧
            if frame is None:
                break

            # 调用检测函数
            annotated_frame = detector.detect_and_annotate(frame)  # 使用YOLODetector实例进行标注

            # 显示带有检测框和信息的画面
            cv2.imshow("YOLOv8 Detection", annotated_frame)

            # 按 'q' 键退出
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("检测已退出")
                break
    except Exception as e:
        print(f"运行错误: {e}")
    finally:
        camera.release()  # 释放相机
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
