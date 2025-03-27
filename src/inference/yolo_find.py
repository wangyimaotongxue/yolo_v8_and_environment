import cv2
import time
import torch
import collections
from camera import Camera
from ultralytics import YOLO
from video_save import VideoSaver


class YOLODetector:
    def __init__(self, model_path):
        """加载YOLO模型"""
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = "0"
        self.model = YOLO(model_path)
        print("模型加载成功")

        # 初始化视频存储
        self.saver = VideoSaver(fps=30.0, frame_size=(1280, 720))

    def detect_and_annotate(self, frame):
        """对输入帧进行目标检测并返回标注后的帧"""
        results = self.model.predict(source=frame, show=False)
        # results = self.model.predict(source=frame, show=False, imgsz=(720, 1280))

        # return results[0].plot()  # 返回带有检测框的帧
        annotated_frame = results[0].plot()

        # 检测目标
        detections = results[0].boxes
        if len(detections) > 0 and not self.saver.is_recording:
            self.saver.start_recording("output.mp4")  # 触发保存

        # 继续检查是否需要停止存储
        self.saver.stop_recording_if_needed()

        return annotated_frame  # 返回带有检测框的帧

def main():
    model_path = "/home/cc/yolo_v8_and_environment/src/runs/detect/train/weights/best.pt"
    detector = YOLODetector(model_path)
    camera = Camera(4)

    print("开始实时检测，按 'q' 退出")
    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                break

            detector.saver.add_frame(frame)  # 添加帧到缓冲区
            annotated_frame = detector.detect_and_annotate(frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("检测已退出")
                break
    except Exception as e:
        print(f"运行错误: {e}")
    finally:
        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
