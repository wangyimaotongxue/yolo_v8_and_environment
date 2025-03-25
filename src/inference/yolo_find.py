import cv2
import torch
# from main import model
from ultralytics import YOLO
from camera_out import Camera

class YOLODetector:
    def __init__(self, model_path):
        """加载YOLO模型"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path)
        print("模型加载成功")

    def detect_and_annotate(self, frame):
        """对输入帧进行目标检测并返回标注后的帧"""
        results = self.model.predict(source=frame, show=False)
        return results[0].plot()  # 返回带有检测框的帧

def main():
    model_path = "/home/cc/Documents/yolov8_project/src/runs/detect/train/weights/best.pt"
    detector = YOLODetector(model_path)
    camera = Camera(0)

    print("开始实时检测，按 'q' 退出")
    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                break

            annotated_frame = detector.detect_and_annotate(frame)

            cv2.imshow("YOLOv8 Detection", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("检测已退出")
                break
    except Exception as e:
        print(f"运行错误: {e}")
    finally:
        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
