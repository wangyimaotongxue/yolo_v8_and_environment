import os
import time
import cv2
from camera_out import Camera
from yolo_find import YOLODetector
from video_save import VideoSaver


def main():
    model_path = "/home/cc/Documents/yolov8_project/src/runs/detect/train/weights/best.pt"

    camera = Camera(4)
    detector = YOLODetector(model_path)
    video_saver = VideoSaver("output.avi", fps=30.0, frame_size=(640, 480))

    # 开始时间，用于计算FPS
    start_time = time.time()
    frame_count = 0

    print("开始检测并保存视频，按 'q' 退出")
    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                break


            # 检测并标注
            annotated_frame = detector.detect_and_annotate(frame)

            # 计算FPS
            frame_count += 1
            current_time = time.time()
            elapsed_time = current_time - start_time
            fps = frame_count / elapsed_time

            # 在帧上显示FPS
            cv2.putText(
                annotated_frame,
                f"FPS: {fps:.2f}",
                (10, 30),  # 文本位置
                cv2.FONT_HERSHEY_SIMPLEX,  # 字体
                1,  # 字体大小
                (0, 255, 0),  # 颜色 (绿色)
                2,  # 粗细
                cv2.LINE_AA  # 抗锯齿
            )

            ##
            cv2.namedWindow("YOLOv8 Detection",cv2.WINDOW_NORMAL)
            # 显示
            cv2.imshow("YOLOv8 Detection", annotated_frame)

            # 保存到本地
            video_saver.write_frame(annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("检测已退出")
                break
    except Exception as e:
        print(f"运行错误: {e}")
    finally:
        camera.release()
        video_saver.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
