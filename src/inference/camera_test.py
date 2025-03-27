import cv2
import torch
from ultralytics import YOLO
import time


# 初始化YOLOv8模型
def load_yolo_model(model_path):
    try:
        device = "0" if torch.cuda.is_available() else "cpu"  # 自动选择 GPU 或 CPU
        model = YOLO(model_path)
        print("模型加载成功")
        return model
    except Exception as e:
        print(f"模型加载失败: {e}")
        exit()

# 初始化摄像头
def initialize_camera(camera_index):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("无法打开摄像头")
        exit()
    print("摄像头已成功打开")
    return cap

# 实时目标监测
def real_time_detection(model, cap, save_output=False):
    # 初始化视频写入器（如果需要保存视频）
    video_writer = None
    if save_output:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_writer = cv2.VideoWriter(
            "output.avi",
            cv2.VideoWriter_fourcc(*'XVID'),
            20.0,
            (frame_width, frame_height)
        )
    # 开始时间，用于计算FPS
    start_time = time.time()
    frame_count = 0

    print("开始实时监测，按 'q' 键退出")
    try:
        while True:
            # 读取摄像头帧
            ret, frame = cap.read()
            if not ret:
                print("无法读取摄像头帧")
                break

            # 使用YOLOv8模型推理
            results = model.predict(source=frame, show=False)

            # 在帧上绘制检测结果
            annotated_frame = results[0].plot()
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

            # 保存检测结果到视频（可选）
            if save_output and video_writer:
                video_writer.write(annotated_frame)

            # 创建显示窗口并调整大小
            cv2.namedWindow("YOLOv8 Realtime Detection", cv2.WINDOW_NORMAL)
            # cv2.resizeWindow("YOLOv8 Realtime Detection", 640, 720)

            # 获取窗口的大小
            # window_width, window_height = cv2.getWindowImageRect("YOLOv8 Realtime Detection")[2:4]

            # 调整帧大小以适配窗口

            # resized_frame = cv2.resize(annotated_frame, (window_width, window_height), interpolation=cv2.INTER_LINEAR)

            # 显示实时视频流
            cv2.imshow("YOLOv8 Realtime Detection", annotated_frame)

            # 按 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("检测已退出")
                break
    except Exception as e:
        print(f"运行中出现错误: {e}")
    finally:
        # 释放资源
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        print("资源已释放")

# 主函数
if __name__ == "__main__":
    # 替换为你的训练好的模型路径
    model_path = "/home/cc/Documents/yolov8_project/src/runs/detect/train/weights/best.pt"

    # 加载模型
    yolo_model = load_yolo_model(model_path)

    # 初始化摄像头
    camera = initialize_camera(4)

    # 开始实时目标监测（设置 save_output=True 可以保存结果视频）
    real_time_detection(yolo_model, camera, save_output=False)
