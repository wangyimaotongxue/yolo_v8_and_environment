import cv2


class VideoSaver:
    def __init__(self, output_path="src/inference/videos/", fps=30.0, frame_size=(1280, 720)):
        """初始化视频写入器"""
        fourcc = cv2.VideoWriter_fourcc(*'H264')  # 试试 'H264' 或 'avc1'
        self.video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

        if not self.video_writer.isOpened():
            print("⚠️ 视频写入器初始化失败，尝试不同的编码器")

        print(f"✅ 视频写入器已初始化: {output_path}")

    def write_frame(self, frame):
        """写入单帧到视频"""
        self.video_writer.write(frame)

    def release(self):
        """释放视频写入器"""
        self.video_writer.release()
        print("✅ 视频写入器已释放")
