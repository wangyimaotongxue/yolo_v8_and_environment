import cv2
import time
import collections


class VideoSaver:
    def __init__(self, output_path="src/inference/videos/", fps=30.0, frame_size=(1280, 720)):
        """初始化视频存储"""
        self.fps = fps
        self.frame_size = frame_size
        self.buffer = collections.deque(maxlen=int(fps * 30))  # 存储最近 30s 的帧
        self.is_recording = False
        self.video_writer = None  # ✅ 统一命名，避免 `AttributeError`
        self.start_time = None

    def write_frame(self, frame):
        """写入单帧到视频"""
        if self.is_recording and self.video_writer:
            self.video_writer.write(frame)

    def add_frame(self, frame):
        """将帧添加到缓冲区"""
        self.buffer.append(frame)
        if self.is_recording and self.video_writer:
            self.video_writer.write(frame)

    def start_recording(self, output_path):
        """开始保存视频（前30秒+后30秒）"""
        if self.is_recording:
            return  # 已经在录制，则不重复启动

        fourcc = cv2.VideoWriter_fourcc(*'H264')
        self.video_writer = cv2.VideoWriter(output_path, fourcc, self.fps, self.frame_size)

        print(f"📼 开始保存视频: {output_path}")
        for frame in self.buffer:
            self.video_writer.write(frame)

        self.is_recording = True
        self.start_time = time.time()

    def stop_recording_if_needed(self):
        """在录制达到30秒后停止"""
        if self.is_recording and time.time() - self.start_time >= 30:
            self.is_recording = False
            if self.video_writer:
                self.video_writer.release()
            print("✅ 录制完成，视频已保存")

    def release(self):
        """释放资源"""
        if self.video_writer:
            self.video_writer.release()
            print("📁 视频存储资源已释放")
        else:
            print("⚠️ 没有正在写入的视频，无需释放")
