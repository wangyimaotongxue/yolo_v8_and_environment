import cv2

class Camera:
    def __init__(self, camera_index):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            print("无法打开摄像头")
            exit()
        print(f"摄像头 {camera_index} 已成功打开")

    def get_frame(self):
        """获取摄像头帧"""
        ret, frame = self.cap.read()
        if not ret:
            print("无法读取摄像头帧")
            return None
        return frame

    def release(self):
        """释放摄像头资源"""
        self.cap.release()
        print("摄像头已释放")

