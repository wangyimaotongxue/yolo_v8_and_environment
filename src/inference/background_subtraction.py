# background_subtraction.py

import cv2
import time
import numpy as np
from collections import deque

class BackgroundSubtractor:
    def __init__(self):
        """初始化背景分割"""
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=100, detectShadows=True)
        self.unknown_count = 0
        self.detection_history = deque(maxlen=10)  # 保存最后10帧的检测记录

    def process_frame(self, frame):
        """处理帧并进行背景分割"""
        # 转为灰度图像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 应用背景减法
        fg_mask = self.bg_subtractor.apply(gray)

        # 提取前景轮廓
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        unknown_objects = []

        for contour in contours:
            if cv2.contourArea(contour) > 500:  # 如果轮廓面积较大，则认为是一个物体
                x, y, w, h = cv2.boundingRect(contour)
                unknown_objects.append((x, y, w, h))

        return unknown_objects, fg_mask

    def display_unknown_objects(self, frame, unknown_objects):
        """在画面上显示未知物种和时间"""
        for (x, y, w, h) in unknown_objects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 画出矩形框

        # 获取当前时间
        current_time = time.strftime("%H:%M:%S", time.localtime())

        # 显示“未知物种”标签和时间
        cv2.putText(frame, f"未知物种, 时间: {current_time}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        return frame

def main():
    camera = cv2.VideoCapture(4)  # 使用摄像头
    subtractor = BackgroundSubtractor()

    print("开始背景分割检测，按 'q' 退出")
    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                break

            # 获取未知物种（背景中出现的新物体）
            unknown_objects, fg_mask = subtractor.process_frame(frame)

            # 显示未知物种和时间
            annotated_frame = subtractor.display_unknown_objects(frame, unknown_objects)

            cv2.imshow("背景分割 - 未知物种", annotated_frame)

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
