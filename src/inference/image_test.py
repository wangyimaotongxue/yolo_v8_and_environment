import cv2
from ultralytics import YOLO

# 加载训练好的YOLOv8模型
model = YOLO("/home/stick/yolov8_project/src/runs/detect/train/weights/best.pt")  # 替换为你的模型路径

# 加载待识别的图片
image_path = "/home/stick/test_image.jpg"  # 替换为你的图片路径
image = cv2.imread(image_path)

if image is None:
    print("无法加载图片，请检查路径是否正确")
    exit()

# 使用YOLOv8模型进行推理
results = model.predict(source=image)

# 绘制检测结果
annotated_image = results[0].plot()

# 检查图片是否正确处理
if annotated_image is None:
    print("图片处理失败")
    exit()

# 调整窗口大小适配图片
window_name = "目标检测结果"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # 设置窗口可以调整大小
cv2.imshow(window_name, annotated_image)

# 按任意键退出
cv2.waitKey(0)
cv2.destroyAllWindows()

