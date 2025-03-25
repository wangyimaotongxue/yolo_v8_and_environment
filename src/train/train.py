from ultralytics import YOLO  # 引入 YOLOv8 训练模块

# 选择要训练的模型（可以是 yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt）
model = YOLO("yolov8n.pt")  # 这里使用 YOLOv8 最小模型 yolov8n

# 训练模型
model.train(
    data="/home/stick/yolov8_project/src/data.yaml",  # 你的数据集配置文件
    epochs=100,         # 训练轮数
    batch=32,          # 批次大小
    imgsz=720,         # 训练图片尺寸
    device="0"      # 使用 GPU 训练，如无 GPU，可改为 "cpu"
)

