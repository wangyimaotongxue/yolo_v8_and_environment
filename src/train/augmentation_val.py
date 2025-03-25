import os
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm

# 数据路径
IMAGE_DIR = "./images"  # 原始图片
LABEL_DIR = "./labels"  # YOLO 标签
OUTPUT_IMAGE_DIR = "./augmented_images"  # 增强后的图片
OUTPUT_LABEL_DIR = "./augmented_labels"  # 增强后的标签

# 确保输出文件夹存在
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

# 定义数据增强策略
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Rotate(limit=20, p=0.5),
    A.GaussianBlur(blur_limit=(3, 7), p=0.2),
    A.CLAHE(clip_limit=2, tile_grid_size=(8, 8), p=0.2),
    A.RandomSizedCrop(min_max_height=(320, 640), height=640, width=640, p=0.3),
], bbox_params=A.BboxParams(format='yolo', label_fields=['category']))

# 读取所有图片文件
image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))]

# 开始数据增强
for img_file in tqdm(image_files, desc="Augmenting Images"):
    img_path = os.path.join(IMAGE_DIR, img_file)
    label_path = os.path.join(LABEL_DIR, img_file.replace('.jpg', '.txt').replace('.png', '.txt'))

    # 读取图片
    image = cv2.imread(img_path)
    h, w, _ = image.shape

    # 读取 YOLO 标签
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()
        boxes = []
        categories = []
        for line in lines:
            parts = line.strip().split()
            category = int(parts[0])
            x_center, y_center, bbox_w, bbox_h = map(float, parts[1:])
            boxes.append([x_center, y_center, bbox_w, bbox_h])
            categories.append(category)
    else:
        boxes = []
        categories = []

    # 生成增强数据
    for i in range(3):  # 每张图片增强3次
        augmented = transform(image=image, bboxes=boxes, category=categories)
        aug_image = augmented['image']
        aug_boxes = augmented['bboxes']
        aug_categories = augmented['category']

        # 保存增强后的图片
        aug_img_filename = f"aug_{i}_{img_file}"
        aug_img_path = os.path.join(OUTPUT_IMAGE_DIR, aug_img_filename)
        cv2.imwrite(aug_img_path, aug_image)

        # 保存增强后的标签
        aug_label_filename = aug_img_filename.replace('.jpg', '.txt').replace('.png', '.txt')
        aug_label_path = os.path.join(OUTPUT_LABEL_DIR, aug_label_filename)
        with open(aug_label_path, 'w') as f:
            for category, bbox in zip(aug_categories, aug_boxes):
                f.write(f"{category} {' '.join(map(str, bbox))}\n")

print("✅ 数据增强完成！")
