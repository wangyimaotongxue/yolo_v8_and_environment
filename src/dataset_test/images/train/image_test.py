# import os

# def rename_images(directory):
#     for root, _, files in os.walk(directory):
#         for file in files:
#             if file.lower().endswith(('.jpg', '.jpeg')):
#                 old_path = os.path.join(root, file)
#                 new_path = os.path.join(root, os.path.splitext(file)[0] + '.png')
#                 os.rename(old_path, new_path)
#                 print(f'Renamed: {old_path} -> {new_path}')

# if __name__ == "__main__":
#     target_directory = "./"  # 修改为你的目标文件夹路径
#     rename_images(target_directory)


import os

def rename_images(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg')):
                old_path = os.path.join(root, file)
                new_path = os.path.join(root, os.path.splitext(file)[0] + '.png')

                print(f"Renaming: {old_path} -> {new_path}")  # 调试信息
                
                os.rename(old_path, new_path)

if __name__ == "__main__":
    target_directory = "./"  # 设为当前目录
    rename_images(target_directory)
