from PIL import Image
import os

def convert_images_to_24bit(folder_path):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 构建文件的完整路径
        file_path = os.path.join(folder_path, filename)
        
        # 尝试打开图像文件
        try:
            with Image.open(file_path) as img:
                # 检查图像的位深度
                if img.mode != 'RGB':  # 非24位的RGB图像
                    print(f"Converting {filename} to 24-bit...")
                    # 将图像转换为24位RGB
                    img = img.convert('RGB')
                    # 保存修改后的图像覆盖原文件
                    img.save(file_path)
                    print(f"{filename} has been converted.")
                else:
                    print(f"{filename} is already 24-bit.")
        except IOError:
            print(f"Failed to open {filename}. Skipping...")

# 指定你的文件夹路径
folder_path = 'D:/code/MSB/Datasets/UAV-Rain1k/train/input'
convert_images_to_24bit(folder_path)