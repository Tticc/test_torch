from PIL import Image, ImageDraw
import os

def generate_image(target_size_kb):
    # 图像尺寸 (可以调整)
    width, height = 7024, 7024

    # 创建一张新的白色背景的图像
    image = Image.new('RGB', (width, height), color=(255, 255, 255))

    # 使用 ImageDraw 画一些内容
    draw = ImageDraw.Draw(image)
    draw.text((100, 100), "Hello, World!", fill=(0, 0, 0))

    # 设置输出文件名
    output_file = 'generated_image.jpg'

    # 初始压缩质量
    quality = 85  # 初始设定质量为85

    # 计算目标文件大小 (目标文件大小为目标大小 * 1024)
    target_size = target_size_kb * 1024  # 转换为字节

    # 保存图像并调整质量，直到文件大小接近目标大小
    while True:
        # 保存图像为JPEG文件
        image.save(output_file, 'JPEG', quality=quality)

        # 获取文件大小
        file_size = os.path.getsize(output_file)

        print(f"Current file size: {file_size / 1024:.2f} KB")

        # 检查文件大小是否达到目标大小
        if file_size >= target_size - 1024 and file_size <= target_size + 1024:
            print(f"Image size is now {file_size / 1024:.2f} KB.")
            break

        # 调整压缩质量
        if file_size > target_size:
            quality -= 5  # 如果文件太大，降低质量
        else:
            quality += 5  # 如果文件太小，增加质量

    print(f"Final image size: {os.path.getsize(output_file) / 1024:.2f} KB")
    return output_file

# 目标文件大小为 1024KB
generate_image(1024)
