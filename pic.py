from PIL import Image, ImageDraw
import os


def generate_1024kb_image():
    target_size_kb = 1048576  # 目标大小 1024KB
    output_path = "output.jpg"  # 输出文件路径
    quality = 85  # 初始JPEG压缩质量 (0-100)
    width, height = 1200, 800  # 初始分辨率

    while True:
        # 1. 创建图片（红色背景）
        img = Image.new("RGB", (width, height), color="red")
        draw = ImageDraw.Draw(img)

        # 2. 添加内容（增加文件大小）
        draw.text((10, 10), "1024KB Image", fill="white")  # 文字会增加复杂度

        # 3. 保存并检查文件大小
        img.save(output_path, "JPEG", quality=quality, optimize=True)
        current_size = os.path.getsize(output_path)  # 转换为KB

        print(f"质量: {quality}% | 分辨率: {width}x{height} | 当前大小: {current_size}KB")

        # 4. 判断是否达标
        if current_size == target_size_kb:
            print(f"\n生成成功！文件已保存至: {output_path}")
            break
        elif current_size > target_size_kb:
            # 优先降低质量，质量过低后减小分辨率
            if quality > 10:  # 防止质量过低导致严重失真
                quality -= 2
            else:
                width = int(width * 0.95)
                height = int(height * 0.95)
        else:
            # 优先增加质量，质量到顶后增大分辨率
            if quality < 95:  # 防止质量100时文件过大
                quality += 2
            else:
                width = int(width * 1.05)
                height = int(height * 1.05)

        # 安全限制：防止无限循环
        if width < 100 or height < 100 or quality < 5:
            print("无法生成目标大小的图片！")
            break


if __name__ == "__main__":
    generate_1024kb_image()