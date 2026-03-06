import os
from rembg import remove
from PIL import Image, ImageSequence

input_file = 'o5ns4-1dg70.gif'
output_file = 'robot_animated_no_bg.gif'

def process_gif():
    if not os.path.exists(input_file):
        print(f"找不到檔案：{input_file}")
        return

    print("AI 正在逐幀去背中，這需要一點時間...")
    img = Image.open(input_file)
    
    frames = []
    for frame in ImageSequence.Iterator(img):
        # 逐幀去背
        new_frame = remove(frame.convert("RGBA"))
        frames.append(new_frame)

    # 儲存回動態 GIF
    frames[0].save(
        output_file,
        save_all=True,
        append_images=frames[1:],
        duration=img.info.get('duration', 100),
        loop=0,
        disposal=2
    )
    print(f"完成！請查看資料夾中的 {output_file}")

if __name__ == "__main__":
    process_gif()