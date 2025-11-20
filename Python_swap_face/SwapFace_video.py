# 11327217
import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
from insightface.app import FaceAnalysis
from insightface.model_zoo import model_zoo
import subprocess

def main(args):
    print("🚀 啟動 InsightFace 模型...")
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=-1, det_size=(640, 640))

    print("🔁 載入 InSwapper 模型...")
    swapper = model_zoo.get_model('inswapper_128.onnx', providers=['CPUExecutionProvider'])

    print("🎥 開始處理影片:", args.input)
    video = cv2.VideoCapture(args.input)
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('temp_video.mp4', fourcc, fps, (width, height))

    source_img = cv2.imread(args.source)
    source_faces = app.get(source_img)
    if not source_faces:
        raise ValueError("❌ 無法在 source 圖片中偵測到人臉")
    source_face = source_faces[0]

    print("🔄 開始處理影片幀...")
    for _ in tqdm(range(total_frames), desc="處理中", unit="幀"):
        ret, frame = video.read()
        if not ret or frame is None:
            continue

        faces = app.get(frame)
        for face in faces:
            # ✅ 強制每幀換來源臉
            swapped = swapper.get(frame, face, source_face, paste_back=True)
            if swapped is not None and isinstance(swapped, np.ndarray) and swapped.shape == frame.shape:
                frame = swapped

        out.write(frame)

    video.release()
    out.release()

    # 合併音訊
    VIDEO_OUTPUT_NO_AUDIO = "temp_video.mp4"
    VIDEO_OUTPUT = args.output
    print("合併原始音訊…")
    os.system(
        f'ffmpeg -y -i "{VIDEO_OUTPUT_NO_AUDIO}" -i "{args.input}" '
        f'-map 0:v -map 1:a -c:v copy -c:a aac -b:a 192k -shortest "{VIDEO_OUTPUT}"'
    )
    print("✅ 換臉完成，音訊已整合，輸出檔案:", VIDEO_OUTPUT)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, help='要換上去的臉部圖片')
    parser.add_argument('--input', required=True, help='原始影片')
    parser.add_argument('--output', required=True, help='輸出影片')
    args = parser.parse_args()
    main(args)
