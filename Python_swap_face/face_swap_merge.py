#!/usr/bin/env python3
"""
FaceSwap + 音訊合併工具
功能：
1️⃣ 使用 InsightFace / GFPGAN 做影片換臉
2️⃣ 將指定音訊或原影片音訊合併到換臉影片
"""

import os
import sys
import cv2
import argparse
import subprocess
from pathlib import Path

# --- FaceSwap 相關套件 ---
try:
    from insightface.app import FaceAnalysis
    from insightface.model_zoo import get_model
except ImportError:
    print("❌ 未安裝 InsightFace，請先執行: pip install insightface")
    sys.exit(1)

try:
    from gfpgan import GFPGANer
    GFPGAN_AVAILABLE = True
except ImportError:
    GFPGAN_AVAILABLE = False

# -------------------- FaceSwapper 類別 --------------------
class FaceSwapper:
    def __init__(self, model_path='inswapper_128.onnx', use_gpu=False, enhance=False):
        """
        初始化 FaceSwapper
        model_path: InsightFace 換臉模型路徑
        use_gpu: 是否使用 GPU
        enhance: 是否使用 GFPGAN 增強臉部細節
        """
        print("🔧 載入 InsightFace 模型...")
        self.face_detector = FaceAnalysis(name='buffalo_l')
        ctx_id = 0 if use_gpu else -1
        self.face_detector.prepare(ctx_id=ctx_id, det_size=(640, 640))

        providers = ['CUDAExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
        self.swapper = get_model(model_path, providers=providers)
        print("✅ InsightFace 模型載入完成")

        # GFPGAN 增強設定
        self.restorer = None
        if enhance:
            if not GFPGAN_AVAILABLE:
                print("❌ 未安裝 GFPGAN 套件，無法啟用增強")
                sys.exit(1)
            print("🔧 載入 GFPGAN 模型...")
            self.restorer = GFPGANer(
                model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
                upscale=1, arch='clean', channel_multiplier=2, bg_upsampler=None
            )
            print("✅ GFPGAN 模型載入完成")

    def get_faces(self, image):
        """取得影像中的所有人臉"""
        return self.face_detector.get(image)

    def swap_video(self, source_path, video_path, temp_output, face_index=0):
        """
        將影片中的人臉換成來源人臉
        source_path: 來源人臉圖片
        video_path: 目標影片
        temp_output: 暫存輸出影片路徑
        face_index: 若來源圖有多張臉，選擇哪一張
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ 無法開啟影片 {video_path}")
            return False

        # 影片資訊
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 建立 VideoWriter 物件
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

        # 讀取來源人臉
        source_img = cv2.imread(source_path)
        source_faces = self.get_faces(source_img)
        if not source_faces:
            print("❌ 未偵測到來源人臉")
            cap.release()
            out.release()
            return False
        source_face = source_faces[face_index]

        print(f"🎬 開始處理 {total_frames} 幀影片...")
        processed = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 偵測目標影片中的人臉並換臉
            target_faces = self.get_faces(frame)
            if target_faces:
                swapped = self.swapper.get(frame, target_faces[0], source_face, paste_back=True)
                if swapped is not None:
                    frame = swapped
                    # 若啟用 GFPGAN 增強
                    if self.restorer is not None:
                        _, _, restored = self.restorer.enhance(
                            frame, has_aligned=False, only_center_face=False, paste_back=True
                        )
                        if restored is not None:
                            frame = restored
            out.write(frame)
            processed += 1

        cap.release()
        out.release()
        print(f"✅ 暫存換臉影片已保存: {temp_output}")
        return True

# -------------------- 音訊合併函式 --------------------
def merge_audio(video_path, audio_path, output_path):
    """
    使用 ffmpeg 將影片與音訊合併
    video_path: 影片檔
    audio_path: 音訊檔
    output_path: 合併後輸出檔
    """
    if not os.path.isfile(video_path):
        print(f"❌ 找不到影片檔: {video_path}")
        return False
    if not os.path.isfile(audio_path):
        print(f"❌ 找不到音訊檔: {audio_path}")
        return False

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",        # 影片直接複製，不重新編碼
        "-c:a", "aac",         # 音訊轉 AAC
        "-b:a", "192k",        # 音訊位元率
        "-map", "0:v:0",       # 影片來源取第一個輸入
        "-map", "1:a:0",       # 音訊來源取第二個輸入
        "-shortest",           # 以最短長度為準
        output_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print("❌ ffmpeg 合併錯誤:\n", result.stderr)
        return False
    print(f"✅ 最終影片已保存: {output_path}")
    return True

# -------------------- 主程式 --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="來源人臉圖片")
    parser.add_argument("--target", required=True, help="目標影片")
    parser.add_argument("--audio", required=True, help="要合併的音訊檔")
    parser.add_argument("--output", required=True, help="最終輸出影片路徑")
    parser.add_argument("--gpu", action="store_true", help="使用 GPU 加速")
    parser.add_argument("--enhance", action="store_true", help="啟用 GFPGAN 增強臉部細節")
    args = parser.parse_args()

    temp_video = "temp_swap.mp4"  # 暫存換臉影片

    # 1️⃣ 換臉影片
    swapper = FaceSwapper(use_gpu=args.gpu, enhance=args.enhance)
    if swapper.swap_video(args.source, args.target, temp_video):
        # 2️⃣ 合併音訊
        merge_audio(temp_video, args.audio, args.output)
        # 3️⃣ 刪除暫存影片
        os.remove(temp_video)

if __name__ == "__main__":
    main()
