#!/usr/bin/env python3
"""
WORKING FACE SWAP with ENHANCE mode - 2025
Uses InsightFace wrapper for face detection, embedding, swapping,
and optional GFPGAN for face enhancement (restoration).
"""

# python face_swap.py --source source.jpg --target target.mp4 --output output_video.mp4 --enhance --gpu

import os
import sys
import cv2
import numpy as np
from pathlib import Path
import argparse

# Install required packages if not already installed:
# pip install insightface onnxruntime opencv-python tqdm huggingface-hub gfpgan

try:
    from insightface.app import FaceAnalysis
    from insightface.model_zoo import get_model
except ImportError:
    print("❌ Install InsightFace: pip install insightface")
    sys.exit(1)


try:
    import onnxruntime as ort
except ImportError:
    print("❌ Install onnxruntime: pip install onnxruntime")
    sys.exit(1)


try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


try:
    from gfpgan import GFPGANer
except ImportError:
    GFPGAN_AVAILABLE = False
else:
    GFPGAN_AVAILABLE = True


class FaceSwapper:
    def __init__(self, model_path='inswapper_128.onnx', use_gpu=False, enhance=False):
        print("🔧 Loading InsightFace models...")
        self.face_detector = FaceAnalysis(name='buffalo_l')
        ctx_id = 0 if use_gpu else -1
        self.face_detector.prepare(ctx_id=ctx_id, det_size=(640, 640))
        self.model_path = model_path

        providers = ['CUDAExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
        self.swapper = get_model(self.model_path, providers=providers)
        print("✅ InsightFace models loaded")

        self.restorer = None
        if enhance:
            if not GFPGAN_AVAILABLE:
                print("❌ GFPGAN package not installed, cannot enable enhancement.")
                sys.exit(1)
            print("🔧 Loading GFPGAN face restoration model...")
            self.restorer = GFPGANer(
                model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
                upscale=1,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=None
            )
            print("✅ GFPGAN model loaded for enhancement")

    def get_faces(self, image):
        return self.face_detector.get(image)

    def swap_face(self, source_img, target_img, source_face_index=0, target_face_index=0, paste_back=True):
        source_faces = self.get_faces(source_img)
        if not source_faces:
            print("❌ No face detected in source image")
            return None
        source_face = source_faces[source_face_index]

        target_faces = self.get_faces(target_img)
        if not target_faces:
            print("❌ No face detected in target image")
            return None
        target_face = target_faces[target_face_index]

        # Perform face swap
        swapped = self.swapper.get(target_img, target_face, source_face, paste_back=paste_back)
        if swapped is None:
            print("❌ Face swap failed during model inference")
            return None

        # Optional enhancement
        if self.restorer is not None:
            _, _, restored_img = self.restorer.enhance(
                swapped,
                has_aligned=False,
                only_center_face=False,
                paste_back=True
            )
            if restored_img is not None:
                swapped = restored_img

        return swapped

    def swap_video(self, source_path, video_path, output_path, face_index=0, fps=None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Cannot open video {video_path}")
            return
        fps_input = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        output_fps = fps or fps_input

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
        print(f"🎬 Processing {total_frames} frames at {output_fps:.2f} FPS")

        source_img = cv2.imread(source_path)
        source_faces = self.get_faces(source_img)
        if not source_faces:
            print("❌ No face detected in source image.")
            cap.release()
            out.release()
            return
        source_face = source_faces[face_index]

        processed = 0
        failed = 0

        frame_iterator = tqdm(range(total_frames), desc='Processing') if HAS_TQDM else range(total_frames)

        for _ in frame_iterator:
            ret, frame = cap.read()
            if not ret:
                break
            try:
                target_faces = self.get_faces(frame)
                if target_faces:
                    swapped_frame = self.swapper.get(frame, target_faces[0], source_face, paste_back=True)
                    if swapped_frame is None:
                        # If failed, keep original frame
                        swapped_frame = frame
                    # Optional GFPGAN enhancement
                    if self.restorer is not None:
                        _, _, restored_img = self.restorer.enhance(
                            swapped_frame,
                            has_aligned=False,
                            only_center_face=False,
                            paste_back=True
                        )
                        if restored_img is not None:
                            swapped_frame = restored_img
                    frame = swapped_frame
                out.write(frame)
                processed += 1
            except Exception as e:
                print(f"⚠️ Frame processing failed: {e}")
                failed += 1

        cap.release()
        out.release()
        print(f"✅ Saved swapped video: {output_path}")
        print(f"Faces processed: {processed}, failed: {failed}")


def main():
    parser = argparse.ArgumentParser(description='🎭 Face Swap Tool with Optional GFPGAN Enhancement')
    parser.add_argument('--source', required=True, help='Source face image')
    parser.add_argument('--target', required=True, help='Target image or video')
    parser.add_argument('--output', required=True, help='Output file path')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for models')
    parser.add_argument('--face-index', type=int, default=0, help='Face index in source image when multiple faces')
    parser.add_argument('--fps', type=float, default=None, help='FPS for output video (if target is video)')
    parser.add_argument('--enhance', action='store_true', help='Enable GFPGAN face enhancement')
    args = parser.parse_args()

    swapper = FaceSwapper(use_gpu=args.gpu, enhance=args.enhance)

    is_video = args.target.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))

    if is_video:
        swapper.swap_video(args.source, args.target, args.output, face_index=args.face_index, fps=args.fps)
    else:
        source_img = cv2.imread(args.source)
        target_img = cv2.imread(args.target)
        if source_img is None or target_img is None:
            print("❌ Cannot load source or target images")
            sys.exit(1)
        result = swapper.swap_face(source_img, target_img, source_face_index=args.face_index)
        if result is not None:
            cv2.imwrite(args.output, result)
            print(f"✅ Saved swapped image: {args.output}")
        else:
            print("❌ Face swap failed.")


if __name__ == '__main__':
    main()
