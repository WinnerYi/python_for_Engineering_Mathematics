import subprocess
import argparse

def main(args):
    VIDEO_INPUT = args.video      # e.g., video_b.mp4
    AUDIO_INPUT = args.audio      # e.g., voice_a.mp4
    OUTPUT = args.output          # e.g., final_video.mp4

    command = [
        "ffmpeg",
        "-y",
        "-i", VIDEO_INPUT,
        "-i", AUDIO_INPUT,
        "-c:v", "copy",       # copy video as-is
        "-c:a", "aac",        # encode audio as AAC
        "-b:a", "192k",       # audio bitrate
        "-map", "0:v:0",      # take video from first input
        "-map", "1:a:0",      # take audio from second input
        "-shortest",          # trim to shorter of video/audio
        OUTPUT
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print("❌ ffmpeg error:", result.stderr)
    else:
        print("✅ Done! Output saved as:", OUTPUT)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Video to keep")
    parser.add_argument("--audio", required=True, help="Audio to replace with")
    parser.add_argument("--output", required=True, help="Output video file")
    args = parser.parse_args()
    main(args)
