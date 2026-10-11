#!/usr/bin/env python3
"""Extract ordered JPEG frames from a video for ObjectGS."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=3.0)
    parser.add_argument("--quality", type=int, default=2)
    args = parser.parse_args()

    video = args.video.expanduser().resolve()
    output = args.output.expanduser().resolve()

    if not video.is_file():
        parser.error(f"Video not found: {video}")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if not 2 <= args.quality <= 31:
        parser.error("--quality must be between 2 and 31")
    if shutil.which("ffmpeg") is None:
        parser.error("ffmpeg not found in PATH")
    if output.exists() and any(output.iterdir()):
        parser.error(f"Output directory is not empty: {output}")

    output.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-nostdin",
        "-i", str(video),
        "-vf", f"fps={args.fps}",
        "-q:v", str(args.quality),
        "-start_number", "0",
        str(output / "frame_%04d.jpg"),
    ]

    print("Video:", video, flush=True)
    print("FPS:", args.fps, flush=True)
    print("Output:", output, flush=True)

    subprocess.run(cmd, check=True)

    frames = sorted(output.glob("frame_*.jpg"))

    if not frames:
        raise RuntimeError("FFmpeg produced no frames")

    manifest = {
        "video": str(video),
        "video_size": video.stat().st_size,
        "video_mtime_ns": video.stat().st_mtime_ns,
        "fps": args.fps,
        "quality": args.quality,
        "frames": [p.name for p in frames],
    }

    (output / "frames_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(f"Extracted {len(frames)} frames")
    print("First:", frames[0].name)
    print("Last:", frames[-1].name)


if __name__ == "__main__":
    main()
