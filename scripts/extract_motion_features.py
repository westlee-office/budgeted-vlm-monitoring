#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, Tuple


def iter_videos(root: Path) -> Iterable[Path]:
    exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() in exts:
            yield path


def extract_motion(video_path: Path, sample_fps: float) -> Iterable[Tuple[float, float]]:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise SystemExit("OpenCV is required: pip install opencv-python") from exc

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, int(round(fps / sample_fps)))
    prev = None
    frame_idx = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_idx % stride != 0:
            frame_idx += 1
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (160, 90))
        if prev is None:
            motion = 0.0
        else:
            diff = cv2.absdiff(gray, prev)
            motion = min(float(diff.mean()) / 48.0, 1.0)
        yield frame_idx / fps, motion
        prev = gray
        frame_idx += 1
    capture.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract simple motion scores into a BMVM signals CSV.")
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--episode-id", default="episode-0000")
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--default-anomaly", type=float, default=0.0)
    parser.add_argument("--default-clip", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_root = Path(args.video_root)
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    videos = list(iter_videos(video_root))
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["episode_id", "stream_id", "t_s", "motion", "anomaly", "clip", "event_ids"],
        )
        writer.writeheader()
        for stream_idx, video in enumerate(videos):
            stream_id = f"s{stream_idx:03d}"
            for t_s, motion in extract_motion(video, sample_fps=args.sample_fps):
                writer.writerow(
                    {
                        "episode_id": args.episode_id,
                        "stream_id": stream_id,
                        "t_s": f"{t_s:.3f}",
                        "motion": f"{motion:.6f}",
                        "anomaly": f"{args.default_anomaly:.6f}",
                        "clip": f"{args.default_clip:.6f}",
                        "event_ids": "",
                    }
                )
    print(f"Wrote {output} from {len(videos)} videos")


if __name__ == "__main__":
    main()
