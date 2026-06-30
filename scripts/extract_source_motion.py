#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, Tuple


def read_videos(path: Path) -> Iterable[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f)


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
    parser = argparse.ArgumentParser(description="Extract source-video motion features keyed by video_id,t_s.")
    parser.add_argument("--videos-csv", required=True, help="CSV with video_id,path columns.")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--path-root", default="", help="Optional root prepended to relative video paths.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path_root = Path(args.path_root) if args.path_root else Path(".")
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "t_s", "motion"])
        writer.writeheader()
        count = 0
        for row in read_videos(Path(args.videos_csv)):
            video_id = row.get("video_id") or row.get("id")
            if not video_id:
                raise SystemExit("videos_csv must include video_id or id")
            raw_path = row.get("path")
            if not raw_path:
                continue
            video_path = Path(raw_path)
            if not video_path.is_absolute():
                video_path = path_root / video_path
            for t_s, motion in extract_motion(video_path, args.sample_fps):
                writer.writerow({"video_id": video_id, "t_s": f"{t_s:.3f}", "motion": f"{motion:.6f}"})
                count += 1
    print(f"Wrote {output} with {count} rows")


if __name__ == "__main__":
    main()
