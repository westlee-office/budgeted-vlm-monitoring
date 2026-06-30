#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def read_videos(path: Path) -> Iterable[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f)


def read_prompts(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def iter_frames(video_path: Path, sample_fps: float):
    try:
        import cv2  # type: ignore
        from PIL import Image  # type: ignore
    except ImportError as exc:
        raise SystemExit("OpenCV and Pillow are required: pip install opencv-python pillow") from exc

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, int(round(fps / sample_fps)))
    frame_idx = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_idx % stride != 0:
            frame_idx += 1
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yield frame_idx / fps, Image.fromarray(rgb)
        frame_idx += 1
    capture.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract CLIP prompt similarity scores keyed by video_id,t_s.")
    parser.add_argument("--videos-csv", required=True, help="CSV with video_id,path columns.")
    parser.add_argument("--prompts", required=True, help="Text file with one incident prompt per line.")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--path-root", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import torch  # type: ignore
        from transformers import CLIPModel, CLIPProcessor  # type: ignore
    except ImportError as exc:
        raise SystemExit("Torch and transformers are required: pip install torch transformers") from exc

    prompts = read_prompts(Path(args.prompts))
    if not prompts:
        raise SystemExit("Prompt file is empty")

    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(args.model).to(device)
    processor = CLIPProcessor.from_pretrained(args.model)
    model.eval()

    with torch.no_grad():
        text_inputs = processor(text=prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        text_features = model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    path_root = Path(args.path_root) if args.path_root else Path(".")
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    total = 0

    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "t_s", "clip"])
        writer.writeheader()
        for row in read_videos(Path(args.videos_csv)):
            video_id = row.get("video_id") or row.get("id")
            raw_path = row.get("path")
            if not video_id:
                raise SystemExit("videos_csv must include video_id or id")
            if not raw_path:
                continue
            video_path = Path(raw_path)
            if not video_path.is_absolute():
                video_path = path_root / video_path

            batch_images = []
            batch_times = []
            for t_s, image in iter_frames(video_path, args.sample_fps):
                batch_images.append(image)
                batch_times.append(t_s)
                if len(batch_images) >= args.batch_size:
                    total += write_batch(writer, video_id, batch_times, batch_images, processor, model, text_features, device)
                    batch_images = []
                    batch_times = []
            if batch_images:
                total += write_batch(writer, video_id, batch_times, batch_images, processor, model, text_features, device)
    print(f"Wrote {output} with {total} rows")


def write_batch(writer, video_id, times, images, processor, model, text_features, device) -> int:
    import torch  # type: ignore

    with torch.no_grad():
        image_inputs = processor(images=images, return_tensors="pt").to(device)
        image_features = model.get_image_features(**image_inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        scores = image_features @ text_features.T
        max_scores = scores.max(dim=1).values.detach().cpu().tolist()
    for t_s, score in zip(times, max_scores):
        writer.writerow({"video_id": video_id, "t_s": f"{t_s:.3f}", "clip": f"{float(score):.6f}"})
    return len(times)


if __name__ == "__main__":
    main()
