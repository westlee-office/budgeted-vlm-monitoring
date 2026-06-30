#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List


def read_prompts(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def prompt_hash(prompts: List[str]) -> str:
    return hashlib.sha256("\n".join(prompts).encode("utf-8")).hexdigest()[:16]


def read_query_pool(path: Path) -> List[Dict[str, object]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_frame(path: Path, t_s: float):
    try:
        import cv2  # type: ignore
        from PIL import Image  # type: ignore
    except ImportError as exc:
        raise SystemExit("OpenCV and Pillow are required: pip install opencv-python pillow") from exc

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return None
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = max(0, int(round(t_s * fps)))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        return None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a BMVM verifier cache from CLIP prompt similarity.")
    parser.add_argument("--query-pool", required=True)
    parser.add_argument("--prompts", default="configs/prompts/incidents.txt")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="openai/clip-vit-base-patch32")
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
    queries = read_query_pool(Path(args.query_pool))
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
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    batch_images = []
    batch_queries = []
    written = 0
    missing = 0
    hash_value = prompt_hash(prompts)
    with output.open("w", encoding="utf-8") as f:
        for query in queries:
            raw_path = str(query.get("path") or "")
            if not raw_path:
                missing += 1
                continue
            video_path = Path(raw_path)
            if not video_path.is_absolute():
                video_path = path_root / video_path
            image = load_frame(video_path, float(query["t_s"]))
            if image is None:
                missing += 1
                continue
            batch_images.append(image)
            batch_queries.append(query)
            if len(batch_images) >= args.batch_size:
                written += write_batch(f, batch_queries, batch_images, processor, model, text_features, prompts, device, args.model, hash_value)
                batch_images = []
                batch_queries = []
        if batch_images:
            written += write_batch(f, batch_queries, batch_images, processor, model, text_features, prompts, device, args.model, hash_value)
    print(f"Wrote {output} with {written} records; skipped {missing} queries")


def write_batch(handle, queries, images, processor, model, text_features, prompts, device, model_name, prompt_hash_value) -> int:
    import torch  # type: ignore

    with torch.no_grad():
        image_inputs = processor(images=images, return_tensors="pt").to(device)
        image_features = model.get_image_features(**image_inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        scores = image_features @ text_features.T
        best_scores, best_indices = scores.max(dim=1)
        best_scores = best_scores.detach().cpu().tolist()
        best_indices = best_indices.detach().cpu().tolist()
    for query, score, idx in zip(queries, best_scores, best_indices):
        normalized = max(0.0, min((float(score) + 1.0) / 2.0, 1.0))
        handle.write(
            json.dumps(
                {
                    "episode_id": query["episode_id"],
                    "stream_id": query["stream_id"],
                    "t_s": query["t_s"],
                    "score": normalized,
                    "summary": prompts[int(idx)],
                    "model": model_name,
                    "prompt_hash": prompt_hash_value,
                },
                sort_keys=True,
            )
            + "\n"
        )
    return len(queries)


if __name__ == "__main__":
    main()
