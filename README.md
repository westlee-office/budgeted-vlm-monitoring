# Budgeted VLM Monitoring

This repository contains the paper workspace and runnable code suite for:

**Seeing Less, Monitoring More: Budgeted Vision-Language Agents for Multi-Stream Video Monitoring**

Target strategy:

- **Primary:** ICLR 2027
- **Strong secondary:** CVPR 2027

The core research question is:

> How should a multimodal agent decide what to look at, when visual perception itself is expensive?

## What Is Here

- `AGENTS.md`: first-read handoff guide for Codex/agent sessions on an experiment server.
- `paper/iclr2027/`: answer-first paper draft written as if the full experiment table is available.
- `docs/EXPERIMENT_PLAN_KO.md`: dataset, benchmark, experiment, and GPU-hour plan.
- `docs/SERVER_RUNBOOK_KO.md`: commands for running the suite on a GPU server.
- `src/bmvm/`: simulator, metrics, budget accounting, and baseline/selective policies.
- `scripts/`: manifest generation, experiment runner, result summarizer, paper asset generator.
- `configs/`: dataset and experiment configuration templates.
- `tests/`: dependency-free unit tests for the benchmark core.

The paper draft intentionally uses assumed answer-first numbers. They are clearly separated in `paper/iclr2027/assumed_results.json` and generated into the LaTeX tables/figures. Replace that JSON with actual experiment outputs once the GPU runs finish.

## Quick Start

Run a synthetic smoke test without installing any dependencies:

```bash
python3 scripts/make_synthetic_manifest.py \
  --output data/synthetic/smoke.json \
  --episodes 8 \
  --streams 32 \
  --horizon-s 300 \
  --events-per-episode 8

python3 scripts/run_experiment.py \
  --manifest data/synthetic/smoke.json \
  --policies random,uniform,motion_topk,anomaly_topk,clip_topk,voi \
  --query-budget 4 \
  --output results/smoke_results.json

python3 scripts/summarize_results.py results/smoke_results.json --format md
```

Run tests:

```bash
python3 -m unittest discover -s tests
```

Build a real multi-stream manifest from normalized dataset CSV files:

```bash
python3 scripts/extract_source_motion.py \
  --videos-csv data/manifests/ucf_videos.csv \
  --output-csv data/features/ucf_motion.csv \
  --sample-fps 1

python3 scripts/extract_clip_scores.py \
  --videos-csv data/manifests/ucf_videos.csv \
  --prompts configs/prompts/incidents.txt \
  --output-csv data/features/ucf_clip.csv \
  --sample-fps 1

python3 scripts/merge_signal_csvs.py \
  --inputs data/features/ucf_motion.csv data/features/ucf_clip.csv \
  --output-csv data/features/ucf_signals.csv \
  --round-timestep 1

python3 scripts/multiplex_dataset.py \
  --videos-csv data/manifests/ucf_videos.csv \
  --events-csv data/manifests/ucf_events.csv \
  --signals-csv data/features/ucf_signals.csv \
  --output data/manifests/ucf_crime_multistream_128.json \
  --episodes 64 \
  --streams 128 \
  --horizon-s 1800 \
  --step-s 2
```

Run a grid over datasets, policies, budgets, and seeds:

```bash
python3 scripts/run_grid.py \
  --config configs/experiments/core_grid.json \
  --manifest-dir data/manifests \
  --output-dir results/grid
```

Run with cached VLM verifier outputs:

```bash
python3 scripts/run_grid.py \
  --config configs/experiments/core_grid.json \
  --manifest-dir data/manifests \
  --vlm-cache-dir data/vlm_cache \
  --no-simulated-vlm-fallback \
  --output-dir results/grid
```

Update the paper result JSON from completed grid output:

```bash
python3 scripts/update_paper_results_from_grid.py \
  --aggregate results/grid/aggregate.json \
  --template paper/iclr2027/assumed_results.json \
  --output paper/iclr2027/assumed_results.json \
  --main-budget 4 \
  --write
python3 scripts/make_paper_assets.py
```

Regenerate paper tables and figures:

```bash
python3 scripts/make_paper_assets.py
```

Build the paper if `tectonic` is installed:

```bash
cd paper/iclr2027
tectonic main.tex
```

## Real Dataset Integration

The runnable core expects a JSON manifest with episodes, streams, and event intervals. Dataset-specific preprocessing should map UCF-Crime, XD-Violence, ShanghaiTech, Street Scene, and optional Ego4D/Video-MME style metadata into normalized CSV files, then `scripts/multiplex_dataset.py` converts them into BMVM manifests. The manifest contract is documented in `docs/EXPERIMENT_PLAN_KO.md`, the multiplexing path in `docs/MULTIPLEXING_KO.md`, and the cached VLM verifier interface in `docs/VLM_CACHE_KO.md`.

Raw datasets and extracted features are intentionally ignored by git.
