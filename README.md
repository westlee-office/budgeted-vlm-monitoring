# Budgeted VLM Monitoring

This repository contains the paper workspace and runnable code suite for:

**Seeing Less, Monitoring More: Budgeted Vision-Language Agents for Multi-Stream Video Monitoring**

Target strategy:

- **Primary:** ICLR 2027
- **Strong secondary:** CVPR 2027

The core research question is:

> How should a multimodal agent decide what to look at, when visual perception itself is expensive?

## What Is Here

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

The runnable core expects a JSON manifest with episodes, streams, and event intervals. Dataset-specific converters should map UCF-Crime, XD-Violence, ShanghaiTech, Street Scene, and optional Ego4D/Video-MME style metadata into this manifest. The manifest contract is documented in `docs/EXPERIMENT_PLAN_KO.md`.

Raw datasets and extracted features are intentionally ignored by git.
