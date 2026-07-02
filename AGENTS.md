# Agent Handoff Guide

This file is the first-read guide for Codex or any other coding agent running on a fresh experiment server.

## Project Snapshot

- Project: Budgeted VLM Monitoring
- Paper: `Seeing Less, Monitoring More: Budgeted Vision-Language Agents for Multi-Stream Video Monitoring`
- Target venue strategy: ICLR 2027 primary, CVPR 2027 strong secondary
- Public repo: `https://github.com/westlee-office/budgeted-vlm-monitoring`
- ZIP download: `https://github.com/westlee-office/budgeted-vlm-monitoring/archive/refs/heads/main.zip`
- Core question: how should a multimodal agent decide what to look at when visual perception itself is expensive?

The current paper is an answer-first draft. The numbers in `paper/iclr2027/assumed_results.json` are placeholders for the intended empirical story. Do not treat them as real experimental results.

## Current State

The repo contains:

- `paper/iclr2027/`: LaTeX paper draft, generated tables, and generated TikZ figures.
- `src/bmvm/`: benchmark simulator, policies, VLM-cache interface, metrics, and evaluation loop.
- `scripts/`: synthetic data generation, real-data multiplexing, feature extraction, VLM cache adapters, grid runner, result summarizer, and paper updater.
- `configs/experiments/`: smoke/core/ablation grid configs.
- `configs/datasets/`: dataset config templates.
- `docs/`: Korean planning docs, runbooks, schema docs, and status reviews.
- `tests/`: dependency-light unit tests.

Known limitations:

- UCF-Crime/XD-Violence dataset-specific raw annotation converters are not implemented yet.
- Real Qwen/LLaVA video VLM backend runner is not implemented yet; the current path supports cached verifier outputs and external prediction adapters.
- `simulated_vlm_fallback=True` exists for smoke tests. Real experiments must use `--no-simulated-vlm-fallback`.
- Real-data runs should pass `scripts/validate_dataset_csvs.py` before multiplexing and `scripts/validate_vlm_cache.py` before `run_grid.py`.
- The paper is not yet in official ICLR/CVPR style.

## First Commands On A Server

If git authentication is blocked, download the public ZIP:

```bash
curl -L -o budgeted-vlm-monitoring.zip \
  https://github.com/westlee-office/budgeted-vlm-monitoring/archive/refs/heads/main.zip

unzip budgeted-vlm-monitoring.zip
cd budgeted-vlm-monitoring-main
```

If git clone is allowed:

```bash
git clone https://github.com/westlee-office/budgeted-vlm-monitoring.git
cd budgeted-vlm-monitoring
```

Set up Python:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

For GPU feature extraction and CLIP/VLM-adjacent utilities:

```bash
pip install -e ".[vision]"
```

Check environment:

```bash
nvidia-smi
python3 --version
ffmpeg -version
python3 -m unittest discover -s tests
```

## Quick Smoke Test

Run this before touching real data:

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

Expected outcome: the command finishes and prints event-recall, time-to-detect, false-alarm, and cost metrics for several policies.

## Data Layout

Keep raw data, features, VLM caches, and results out of git.

Recommended layout:

```text
data/raw/ucf_crime/
data/raw/xd_violence/
data/raw/shanghaitech/
data/raw/street_scene/
data/features/
data/manifests/
data/vlm_cache/
results/
```

The normalized CSV contracts are documented in `docs/CSV_SCHEMA.md`, `docs/MULTIPLEXING_KO.md`, and `src/bmvm/data/manifest_schema.md`.

## Real Experiment Flow

The intended real-data pipeline is:

1. Download source datasets and verify license/access constraints.
2. Convert source annotations into normalized `videos.csv` and `events.csv`.
3. Extract cheap signals into `signals.csv`.
4. Multiplex source videos into BMVM multi-stream manifests.
5. Build a VLM query pool.
6. Generate a VLM verifier cache through either a real VLM backend or external predictions.
7. Run grid experiments with simulated fallback disabled.
8. Summarize grid output.
9. Update paper result JSON from completed grid output.
10. Regenerate paper assets and rebuild the paper.

Core commands:

```bash
python3 scripts/extract_source_motion.py \
  --videos-csv data/manifests/ucf_videos.csv \
  --output-csv data/features/ucf_motion.csv \
  --sample-fps 1 \
  --path-root .

python3 scripts/extract_clip_scores.py \
  --videos-csv data/manifests/ucf_videos.csv \
  --prompts configs/prompts/incidents.txt \
  --output-csv data/features/ucf_clip.csv \
  --sample-fps 1 \
  --batch-size 32 \
  --device cuda

python3 scripts/merge_signal_csvs.py \
  --inputs data/features/ucf_motion.csv data/features/ucf_clip.csv \
  --output-csv data/features/ucf_signals.csv \
  --round-timestep 1

python3 scripts/validate_dataset_csvs.py \
  --videos data/manifests/ucf_videos.csv \
  --events data/manifests/ucf_events.csv \
  --signals data/features/ucf_signals.csv \
  --path-root . \
  --check-paths \
  --output data/manifests/ucf_validation_report.json

python3 scripts/multiplex_dataset.py \
  --videos-csv data/manifests/ucf_videos.csv \
  --events-csv data/manifests/ucf_events.csv \
  --signals-csv data/features/ucf_signals.csv \
  --output data/manifests/ucf_crime_multistream_128.json \
  --episodes 64 \
  --streams 128 \
  --horizon-s 1800 \
  --step-s 2 \
  --event-streams-per-episode 12 \
  --seed 7
```

Build query pool and VLM cache:

```bash
python3 scripts/build_vlm_query_pool.py \
  --manifest data/manifests/ucf_crime_multistream_128.json \
  --output data/vlm_cache/ucf_query_pool.jsonl \
  --mode policies \
  --policies anomaly_topk,clip_topk,voi \
  --query-budget 4

python3 scripts/build_vlm_cache_from_predictions.py \
  --query-pool data/vlm_cache/ucf_query_pool.jsonl \
  --predictions data/vlm_cache/ucf_vlm_predictions.jsonl \
  --output data/vlm_cache/ucf_crime_multistream_128.jsonl \
  --model qwen2.5-vl-7b

python3 scripts/validate_vlm_cache.py \
  --query-pool data/vlm_cache/ucf_query_pool.jsonl \
  --cache data/vlm_cache/ucf_crime_multistream_128.jsonl \
  --require-provenance \
  --output data/vlm_cache/ucf_cache_validation_report.json
```

Run real grid:

```bash
python3 scripts/run_grid.py \
  --config configs/experiments/core_grid.json \
  --manifest-dir data/manifests \
  --vlm-cache-dir data/vlm_cache \
  --no-simulated-vlm-fallback \
  --stream-counts 128 \
  --output-dir results/grid
```

For SLURM:

```bash
sbatch scripts/slurm/run_core_grid.sbatch
```

## Updating The Paper After Real Runs

Only do this after the grid results are real and audited:

```bash
python3 scripts/update_paper_results_from_grid.py \
  --aggregate results/grid/aggregate.json \
  --template paper/iclr2027/assumed_results.json \
  --output paper/iclr2027/assumed_results.json \
  --main-budget 4 \
  --write

python3 scripts/make_paper_assets.py
cd paper/iclr2027
tectonic main.tex
```

The current generated PDF is ignored by git. Source `.tex`, `.bib`, JSON, tables, and TikZ assets are tracked.

## Priority Work For The Next Agent

Recommended order:

1. Verify server setup with unit tests and synthetic smoke test.
2. Implement dataset-specific converters:
   - `scripts/convert_ucf_crime.py`
   - `scripts/convert_xd_violence.py`
3. Validate normalized CSV output with `scripts/validate_dataset_csvs.py`:
   - video count
   - event count
   - duration histogram
   - class histogram
   - train/test split leakage
4. Run cheap feature extraction for a small subset.
5. Generate a small real-data multiplexed manifest and run a real smoke grid.
6. Build or adapt a real VLM verifier cache path.
7. Validate VLM cache coverage with `scripts/validate_vlm_cache.py`.
8. Run full core grid with `--no-simulated-vlm-fallback`.
9. Use stream-count-specific manifests/caches such as `ucf_crime_multistream_32.json` and `ucf_crime_multistream_32.jsonl`.
10. Replace answer-first numbers in the paper only after result provenance is recorded.

## Do Not Do

- Do not commit raw datasets, extracted features, VLM caches, or `results/`.
- Do not use simulated VLM fallback for paper-quality real experiments.
- Do not overwrite `paper/iclr2027/assumed_results.json` with partial or unaudited results unless the output path is explicitly marked as draft.
- Do not report assumed paper numbers as actual experimental results.
- Do not change unrelated paper text or generated figure files by hand if the generator can be updated instead.

## Important References Inside The Repo

- `README.md`: overview and quick commands.
- `docs/SERVER_RUNBOOK_KO.md`: detailed server execution commands.
- `docs/ARTIFACT_STATUS_REVIEW_KO.md`: current artifact status and top-tier gaps.
- `docs/EXPERIMENT_PLAN_KO.md`: dataset, experiment, and GPU-hour plan.
- `docs/VLM_CACHE_KO.md`: cached verifier format.
- `docs/MULTIPLEXING_KO.md`: dataset-to-manifest flow.
- `docs/CSV_SCHEMA.md`: normalized CSV schema.

## Verification Checklist Before Reporting Progress

Run at least:

```bash
python3 -m unittest discover -s tests
python3 scripts/make_paper_assets.py
```

If paper files changed and `tectonic` is installed:

```bash
cd paper/iclr2027 && tectonic main.tex
```

When reporting results, include:

- command run
- git commit hash
- dataset manifest path/hash
- VLM cache path/hash
- whether simulated fallback was disabled
- GPU type
- output directory
