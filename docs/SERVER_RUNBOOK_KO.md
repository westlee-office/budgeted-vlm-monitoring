# 서버 실행 Runbook

## 1. Repo 준비

```bash
git clone <repo-url> budgeted-vlm-monitoring
cd budgeted-vlm-monitoring
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

GPU feature extraction을 돌릴 서버에서는 optional dependency를 설치한다.

```bash
pip install -e ".[vision]"
```

## 2. CPU Smoke Test

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
python3 -m unittest discover -s tests
```

## 3. 실제 데이터 위치

권장 디렉터리:

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

`data/raw`, `data/features`, `results`는 git에 올리지 않는다.

## 4. 실행 순서

1. 원본 데이터 다운로드 및 라이선스 확인.
2. dataset-specific annotation을 공통 `videos.csv`, `events.csv`로 변환.
3. cheap feature cache를 `signals.csv`로 생성.
4. `scripts/validate_dataset_csvs.py`로 source CSV 무결성 검증.
5. `scripts/multiplex_dataset.py`로 BMVM manifest 생성.
6. `scripts/build_vlm_query_pool.py`로 VLM batch candidate 생성.
7. VLM verifier batch job으로 `data/vlm_cache/<dataset>_<streams>.jsonl` 생성.
8. `scripts/validate_vlm_cache.py`로 query pool/cache coverage 검증.
9. `scripts/run_grid.py`로 policies x streams x budgets x seeds 실험 실행.
10. `scripts/update_paper_results_from_grid.py`로 실제 결과를 paper JSON에 반영.
11. `python3 scripts/make_paper_assets.py`.
12. `cd paper/iclr2027 && tectonic main.tex`.

## 4.1 Manifest Multiplexing

Cheap feature cache:

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
```

CSV validator:

```bash
python3 scripts/validate_dataset_csvs.py \
  --videos data/manifests/ucf_videos.csv \
  --events data/manifests/ucf_events.csv \
  --signals data/features/ucf_signals.csv \
  --path-root . \
  --check-paths \
  --output data/manifests/ucf_validation_report.json
```

```bash
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

## 4.2 Grid Runner

VLM query pool:

```bash
python3 scripts/build_vlm_query_pool.py \
  --manifest data/manifests/ucf_crime_multistream_128.json \
  --output data/vlm_cache/ucf_query_pool.jsonl \
  --mode policies \
  --policies anomaly_topk,clip_topk,voi \
  --query-budget 4
```

Diagnostic CLIP verifier cache:

```bash
python3 scripts/run_clip_verifier_cache.py \
  --query-pool data/vlm_cache/ucf_query_pool.jsonl \
  --prompts configs/prompts/incidents.txt \
  --output data/vlm_cache/ucf_crime_multistream.jsonl \
  --device cuda \
  --batch-size 32
```

외부 VLM batch job 결과를 BMVM cache로 변환:

```bash
python3 scripts/build_vlm_cache_from_predictions.py \
  --query-pool data/vlm_cache/ucf_query_pool.jsonl \
  --predictions data/vlm_cache/ucf_vlm_predictions.jsonl \
  --output data/vlm_cache/ucf_crime_multistream_128.jsonl \
  --model qwen2.5-vl-7b
```

Cache validator:

```bash
python3 scripts/validate_vlm_cache.py \
  --query-pool data/vlm_cache/ucf_query_pool.jsonl \
  --cache data/vlm_cache/ucf_crime_multistream_128.jsonl \
  --require-provenance \
  --output data/vlm_cache/ucf_cache_validation_report.json
```

```bash
python3 scripts/run_grid.py \
  --config configs/experiments/core_grid.json \
  --manifest-dir data/manifests \
  --output-dir results/grid
```

VLM cache를 강제하는 실제 실험:

```bash
python3 scripts/run_grid.py \
  --config configs/experiments/core_grid.json \
  --manifest-dir data/manifests \
  --vlm-cache-dir data/vlm_cache \
  --no-simulated-vlm-fallback \
  --stream-counts 128 \
  --output-dir results/grid
```

Grid 결과를 paper JSON으로 반영:

```bash
python3 scripts/update_paper_results_from_grid.py \
  --aggregate results/grid/aggregate.json \
  --template paper/iclr2027/assumed_results.json \
  --output paper/iclr2027/assumed_results.json \
  --main-budget 4 \
  --write
```

SLURM cluster:

```bash
sbatch scripts/slurm/run_core_grid.sbatch
```

Smoke용 SLURM run은 simulated fallback을 명시적으로 켠다.

```bash
SIM_FALLBACK=1 LIMIT=4 sbatch scripts/slurm/run_core_grid.sbatch
```

실제 H200/A100 partition이나 constraint는 cluster 정책에 맞춰 `sbatch` 옵션으로 넘긴다.

```bash
sbatch -p h200 --gres=gpu:h200:1 scripts/slurm/run_core_grid.sbatch
sbatch -p a100 --gres=gpu:a100:1 scripts/slurm/run_core_grid.sbatch
```

## 5. 결과 관리 규칙

각 실험 결과에는 다음 metadata를 저장한다.

- git commit hash.
- dataset manifest hash.
- policy name and parameters.
- VLM model/checkpoint.
- prompt template hash.
- frame sampling rate.
- budget setting.
- VLM cache path/hash.
- GPU type and measured runtime.

## 6. 서버에서 가장 먼저 확인할 명령

```bash
nvidia-smi
python3 --version
ffmpeg -version
git rev-parse HEAD  # ZIP이면 생략하고 run_grid.py --source-commit 사용
python3 -m unittest discover -s tests
```
