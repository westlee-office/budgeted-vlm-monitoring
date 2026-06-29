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
results/
```

`data/raw`, `data/features`, `results`는 git에 올리지 않는다.

## 4. 실행 순서

1. 원본 데이터 다운로드 및 라이선스 확인.
2. dataset-specific converter로 manifest 생성.
3. cheap feature cache 생성.
4. VLM candidate pool 생성.
5. VLM verifier cache 생성.
6. policies x budgets x seeds 실험 실행.
7. `assumed_results.json`을 실제 결과로 교체.
8. `python3 scripts/make_paper_assets.py`.
9. `cd paper/iclr2027 && tectonic main.tex`.

## 5. 결과 관리 규칙

각 실험 결과에는 다음 metadata를 저장한다.

- git commit hash.
- dataset manifest hash.
- policy name and parameters.
- VLM model/checkpoint.
- prompt template hash.
- frame sampling rate.
- budget setting.
- GPU type and measured runtime.

## 6. 서버에서 가장 먼저 확인할 명령

```bash
nvidia-smi
python3 --version
ffmpeg -version
git rev-parse HEAD
python3 -m unittest discover -s tests
```
