# Completion Audit

작성일: 2026-06-30

## 원 목표

ICLR primary, CVPR strong secondary 전략으로 paper를 준비한다.

1. 모든 실험이 완료됐다는 가정 하의 answer-first 완성형 paper draft.
2. 필요한 benchmark dataset, 실험, GPU-hour 계획 명확화.
3. 바로 실험을 돌릴 수 있는 code suite 준비.
4. GitHub로 관리하고, 연산 서버에서 clone 후 실험 가능하게 준비.

## Evidence

### 1. Answer-First Paper Draft

상태: 완료

증거:

- `paper/iclr2027/main.tex`
- `paper/iclr2027/sections/*.tex`
- `paper/iclr2027/assumed_results.json`
- generated tables:
  - `paper/iclr2027/tables/main_results.tex`
  - `paper/iclr2027/tables/ablation.tex`
  - `paper/iclr2027/tables/benchmark_comparison.tex`
- generated figures:
  - `paper/iclr2027/figures/architecture_tikz.tex`
  - `paper/iclr2027/figures/frontier_tikz.tex`
  - `paper/iclr2027/figures/ttd_cdf_tikz.tex`
  - `paper/iclr2027/figures/stream_scaling_tikz.tex`
  - `paper/iclr2027/figures/timeline_tikz.tex`
- PDF build verified with `tectonic main.tex`.

주의:

- 결과 수치는 answer-first assumed numbers이며, 실제 실험 후 `scripts/update_paper_results_from_grid.py`로 갱신해야 한다.

### 2. Experiment Plan

상태: 완료

증거:

- `docs/EXPERIMENT_PLAN_KO.md`
- `docs/VENUE_STRATEGY_KO.md`
- `configs/experiments/core_grid.json`
- `configs/experiments/ablation_grid.json`
- `configs/datasets/ucf_crime.json`
- `configs/datasets/xd_violence.json`

명시된 핵심 데이터셋:

- UCF-Crime-Multistream
- XD-Violence-Multistream
- ShanghaiTech / Street Scene stress track
- Optional semantic track: Ego4D / Video-MME style long-video resources

GPU-hour estimate:

- A100 80GB: 150-275 GPU-hour for core paper.
- RTX 4090: 260-460 GPU-hour for core paper.

### 3. Runnable Code Suite

상태: 완료

증거:

- package core: `src/bmvm/`
- synthetic smoke: `scripts/make_synthetic_manifest.py`
- CSV manifest path: `scripts/build_manifest_from_csv.py`
- dataset multiplexing: `scripts/multiplex_dataset.py`
- feature cache:
  - `scripts/extract_source_motion.py`
  - `scripts/extract_clip_scores.py`
  - `scripts/merge_signal_csvs.py`
- VLM query/cache:
  - `scripts/build_vlm_query_pool.py`
  - `scripts/run_clip_verifier_cache.py`
  - `scripts/build_vlm_cache_from_predictions.py`
  - `scripts/make_oracle_vlm_cache.py`
- experiments:
  - `scripts/run_experiment.py`
  - `scripts/run_grid.py`
  - `scripts/summarize_results.py`
- paper update:
  - `scripts/update_paper_results_from_grid.py`
  - `scripts/make_paper_assets.py`
- server docs:
  - `docs/SERVER_RUNBOOK_KO.md`
  - `docs/MULTIPLEXING_KO.md`
  - `docs/VLM_CACHE_KO.md`
- CI:
  - `.github/workflows/tests.yml`
- SLURM:
  - `scripts/slurm/run_core_grid.sbatch`

Verified commands:

```bash
python3 -m unittest discover -s tests
PYTHONPYCACHEPREFIX=/Users/westlee/2026_research/budgeted-vlm-monitoring/.pycache_tmp python3 -m py_compile ...
python3 scripts/make_paper_assets.py
cd paper/iclr2027 && tectonic main.tex
```

### 4. Git/GitHub Management

상태: 부분 완료

완료:

- local git repository initialized.
- source/docs/tests/paper source committed.
- generated/raw artifacts ignored.
- current branch: `main`.

Commits:

- `4f083e6 Initialize budgeted VLM monitoring paper and benchmark suite`
- `f8edab6 Document GitHub remote setup`
- `0fea30e Add real-data grid execution and VLM cache pipeline`
- `08aef7f Add paper scaling figures and feature extraction tools`
- `ada2bb3 Add result-to-paper and verifier cache adapters`

미완료:

- GitHub remote creation and push.

Reason:

- GitHub remote creation/push is an external upload. It requires explicit user approval.

Ready command after approval:

```bash
gh repo create budgeted-vlm-monitoring \
  --private \
  --source=. \
  --remote=origin \
  --push
```

## Current Verification Summary

상태:

- Tests: 12 passed.
- PDF: built successfully.
- Git worktree: clean except ignored generated artifacts.
- Remaining blocker: explicit approval for GitHub remote creation/push.
