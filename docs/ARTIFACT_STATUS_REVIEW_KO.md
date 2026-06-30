# Artifact Status Review

작성일: 2026-06-30

대상 repo: `westlee-office/budgeted-vlm-monitoring`  
로컬 경로: `/Users/westlee/2026_research/budgeted-vlm-monitoring`

## 1. 현재 상태 요약

### Git / GitHub

상태: 양호

- 원격 repo: `https://github.com/westlee-office/budgeted-vlm-monitoring`
- visibility: `PRIVATE`
- branch: `main`
- local `main` tracks `origin/main`
- 현재 local tracked 변경 없음
- ignored artifact:
  - `.DS_Store`
  - `.pycache_tmp/`
  - `data/`
  - `results/`
  - generated `paper/iclr2027/main.pdf`

주의:

- ignored artifact가 로컬에 남아 있다. repo에는 올라가지 않지만, 깨끗한 재현 환경을 위해 `make clean` 또는 cleanup script가 있으면 좋다.

### Verification

상태: 양호

최근 확인:

```bash
python3 -m unittest discover -s tests
```

결과:

- 12 tests OK

```bash
PYTHONPYCACHEPREFIX=/Users/westlee/2026_research/budgeted-vlm-monitoring/.pycache_tmp python3 -m py_compile ...
```

결과:

- Python 3.9.6 기준 compile OK

```bash
python3 scripts/make_paper_assets.py
cd paper/iclr2027 && tectonic main.tex
```

결과:

- PDF 생성 OK
- `tectonic`의 BibTeX 재실행 관련 internal consistency warning은 남아 있음
- TeX fatal error는 없음

### Paper PDF

상태: 초안으로는 양호, 제출용으로는 미완

- PDF: `paper/iclr2027/main.pdf`
- page count: 14
- extracted word count: 약 5,828 words
- cited references: 64
- 구성:
  - abstract
  - introduction
  - related work
  - problem formulation
  - benchmark
  - method
  - experiments
  - analysis
  - limitations/ethics/reproducibility
- 생성된 표/그림:
  - architecture
  - benchmark comparison
  - main results
  - cost-performance frontier
  - time-to-detect CDF
  - stream-count scaling
  - qualitative timeline
  - ablation table
  - dataset-track protocol table
  - experimental setup table
  - per-dataset result breakdown table
  - failure taxonomy table

주의:

- 본문은 assumed result를 실제 결과처럼 서술한다. answer-first draft 목적에는 맞지만, 실제 제출 전에는 반드시 숫자와 표현을 실험 결과로 교체해야 한다.
- ICLR/CVPR 공식 style file이 아직 적용되지 않았다.
- answer-first depth와 reference coverage는 이전 9-page draft 대비 강화되었지만, top-tier 제출용으로는 실제 결과 provenance, appendix, official style, learned-policy evidence가 아직 부족하다.

### Experiment Plan

상태: 방향성은 좋고, 실행 계획은 충분히 구체적이나 데이터셋별 실제 annotation converter가 미완

문서:

- `docs/EXPERIMENT_PLAN_KO.md`
- `docs/SERVER_RUNBOOK_KO.md`
- `docs/MULTIPLEXING_KO.md`
- `docs/VLM_CACHE_KO.md`

명시된 core tracks:

- UCF-Crime-Multistream
- XD-Violence-Multistream
- ShanghaiTech / Street Scene stress track
- optional Ego4D / Video-MME semantic track

GPU-hour estimate:

- A100 80GB: 150-275 GPU-hour
- RTX 4090: 260-460 GPU-hour

주의:

- UCF-Crime/XD-Violence 실제 annotation 파일을 `videos.csv`, `events.csv`로 바꾸는 dataset-specific converter는 아직 없다.
- XD-Violence audio signal은 계획에는 있지만 코드에는 아직 feature path가 없다.
- benchmark split leakage 방지, video-level grouping, episode sampling reproducibility가 더 엄격히 문서화되어야 한다.

### Code Suite

상태: runnable skeleton으로는 양호, 실제 논문 실험용으로는 adapter와 backend 보강 필요

구성:

- core package: `src/bmvm/`
- policy/evaluation:
  - random
  - uniform
  - motion top-k
  - anomaly top-k
  - CLIP top-k
  - dense VLM reference
  - VOI/TriageVLM variants
- manifest:
  - synthetic generator
  - CSV manifest builder
  - multiplex builder
- feature/cache:
  - source motion extraction
  - CLIP score extraction
  - signal merge
  - VLM query pool
  - CLIP verifier cache
  - external prediction adapter
  - oracle debug cache
- experiment:
  - single run
  - grid run
  - result summarizer
  - paper result updater
- infra:
  - GitHub Actions unit test
  - SLURM example

주의:

- tests는 synthetic/smoke 중심이다.
- real video feature extraction은 optional dependency와 data availability가 필요해서 CI에서 검증되지 않는다.
- VLM verifier는 cache adapter 중심이며, Qwen/LLaVA video backend를 직접 실행하는 구현은 아직 없다.
- evaluation의 `simulated_vlm_fallback=True` 기본값은 실험자가 실수하면 synthetic verifier 결과를 실제 결과처럼 만들 위험이 있다.

## 2. Top-Tier 관점 주요 리스크

### P0. Assumed Results가 실제 결과로 교체되지 않음

위험:

- 현재 paper 수치는 answer-first 가정값이다.
- 제출 전 실제 UCF/XD/Stress track 결과로 교체하지 않으면 논문 신뢰성이 없다.

개선:

- full grid run 완료 후:

```bash
python3 scripts/update_paper_results_from_grid.py \
  --aggregate results/grid/aggregate.json \
  --template paper/iclr2027/assumed_results.json \
  --output paper/iclr2027/assumed_results.json \
  --main-budget 4 \
  --write
python3 scripts/make_paper_assets.py
```

- paper 본문에서 “we show”류 문장을 실제 결과 검증 이후에만 유지한다.
- interim draft에는 “answer-first placeholder” watermark 또는 comment를 둘지 결정한다.

### P0. 실제 데이터셋 converter 부재

위험:

- `multiplex_dataset.py`는 공통 CSV를 받지만 UCF-Crime/XD-Violence 원 annotation을 직접 읽지는 않는다.
- 서버 실험자가 원 데이터 포맷을 보고 수동 CSV를 만들어야 한다.

개선:

- `scripts/convert_ucf_crime.py`
  - raw video index 생성
  - temporal annotation parsing
  - train/test split preservation
  - normal/background labeling
- `scripts/convert_xd_violence.py`
  - video metadata parsing
  - weak/test temporal annotation parsing
  - optional audio path linking
- converter output validation:
  - video count
  - event count
  - class histogram
  - duration histogram
  - split leakage check

### P0. 실제 VLM backend 부재

위험:

- 현재 실제 VLM은 external prediction adapter로만 연결된다.
- 논문 결과를 재현하려면 Qwen2.5-VL/LLaVA-NeXT-Video 등 backend runner가 필요하다.

개선:

- `scripts/run_qwen_vl_cache.py` 또는 generic HuggingFace VLM runner 추가
- query pool에서 frame/clip extraction
- prompt template versioning
- output score calibration
- batch retry/checkpoint resume
- model metadata 저장:
  - checkpoint
  - prompt hash
  - decoding config
  - frame sampling
  - GPU type

### P0. Method가 heuristic으로 보일 위험

위험:

- ICLR에서는 fixed-weight VOI scheduler가 “engineering heuristic”으로 보일 수 있다.
- top-tier contribution이 benchmark만으로 보이면 reject risk가 커진다.

개선:

- learned policy variant 추가:
  - logistic value model
  - contextual bandit
  - offline imitation from dense oracle
  - budgeted knapsack/ranking objective
- theoretical/algorithmic framing 강화:
  - marginal value of information
  - submodular/knapsack-style scheduling
  - uncertainty-calibrated active perception
- ablation에 fixed vs learned weights 추가.

### P1. Benchmark validity와 fairness protocol 부족

위험:

- multiplexing이 artificial하다는 비판 가능.
- dense VLM oracle와 selective policy의 candidate pool/cache 조건이 다르면 unfair할 수 있다.

개선:

- episode construction protocol 강화:
  - video-level split
  - background/event stream sampling rule
  - class balancing
  - hard distractor sampling
  - stream-count scaling rule
- 동일 candidate/query cache에서 모든 policy를 평가하는 protocol 명시.
- per-dataset appendix:
  - source dataset license/access
  - splits
  - event interval source
  - preprocessing
  - excluded videos and reason

### P1. Metrics 구현이 아직 최소형

위험:

- Event Recall, mean TTD, FA/h, GPU-s/h, calls/event만 충분히 구현되어 있다.
- 계획 문서에 있는 median TTD, class/duration breakdown, summary correctness, calibration은 아직 코드가 약하거나 없다.

개선:

- metrics 추가:
  - median / p90 TTD
  - Recall@delay threshold
  - per-class recall
  - event-duration bucket recall
  - per-dataset aggregate
  - bootstrap confidence intervals
  - alert calibration ECE
  - summary correctness score ingestion
- table generator가 CI artifact로 metrics CSV/LaTeX를 같이 생성하도록 개선.

### P1. Paper가 아직 ICLR/CVPR 제출 포맷이 아님

위험:

- 현재는 article class 기반 14-page working draft다.
- ICLR style, anonymity checklist, ethics/reproducibility formatting, appendix structure가 없다.

개선:

- `paper/iclr2027/style/` 또는 official style 적용
- appendix:
  - dataset details
  - prompt templates
  - full hyperparameters
  - all budget sweeps
  - class breakdown
  - qualitative failures
  - social impact discussion
- CVPR fork:
  - CVPR style
  - visual qualitative examples 강화
  - dataset protocol 중심으로 재배치

### P1. CI가 paper build와 smoke grid를 커버하지 않음

위험:

- GitHub Actions는 unit test만 실행한다.
- paper assets/PDF build, smoke experiment, grid runner, result-to-paper path가 CI에서 깨져도 발견이 늦다.

개선:

- CI job 추가:
  - `python3 scripts/make_synthetic_manifest.py`
  - `python3 scripts/run_experiment.py`
  - `python3 scripts/run_grid.py --limit ...`
  - `python3 scripts/update_paper_results_from_grid.py` dry run
  - `python3 scripts/make_paper_assets.py`
- `tectonic` CI build는 optional 또는 scheduled job으로 추가.

### P1. Result provenance와 reproducibility lock 부족

위험:

- answer-first 수치는 아직 assumed JSON에서 생성된다.
- 실제 제출 시점에는 각 숫자가 어떤 grid run, commit, model checkpoint, prompt hash, dataset split에서 왔는지 추적 가능해야 한다.
- 지금 구조는 result-to-paper path는 있으나, paper table에 provenance footnote나 machine-readable run index가 붙어 있지 않다.

개선:

- aggregate result에 run metadata 필수화:
  - git commit
  - dataset config hash
  - policy config hash
  - VLM checkpoint and prompt hash
  - cache generation timestamp
- `scripts/update_paper_results_from_grid.py`가 provenance JSON과 LaTeX footnote를 같이 생성하도록 개선.
- paper appendix에 exact run table을 추가.

### P2. Repo hygiene

위험:

- ignored local artifacts가 많다.
- tests가 `data/synthetic/`에 파일을 쓴다.

개선:

- `scripts/clean_artifacts.py` 또는 `Makefile clean`
- tests는 `tempfile.TemporaryDirectory()` 사용
- `.DS_Store` 삭제
- generated PDF를 release artifact로 둘지, git ignored만 둘지 정책 명확화.

### P2. Packaging / developer UX

위험:

- `requirements.txt`, `Makefile`, `Dockerfile`, `environment.yml`이 없다.
- GPU 서버에서 dependency setup이 문서 의존적이다.

개선:

- `Makefile`:
  - `make test`
  - `make smoke`
  - `make paper`
  - `make assets`
  - `make clean`
- `requirements-vision.txt`
- Dockerfile 또는 micromamba env
- CLI entry points 추가.

## 3. 권장 다음 작업 순서

### 1순위: 실제 데이터셋 ingestion

목표:

- UCF-Crime full/near-full manifest 생성
- XD-Violence manifest 생성

산출물:

- `data/manifests/ucf_crime_multistream_128.json`
- `data/manifests/xd_violence_multistream_128.json`
- dataset validation report

### 2순위: 실제 VLM verifier cache

목표:

- Qwen2.5-VL-7B 또는 LLaVA-NeXT-Video-7B로 query pool 처리
- `--no-simulated-vlm-fallback` 실험 강제

산출물:

- `data/vlm_cache/ucf_crime_multistream.jsonl`
- `data/vlm_cache/xd_violence_multistream.jsonl`
- VLM cache validation report

### 3순위: main grid + ablation grid

목표:

- budgets 1/2/4/8
- seeds 7/17/29
- policies + ablations
- stream scaling 32/64/128/256

산출물:

- `results/grid/aggregate.json`
- `results/ablation/aggregate.json`
- updated `assumed_results.json`
- regenerated paper figures/tables

### 4순위: method 강화

목표:

- fixed VOI에서 learned VOI / contextual bandit variant로 확장
- ICLR에서 heuristic criticism 방어

산출물:

- `learned_voi` policy
- training/evaluation script
- fixed vs learned ablation

### 5순위: paper submission polish

목표:

- ICLR style 적용
- appendix 작성
- related work citation 수동 검증
- limitations/ethics 강화
- qualitative failure cases 추가

## 4. 결론

현재 산출물은 “연구 repo + answer-first paper + 실험 파이프라인 skeleton”으로는 잘 정리되어 있다. GitHub private repo도 생성되어 있고, 기본 test/PDF build도 통과한다.

다만 top-tier 제출 가능 상태는 아니다. 가장 큰 차이는 실제 실험 증거다. 우선순위는 명확하다:

1. UCF/XD 실제 manifest를 만든다.
2. 실제 VLM cache를 만든다.
3. main/ablation/scaling grid를 돌린다.
4. assumed results를 실제 결과로 교체한다.
5. method를 learned/value-of-information policy로 강화한다.

이 다섯 가지가 완료되어야 ICLR primary로 방어 가능한 논문이 된다.
