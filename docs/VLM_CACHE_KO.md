# VLM Cache Interface

실제 논문 실험에서는 expensive VLM call을 매번 online으로 실행하지 말고 JSONL cache로 저장한다. BMVM runner는 `--vlm-cache` 또는 `--vlm-cache-dir`를 받으면 cache score를 verifier output으로 사용한다.

## JSONL Schema

각 줄은 하나의 VLM query 결과다.

```json
{"episode_id":"mux-00001","stream_id":"s017","t_s":122.0,"score":0.91,"summary":"A person enters a restricted area.","model":"qwen2.5-vl-7b","prompt_hash":"..."}
```

필수:

- `episode_id`
- `stream_id`
- `t_s`
- `score`

선택:

- `summary`
- `model`
- `prompt_hash`

## Single Experiment

```bash
python3 scripts/run_experiment.py \
  --manifest data/manifests/ucf_crime_multistream_128.json \
  --policies anomaly_topk,clip_topk,voi \
  --query-budget 4 \
  --vlm-cache data/vlm_cache/ucf_crime_multistream.jsonl \
  --no-simulated-vlm-fallback \
  --output results/ucf_budget4.json
```

## Grid

`--vlm-cache-dir`는 `<dataset>.jsonl` 파일을 찾는다.

```bash
python3 scripts/run_grid.py \
  --config configs/experiments/core_grid.json \
  --manifest-dir data/manifests \
  --vlm-cache-dir data/vlm_cache \
  --no-simulated-vlm-fallback \
  --output-dir results/grid
```

## Oracle Debug Cache

실제 VLM cache 생성 전에 evaluation path를 검증할 때만 사용한다.

```bash
python3 scripts/make_oracle_vlm_cache.py \
  --manifest data/synthetic/smoke.json \
  --output data/vlm_cache/smoke.jsonl
```

주의: oracle cache는 ground truth event interval을 사용하므로 논문 결과로 쓰면 안 된다. 디버깅 전용이다.

## Query Pool 생성

실제 VLM batch job에 넘길 candidate query 목록은 manifest에서 뽑는다.

```bash
python3 scripts/build_vlm_query_pool.py \
  --manifest data/manifests/ucf_crime_multistream_128.json \
  --output data/vlm_cache/ucf_query_pool.jsonl \
  --mode policies \
  --policies anomaly_topk,clip_topk,voi \
  --query-budget 4
```

출력 JSONL:

```json
{"episode_id":"mux-00001","stream_id":"s017","t_s":122.0,"video_id":"ucf_robbery_001","path":"data/raw/...mp4","prompt":"Detect and summarize safety-critical incidents in this video segment."}
```

이 query pool을 실제 VLM batch inference로 처리한 뒤, 같은 `(episode_id, stream_id, t_s)` key에 `score`를 추가한 JSONL을 `--vlm-cache`로 넣는다.

## CLIP Verifier Baseline

실제 VLM을 붙이기 전, CLIP prompt similarity를 verifier cache로 사용해 end-to-end evaluation path를 점검할 수 있다.

```bash
python3 scripts/run_clip_verifier_cache.py \
  --query-pool data/vlm_cache/ucf_query_pool.jsonl \
  --prompts configs/prompts/incidents.txt \
  --output data/vlm_cache/ucf_crime_multistream.jsonl \
  --device cuda \
  --batch-size 32
```

이 cache는 논문 main VLM 결과가 아니라 diagnostic baseline으로 취급한다.

## External VLM Prediction Adapter

Qwen/LLaVA/API 등 별도 batch job이 다음 JSONL을 출력했다고 가정한다.

```json
{"episode_id":"mux-00001","stream_id":"s017","t_s":122.0,"score":0.91,"summary":"A person enters a restricted area."}
```

BMVM cache로 변환:

```bash
python3 scripts/build_vlm_cache_from_predictions.py \
  --query-pool data/vlm_cache/ucf_query_pool.jsonl \
  --predictions data/vlm_cache/ucf_vlm_predictions.jsonl \
  --output data/vlm_cache/ucf_crime_multistream.jsonl \
  --model qwen2.5-vl-7b
```
