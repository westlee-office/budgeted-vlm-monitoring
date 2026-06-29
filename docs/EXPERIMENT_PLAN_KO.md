# 실험 계획: ICLR Primary / CVPR Strong Secondary

## 0. 논문 목표

핵심 논문 메시지:

> VLM의 성능보다, 제한된 시각 관측 예산을 어떤 stream/시간/질문에 배분하는지가 대규모 관제형 멀티모달 에이전트의 병목이다.

목표 기여:

1. **문제정의:** Budgeted Multi-Stream VLM Monitoring.
2. **벤치마크:** 기존 event-centric video dataset을 multi-stream online episode로 변환하는 BMVM protocol.
3. **방법:** cheap perception + memory + uncertainty 기반 VLM call scheduler, `TriageVLM`.
4. **분석:** cost-performance frontier, time-to-detect, false alarm, VLM calls/event, GPU-seconds/hour.

## 1. 벤치마크 데이터셋

### Core Track A: UCF-Crime-Multistream

- 원 논문: `Real-World Anomaly Detection in Surveillance Videos`, CVPR 2018.
- 규모: 1,900개의 long untrimmed surveillance videos, 총 128시간, 13개 anomaly class.
- 역할: primary surveillance benchmark.
- 변환:
  - event interval annotation을 stream-level event로 사용.
  - normal videos를 background streams로 배치.
  - episode당 64/128/256 stream 조건을 만든다.
  - timestep은 기본 2초, cheap feature는 1 fps cache.

### Core Track B: XD-Violence-Multistream

- 원 논문: `Not only Look, but also Listen`, ECCV 2020.
- 규모: 4,754 untrimmed videos, 총 217시간, audio signal 포함, weak labels.
- 역할: violence / accident / explosion / shooting 등 semantic event robustness.
- 변환:
  - test annotation이 있는 구간을 event interval로 변환.
  - audio feature는 optional cheap signal로 둔다.
  - visual-only와 audio-visual policy를 분리 비교한다.

### Core Track C: Scene-Specific Stress Track

- 후보: ShanghaiTech Campus, Street Scene.
- 역할: camera/static-scene bias, small-object event, scene-specific false alarm 테스트.
- 변환:
  - 짧은 scene-specific clips를 background-heavy episode로 multiplexing.
  - core claim의 generalization check로 사용하고, main table에는 별도 column 또는 appendix table로 둔다.

### Optional Semantic Track

- 후보: Ego4D, Video-MME, long-video QA/temporal grounding datasets.
- 역할: `incident summary correctness`와 open-vocabulary query robustness.
- 주의:
  - Ego4D는 라이선스 승인과 AWS credential 발급 시간이 필요하다.
  - 이 트랙은 ICLR main claim의 필수 경로가 아니라 semantic/reporting 분석을 강화하는 secondary track으로 둔다.

## 2. Manifest Contract

모든 데이터셋은 다음 JSON manifest로 변환한다.

```json
{
  "episodes": [
    {
      "episode_id": "ucf-mux-0001",
      "num_streams": 128,
      "horizon_s": 1800.0,
      "step_s": 2.0,
      "events": [
        {
          "event_id": "event-0001",
          "stream_id": "s017",
          "start_s": 122.0,
          "end_s": 147.0,
          "label": "robbery",
          "severity": 1.0,
          "description": "A robbery event occurs near the storefront."
        }
      ],
      "signals": [
        {
          "stream_id": "s017",
          "t_s": 122.0,
          "motion": 0.81,
          "anomaly": 0.77,
          "clip": 0.62,
          "event_ids": ["event-0001"]
        }
      ]
    }
  ]
}
```

## 3. Cheap Features

필수 feature:

- `motion`: frame difference 또는 optical flow magnitude.
- `anomaly`: lightweight anomaly model score. 1차는 pretrained I3D/VideoMAE feature + simple temporal classifier.
- `clip`: incident prompt와 frame/clip embedding cosine similarity.

선택 feature:

- object count / person count / vehicle count.
- audio violence score for XD-Violence.
- camera health / blank frame / occlusion indicator.

## 4. Policies

필수 baseline:

- `random`: random stream selection.
- `uniform`: round-robin stream selection.
- `motion_topk`: motion score top-k.
- `anomaly_topk`: anomaly prior top-k.
- `clip_topk`: incident prompt CLIP similarity top-k.
- `dense_vlm`: every stream VLM reference, upper-cost oracle.
- `TriageVLM`: value-of-information scheduler.

필수 ablation:

- w/o event memory.
- w/o uncertainty bonus.
- w/o CLIP prompt score.
- w/o anomaly prior.
- greedy no cooldown.
- learned weights vs fixed weights.

## 5. Metrics

Primary metrics:

- Event Recall@Budget.
- Mean/median Time-to-Detect.
- False Alarms per Hour.
- GPU-Seconds per Hour.
- VLM Calls per Event.

Secondary metrics:

- Event Recall by class.
- Event Recall by stream count: 32/64/128/256.
- Event Recall by event duration: short/medium/long.
- Incident Summary Correctness.
- Escalation calibration: ECE or reliability diagram for alert confidence.

## 6. Experiment Matrix

### Main Table

- Datasets: UCF-Crime-Multistream + XD-Violence-Multistream.
- Stream counts: 128 default, appendix 32/64/256.
- Budgets: 1, 2, 4, 8 VLM calls per timestep.
- Seeds: 3 episode samplings.
- Models:
  - Cheap visual encoder: CLIP ViT-B/32 or SigLIP-B/16.
  - Video feature/anomaly prior: VideoMAE/I3D feature cache + temporal classifier.
  - VLM verifier: Qwen2.5-VL-7B-Instruct or LLaVA-NeXT-Video-7B.

### Cost-Performance Frontier

- x-axis: GPU-seconds per hour.
- y-axis: Event Recall@Budget.
- Plot all policies and budgets.
- Dense VLM is plotted as high-cost reference.

### Latency Analysis

- Time-to-detect CDF.
- Recall under max delay: 10s/30s/60s/120s.
- Missed event analysis by class and duration.

### Summary Quality

- For events detected by each policy, ask VLM to produce structured incident report:
  - event type,
  - evidence,
  - severity,
  - uncertainty,
  - escalation decision.
- Evaluate with annotation matching plus MLLM-as-judge/human audit on a 300-event subset.

## 7. GPU-Hour Estimate

Default hardware assumption:

- Primary estimate: 1x A100 80GB.
- Secondary estimate: 1x RTX 4090.
- Cheap features cached once.
- VLM outputs cached by `(dataset, episode, stream, timestamp, prompt, model)`.
- Default sampling: cheap features at 1 fps, VLM query on 8-frame or short clip windows.

| Phase | Scope | A100 GPU-hour | RTX 4090 GPU-hour | Notes |
|---|---:|---:|---:|---|
| Synthetic smoke tests | no real video | 0 | 0 | CPU only |
| Data decode + cheap motion | UCF + XD full | 8-16 | 12-24 | ffmpeg/OpenCV bound |
| CLIP/SigLIP feature cache | UCF + XD at 1 fps | 12-24 | 20-36 | batched image encoding |
| Video/anomaly prior cache | UCF + XD | 35-60 | 60-95 | VideoMAE/I3D features + classifier |
| VLM verifier cache | all policies, budgets, shared candidate pool | 70-120 | 130-220 | largest variable cost |
| Ablations and budget sweeps | cached features and cached VLM responses | 15-30 | 20-45 | mostly CPU/GPU light |
| Semantic summary eval | 300-600 detected events | 10-25 | 18-40 | can be API-based instead |
| Total core paper | UCF + XD main + ablations | **150-275** | **260-460** | realistic ICLR run budget |

GPU-hour를 줄이는 fallback:

- VLM verifier를 API 또는 smaller 4B/7B model로 고정.
- budget sweep을 1/4/8로 줄이고 2를 appendix로 이동.
- stream count main을 128 하나로 고정하고 32/256은 subset.
- semantic summary eval을 300-event human/LLM audit로 제한.

## 8. Submission-Critical Acceptance Criteria

ICLR main paper에 필요한 최소 증거:

- UCF + XD full 또는 near-full main table.
- cost-performance frontier.
- latency CDF.
- at least 5 ablations.
- stream-count scaling.
- class/duration breakdown.
- qualitative timeline figure 3개 이상.
- reproducible manifest/code release.

CVPR secondary로 돌릴 경우 추가하면 좋은 증거:

- more visual examples and timelines.
- dataset protocol details and annotation QA.
- qualitative failure cases.
- comparison to recent video anomaly / video-language methods with released checkpoints.
