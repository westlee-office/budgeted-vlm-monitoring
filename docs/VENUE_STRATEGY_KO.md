# Venue Strategy

## Primary: ICLR 2027

가장 좋은 framing:

- selective perception as agentic decision-making.
- perception allocation under semantic uncertainty.
- benchmark + method + analysis.
- cost-performance frontier as the main empirical object.

ICLR에서 중요한 점:

- 단순 application paper로 보이면 약하다.
- `what to look at`을 일반적인 agentic perception problem으로 정식화해야 한다.
- benchmark가 novelty의 절반이고, method가 나머지 절반이다.
- ablation과 failure analysis가 부족하면 reject risk가 높다.

## Strong Secondary: CVPR 2027

CVPR로 돌릴 때 강조점:

- multi-stream video benchmark.
- video-language monitoring protocol.
- visual examples, timelines, dataset construction.
- comparison with video anomaly / streaming video LLM / long-video methods.

CVPR에서 중요한 점:

- benchmark protocol과 dataset construction detail을 더 크게 보여야 한다.
- qualitative figure가 많아야 한다.
- visual evidence와 reproducibility가 강해야 한다.

## Risk Register

| Risk | Impact | Mitigation |
|---|---|---|
| 기존 anomaly detection으로 보임 | novelty 약화 | VLM-call budget, semantic report, multi-stream scheduling을 전면에 둠 |
| benchmark simulator가 artificial해 보임 | 설득력 약화 | real dataset intervals, stream-count scaling, hard distractors, real runtime cost 측정 |
| method가 heuristic으로 보임 | ICLR risk | value-of-information objective, learned-weight variant, off-policy analysis 추가 |
| dense VLM oracle와 gap이 큼 | contribution 약화 | cost frontier로 framing하고 fixed-budget recall을 primary metric으로 둠 |
| dataset access 지연 | 일정 risk | UCF-Crime/XD core 먼저, Ego4D는 optional semantic track |
