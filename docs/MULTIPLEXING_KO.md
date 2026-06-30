# Multi-Stream Manifest Multiplexing

실제 데이터셋은 각자 annotation 포맷이 다르다. BMVM에서는 먼저 데이터셋별 annotation을 공통 CSV로 정규화한 뒤, `multiplex_dataset.py`로 multi-stream episode manifest를 만든다.

## videos.csv

필수:

```text
video_id
```

권장:

```text
path,duration_s,label
```

예시:

```csv
video_id,path,duration_s,label
ucf_robbery_001,data/raw/ucf/Robbery/Robbery001_x264.mp4,1800,robbery
ucf_normal_001,data/raw/ucf/Normal/Normal001_x264.mp4,1800,normal
```

## events.csv

필수:

```text
video_id,start_s,end_s,label
```

선택:

```text
event_id,severity,description
```

예시:

```csv
video_id,event_id,start_s,end_s,label,severity,description
ucf_robbery_001,e0001,122.0,147.0,robbery,1.0,A robbery occurs near the storefront.
```

## signals.csv

선택 파일이다. 없으면 `--simulate-missing-signals`로 smoke용 noisy signal을 만들 수 있다. 실제 논문 실험에서는 반드시 cached cheap features를 사용해야 한다.

필수:

```text
video_id,t_s
```

선택:

```text
motion,anomaly,clip
```

예시:

```csv
video_id,t_s,motion,anomaly,clip
ucf_robbery_001,122.0,0.81,0.77,0.62
```

## Manifest 생성

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

Smoke용 simulated signal:

```bash
python3 scripts/multiplex_dataset.py \
  --videos-csv data/manifests/example_videos.csv \
  --events-csv data/manifests/example_events.csv \
  --output data/manifests/example_mux.json \
  --episodes 4 \
  --streams 32 \
  --horizon-s 300 \
  --step-s 5 \
  --simulate-missing-signals
```
