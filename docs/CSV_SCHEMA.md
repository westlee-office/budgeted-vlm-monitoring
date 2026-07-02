# CSV Schema for Real Dataset Conversion

Use these CSV files as the handoff between dataset-specific preprocessing and the BMVM benchmark core.

There are two supported stages:

- Source-level CSVs, used by `scripts/multiplex_dataset.py` before multi-stream episodes are created.
- Episode-level CSVs, used by `scripts/build_manifest_from_csv.py` when data is already multiplexed.

## Source-Level videos.csv

Required columns:

```text
video_id
```

`id` is accepted as an alias for `video_id`.

Recommended columns:

```text
path,duration_s,label,split
```

Example:

```csv
video_id,path,duration_s,label,split
ucf_000001,data/raw/ucf_crime/Robbery001_x264.mp4,312.4,robbery,train
ucf_000002,data/raw/ucf_crime/Normal001_x264.mp4,180.0,normal,train
```

## Source-Level events.csv

Required columns:

```text
video_id,start_s,end_s,label
```

Optional columns:

```text
event_id,severity,description
```

Example:

```csv
video_id,event_id,start_s,end_s,label,severity,description
ucf_000001,ucf_000001_e0001,122.0,147.0,robbery,1.0,A robbery occurs near the storefront.
```

## Source-Level signals.csv

Required columns:

```text
video_id,t_s
```

Optional score columns:

```text
motion,anomaly,clip
```

Example:

```csv
video_id,t_s,motion,anomaly,clip
ucf_000001,122.0,0.81,0.77,0.62
```

Validate source-level CSVs before manifest generation:

```bash
python3 scripts/validate_dataset_csvs.py \
  --videos data/manifests/ucf_videos.csv \
  --events data/manifests/ucf_events.csv \
  --signals data/features/ucf_signals.csv \
  --path-root . \
  --check-paths \
  --output data/manifests/ucf_validation_report.json
```

Then multiplex:

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
  --event-streams-per-episode 12
```

## Episode-Level events.csv

Required columns:

```text
episode_id,stream_id,start_s,end_s,label
```

Optional columns:

```text
event_id,severity,description
```

Example:

```csv
episode_id,stream_id,event_id,start_s,end_s,label,severity,description
ucf-mux-0001,s017,e0001,122.0,147.0,robbery,1.0,A robbery occurs near the storefront.
```

## Episode-Level signals.csv

Required columns:

```text
episode_id,stream_id,t_s
```

Optional score columns:

```text
motion,anomaly,clip,event_ids
```

Example:

```csv
episode_id,stream_id,t_s,motion,anomaly,clip,event_ids
ucf-mux-0001,s017,122.0,0.81,0.77,0.62,e0001
```

Build a manifest from already-multiplexed `events.csv` and `signals.csv`:

```bash
python3 scripts/build_manifest_from_csv.py \
  --events-csv data/manifests/ucf_events.csv \
  --signals-csv data/features/ucf_signals.csv \
  --output data/manifests/ucf_crime_multistream.json \
  --step-s 2.0
```

Validate episode-level CSVs:

```bash
python3 scripts/validate_dataset_csvs.py \
  --schema episode \
  --events data/manifests/episode_events.csv \
  --signals data/features/episode_signals.csv \
  --output data/manifests/episode_validation_report.json
```

For source datasets organized by original videos, prefer the multiplexing path in `docs/MULTIPLEXING_KO.md`.
