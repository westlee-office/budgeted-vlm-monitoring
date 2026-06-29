# CSV Schema for Real Dataset Conversion

Use these CSV files as the handoff between dataset-specific preprocessing and the BMVM benchmark core.

## events.csv

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

## signals.csv

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

Build a manifest:

```bash
python3 scripts/build_manifest_from_csv.py \
  --events-csv data/manifests/ucf_events.csv \
  --signals-csv data/features/ucf_signals.csv \
  --output data/manifests/ucf_crime_multistream.json \
  --step-s 2.0
```
