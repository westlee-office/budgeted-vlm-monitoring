# Manifest Schema

The benchmark core uses one JSON file with `episodes`.

```json
{
  "name": "dataset-name",
  "version": "0.1",
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

Real dataset converters should populate `signals` from cached cheap features:

- `motion`: frame differencing, optical flow, or foreground activity.
- `anomaly`: lightweight anomaly model score.
- `clip`: open-vocabulary similarity to incident prompts.

The expensive VLM query is intentionally simulated in the benchmark core and should be replaced with cached VLM outputs for real runs.
