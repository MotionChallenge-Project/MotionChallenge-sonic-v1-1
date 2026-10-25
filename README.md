# SONIC v1.1

[Author project](https://github.com/NVlabs/GR00T-WholeBodyControl) · [Checkpoint](https://huggingface.co/nvidia/GEAR-SONIC/blob/6733128a3d8a523b1418b06bca3cdf61c8b0987f/sonic_v1_1/model_decoder.onnx) · [Provenance](provenance.json)

## Install and run

```bash
git clone --recurse-submodules https://github.com/MotionChallenge-Project/MotionChallenge-sonic-v1-1.git
cd MotionChallenge-sonic-v1-1
uv sync
uv run evaluate.py
```

One full rollout per motion produces success, completion and all six MPBPE values
in one verified `result.json`. Tuned inference defaults; CPU slots scale with
available cores. `--jobs N` overrides the slots. Progress updates every five seconds.

Missing data and weights are downloaded and hash-checked. Cached files can be
selected with `--weights PATH --data PATH --offline`. Results and traces stay
outside Git.

This repository adapts model inference. The pinned root `MotionChallenge-Core`
submodule owns datasets, physics, metrics and verification. See `--help` for options.

## Observations

Ideal proprioception, without robot world-position or contact feedback. [Checkpoint requirements](admission.json) · [Rules](https://github.com/MotionChallenge-Project/MotionChallenge-Core/blob/da0aec4023bbc3d29be67e1fce9fa9ab8a48c4b5/docs/OBSERVATIONS.md).
