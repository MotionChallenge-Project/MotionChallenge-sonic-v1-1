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
in one verified `result.json`. MuJoCo 3.17 CPU batch; `--jobs 8` sets rollout and
verification parallelism. Progress updates every five seconds.

Missing data and weights are downloaded and hash-checked. Cached files can be
selected with `--weights PATH --data PATH --offline`. Results and traces stay
outside Git.

This repository adapts model inference. The pinned root `MotionChallenge-Core`
submodule owns datasets, physics, metrics and verification. See `--help` for options.

## Observation track

Default: **ideal proprioception, no global position feedback**. [Admission and special handling](admission.json) · [Rules](MotionChallenge-Core/docs/OBSERVATIONS.md).

For the original privileged-state track: `uv run evaluate.py --track privileged`.
