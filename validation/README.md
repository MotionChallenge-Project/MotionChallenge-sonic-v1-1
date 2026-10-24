# Native inference validation

`native-parity.json` is the original 160-case, four-reset report from the pinned
author C++ observation gatherers, encoder, decoder and motor-target comparison.
The script raises an error if its stated numerical tolerances are exceeded.

Run with the author checkout at `runtime.json` → `source_commit` and the original
SHA256-verified encoder and decoder from `runtime.json`:

```bash
uv run validate_inference.py . /path/to/author-source /path/to/decoder.onnx /path/to/encoder.onnx
```

This validates CPU inference. It does not compare TensorRT or replace the
benchmark's shared physics and scoring.
