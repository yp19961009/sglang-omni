# Examples

Run these commands from the repository root after installing `sglang-omni`.

## Qwen3-Omni Server

Text output:

```bash
python examples/run_qwen3_omni_server.py \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --port 8000 \
  --model-name qwen3-omni
```

Text and audio output:

```bash
python examples/run_qwen3_omni_speech_server.py \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --gpu-thinker 0 \
  --gpu-talker 1 \
  --gpu-code2wav 1 \
  --port 8000 \
  --model-name qwen3-omni
```

Qwen3-Omni FP8, one-GPU colocated H100/H20:

```bash
sgl-omni serve \
  --config examples/configs/qwen3_omni_fp8_colocated.yaml \
  --colocate \
  --model-name qwen3-omni \
  --port 8000
```

## Qwen3.5-Omni Optimized Speech Server

The optimized two-H20 launcher enables Thinker and Talker CUDA graphs,
`torch.compile`, Talker partial-start, image-encoder same-batch deduplication,
the native grouped vision encoder, BF16 relay payloads, and a prepared-media
reuse cache. Prepared-media requests must still opt in with `reuse_loaded`.
Radix caching remains disabled for multimodal correctness.

Run a known-good single-request configuration on two of physical GPUs 0-3:

```bash
CUDA_VISIBLE_DEVICES=2,3 \
  ./examples/run_qwen35_omni_speech_optimized.sh
```

For a concurrency experiment, align the Thinker and Talker scheduler, compile,
and graph limits with one setting:

```bash
CUDA_VISIBLE_DEVICES=0,1 MAX_RUNNING_REQUESTS=8 \
  ./examples/run_qwen35_omni_speech_optimized.sh
```

Override `MODEL_PATH`, `MODEL_NAME`, `HOST`, `PORT`,
`ENCODER_MEM_RESERVE`, or `SGLANG_OMNI_PREPROCESSED_MEDIA_CACHE_SIZE` through
environment variables when needed. Larger graph ranges increase startup time
and GPU memory use.

## Ming-Omni Server

Text output:

```bash
python examples/run_ming_omni_server.py \
  --model-path inclusionAI/Ming-flash-omni-2.0 \
  --port 8000 \
  --model-name ming-omni
```

Text and audio output:

```bash
python examples/run_ming_omni_speech_server.py \
  --model-path inclusionAI/Ming-flash-omni-2.0 \
  --gpu-thinker 0 \
  --gpu-talker 1 \
  --port 8000 \
  --model-name ming-omni
```

Use a different `--port` if you run more than one server at the same time.
