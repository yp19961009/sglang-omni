# Qwen3.5-Omni S2T / Video-AMME Development Log

Date: 2026-07-06 (Asia/Shanghai)
Base commit before implementation: `a3707342d1b9730b8fd740a4e9aa60b5ec452559`
Implementation commit: `949d1efdf1a7cf897f734597d6322ba3162308c5` (`feat: support qwen35-omni single-request S2T smoke`)
Workspace: `/myapp/sglang-omni`
Model path: `/myapp/models/qwen3_5_omni_23b_final_multilingual_all_voice_bf16_0315`
Status: implementation commit exists; this log is a follow-up development note.

## Goal

Support Qwen3.5-Omni text output in SGLang-Omni, with the first target being the
Video-AMME S2T/text-only path in `benchmarks/eval/benchmark_omni_videoamme.py`.
The vLLM Qwen3.5-Omni implementation is used only as a data-flow reference. The
actual implementation stays on SGLang-Omni abstractions and reuses SGLang's
Qwen3-Next LLM implementation for the thinker.

The current scope is intentionally narrow:

- support video + audio input and text output;
- run on one GPU for the current development target;
- do not implement talker, codec/code2wav, speech output, or voice cloning;
- do not rely on encoder cache for the first pass;
- keep the Qwen3-Omni path isolated from the new Qwen3.5-Omni path.

## Implementation Shape

A new model package was added under `sglang_omni/models/qwen35_omni/`. It keeps
Qwen3.5-specific config, preprocessing, encoders, merge logic, and thinker glue
separate from the existing `qwen3_omni` implementation.

The main pieces are:

- Local config/processor adaptation for `qwen3_omni_next`, because the current
  container Transformers build does not provide `Qwen3OmniNextProcessor`.
- A processor path for Video-AMME requests with `videos`, `audios`, and text
  prompt. It expands Qwen3.5 multimodal placeholders and uses Qwen3.5 special
  token ids.
- A Qwen3.5 audio encoder tower with the Next-style Conv2D downsampling path:
  `n_window=50`, `n_window_infer=200`, `conv_chunksize=500`, `output_dim=2048`,
  and feature length derived from `downsample_times=4` and `chunk_size=100`.
- A Qwen3.5 vision encoder wrapper that reuses the Qwen3-Omni vision-stage
  structure, with Qwen3.5 config and weight-name adaptation. Deepstack is
  disabled in the current text path because the thinker wrapper does not consume
  deepstack inputs yet.
- Qwen3.5 merge/M-RoPE settings, including `audio_token_id=248076`,
  `audio_start_token_id=248070`, `audio_end_token_id=248071`,
  `position_id_per_seconds=13`, and `rope_scaling.mrope_section=[11, 11, 10]`.
- A thin `Qwen35OmniNextThinkerForCausalLM` wrapper around
  `sglang.srt.models.qwen3_next.Qwen3NextForCausalLM`. It loads
  `thinker.model.*` and `thinker.lm_head.*` weights, and skips talker, codec,
  code2wav, MTP, visual, and audio-tower weights that are not needed for S2T.
- SGLang model-runner registration for the new thinker architecture.

The text pipeline currently has six logical stages:

```text
preprocessing -> image_encoder
              -> audio_encoder
              -> mm_aggregate -> thinker -> decode
```

For the Qwen3.5 text pipeline, `preprocessing` is placed in its own process and
`image_encoder`, `audio_encoder`, `mm_aggregate`, `thinker`, and `decode` share
the GPU pipeline process. This matters because SGLang sets PyTorch intra-op
threads to 1 inside the model process, which makes CPU preprocessing much slower
when preprocessing is colocated with the thinker.

## What Runs Now

The current implementation can run a single-GPU Video-AMME S2T request through
`/v1/chat/completions` with:

- local Video-AMME data/cache under `/myapp/data/videoamme`;
- `videos` + `audios` inputs;
- `modalities=["text"]`;
- `--video-fps 2`;
- `--video-max-frames 128`;
- `--video-max-pixels 401408`;
- `--max-tokens 256`;
- `--temperature 0.0`.

The best current serving shape for single-request correctness is:

```bash
CUDA_VISIBLE_DEVICES=6 PYTORCH_ALLOC_CONF=expandable_segments:True \
python -m sglang_omni.cli serve \
  --model-path /myapp/models/qwen3_5_omni_23b_final_multilingual_all_voice_bf16_0315 \
  --text-only \
  --model-name qwen35-omni-s2t \
  --host 127.0.0.1 \
  --port 8010 \
  --thinker-gpus 0 \
  --thinker-cuda-graph on \
  --thinker-torch-compile off \
  --encoder-mem-reserve 0.30 \
  --stages.0.runtime.max_seq_len 32768 \
  --stages.4.runtime.max_seq_len 32768
```

For correctness-sensitive measurement, warmup and measured requests should be
different requests. Repeating the exact same multimodal request also exercises
prefix/radix cache correctness, which is a separate issue.

Verified single-request examples on 2026-07-06:

| Cache mode | Step | Sample | Expected -> Predicted | Latency |
| --- | --- | --- | --- | --- |
| default radix | warmup | `001-1` | C -> C | 3.496s |
| default radix | measured, same video different question | `001-2` | A -> A | 1.601s |
| default radix | measured, different video | `002-1` | C -> A | 2.948s |
| radix disabled | warmup | `001-1` | C -> C | 3.578s |
| radix disabled | measured, same video different question | `001-2` | A -> A | 2.266s |
| radix disabled | measured, different video | `002-1` | C -> C | 2.987s |

Result artifacts from this run were written under `/tmp/qwen35_single_*`,
including:

- `/tmp/qwen35_single_default_diffreq_summary.json`
- `/tmp/qwen35_single_noradix_diffreq_summary.json`

## Optimizations Tried

### Preprocessing process split

Moving `preprocessing` into its own process is useful. When colocated with the
SGLang thinker process, preprocessing inherited the `torch.set_num_threads(1)`
behavior and a standalone preprocessing call that should take roughly 1 second
was observed around 4.17 seconds. With preprocessing split out, the measured
preprocess section returned to roughly 1.0 second, with about 0.6 second of
inter-process transfer/fanout overhead.

Observed stage-level profile after the split:

| Stage area | Approx time |
| --- | ---: |
| preprocessing total | 1.035s |
| video read/decode inside preprocessing | 0.363s |
| preprocessing minus video read | 0.672s |
| preprocess -> GPU IPC/fanout | ~0.6s |
| image encoder | 0.946s |
| audio encoder | 0.961s |
| encoder wall, parallel | 0.962s |
| thinker total | 1.552s |
| thinker prefill/decode | 1.159s |

### CUDA graph

CUDA graph is worth enabling for the thinker. Startup graph capture was around
1.3 seconds and used about 0.09 GB extra GPU memory. A one-sample request
improved from about 4.148 seconds with CUDA graph disabled to about 3.683
seconds with CUDA graph enabled.

Recommendation: keep `--thinker-cuda-graph on` for this path.

### torch.compile

`torch.compile` is not recommended for the current Qwen3.5 path.

Observed behavior:

| Mode | Result |
| --- | --- |
| torch compile on, CUDA graph off | first request about 4.278s and correct, later repeated request produced 256-token repeated output |
| torch compile on + CUDA graph on | CUDA graph capture took about 42.22s, single request about 5.48s |

Recommendation: keep `--thinker-torch-compile off`.

### Radix/prefix cache

Radix cache can improve latency when the next request shares a useful prefix;
for example, warmup `001-1` followed by same-video `001-2` completed in about
1.601 seconds with default radix cache. However, current correctness is not
reliable for Qwen3.5 multimodal embeddings.

Recommendation for correctness-first testing: disable radix cache or fix the
multimodal cache-key/input-embedding interaction before using cache-dependent
numbers for acceptance.

## Known Issues And Risks

1. Radix cache correctness is not solved for Qwen3.5 multimodal requests.
   Repeating the same request after warmup can produce a 256-token repeated
   timestamp response. A different-video request after warmup also produced an
   incorrect answer in the default-radix run. This likely means the cache key or
   prefix reuse path is not aligned with multimodal `input_embeds`.

2. Disabling radix cache avoids the observed repeated-output issue in the small
   single-request tests, but default no-radix memory behavior is more aggressive.
   A 10-request sequential no-radix test hit OOM after a few requests with the
   default memory contract. A too-low `mem_fraction_static=0.45` could not even
   initialize the model. This needs a proper memory/KV-cache policy before long
   no-radix runs.

3. Multi-request and multi-concurrency are not validated yet. Earlier concurrent
   tests are not acceptance-quality because cache correctness and memory behavior
   were still under investigation.

4. ci-50 accuracy is not validated yet. Small-sample runs are useful for smoke
   and latency, but they are not enough to judge final Video-AMME quality.

5. Deepstack visual features are disabled in the current Qwen3.5 text path. This
   may leave accuracy on the table if the Qwen3.5 thinker expects deepstack
   information.

6. The latency reported by `benchmark_omni_videoamme.py` starts before the HTTP
   request and ends after the response. The client sends local media paths rather
   than preloaded tensors, so service-side video/audio decode and preprocessing
   are included in the request latency.

7. Encoder cache is not implemented. This keeps the first implementation simple,
   but repeated media requests still pay encoder cost unless a separate cache is
   added later.

## Next Steps

- Add a first-class Qwen3.5 text config/CLI option for disabling radix cache, so
  correctness-first runs do not depend on a temporary YAML file.
- Fix radix cache correctness for multimodal `input_embeds`. A likely direction
  is to make media placeholders and media hashes part of the effective cache
  identity, or to prevent prefix reuse across unresolved multimodal embedding
  spans.
- Tune no-radix memory policy if it remains the correctness-first mode. This may
  involve `mem_fraction_static`, max running requests, cache sizing, and the
  SGLang chunk cache behavior.
- Re-enable or correctly consume Qwen3.5 deepstack visual features if reference
  accuracy indicates they are needed.
- Run ci-50 only after the single-request correctness path is stable.
- Compare against a trusted vLLM/HF reference path, then target a gap within
  roughly 3-5 percentage points.

## Signature

Logged by Codex on 2026-07-06.
Base commit before implementation: `a3707342d1b9730b8fd740a4e9aa60b5ec452559`.
Implementation commit: `949d1efdf1a7cf897f734597d6322ba3162308c5`.
