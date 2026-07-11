# SPDX-License-Identifier: Apache-2.0
"""Video-AMME benchmark for Qwen3-Omni video + audio input.

Video-AMME is derived from the Video-MME CI subset. The video is paired with a
spoken audio question; the text prompt contains only routing and answer-format
instructions.

Usage:
    python -m benchmarks.dataset.prepare --dataset videoamme-ci-50

    python examples/run_qwen3_omni_server.py \
        --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
        --model-name qwen3-omni \
        --port 30000 \
        --thinker-max-seq-len 32768 \
        --mem-fraction-static 0.78

    python -m benchmarks.eval.benchmark_omni_videoamme \
        --model qwen3-omni --port 30000 \
        --repo-id zhaochenyang20/Video_AMME_ci \
        --max-samples 50 --max-concurrency 16 \
        --video-fps 2 --video-max-frames 128 --video-max-pixels 401408

H200 Reference Results

Benchmark: Video-AMME | Dataset: zhaochenyang20/Video_AMME_ci test split (50 questions)
Hardware:  1 x H200
Last verified: 2026-04-26

Accuracy

| Model      | Config                | accuracy | correct | failed | mc_fallback | Source                                                              |
| ---------- | --------------------- | -------- | ------- | ------ | ----------- | ------------------------------------------------------------------- |
| Qwen3-Omni | thinker-only, ci-50, c=8 | 66.00%   | 33/50   | 0      | 0           | PR #411 [H200, c=8, max_tokens=256] |
| Qwen3-Omni | thinker-talker, ci-50, c=8 | 64.00%   | 32/50   | 0      | 0           | PR #411 [H200, c=8, max_tokens=256] |
| Qwen3-Omni | thinker-only, ci-50, c=8 | 66.00%   | 33/50   | 0      | 0           | PR #411 [H100, c=8, max_tokens=256] |
| Qwen3-Omni | thinker-talker, ci-50, c=8 | 68.00%   | 34/50   | 0      | 0           | local [H100, c=8, max_tokens=256] |

Speed

| Model      | Config                | completed | failed | latency_mean_s | latency_median_s | latency_p95_s | latency_p99_s | output_tok_per_req_s | output_tokens_mean | output_tokens_total | prompt_tokens_mean | prompt_tokens_total | throughput_qps | Source                                                              |
| ---------- | --------------------- | --------- | ------ | -------------- | ---------------- | ------------- | ------------- | ------------------------------ | --------------- | ---------------- | ------------------ | ------------------- | -------------- | ------------------------------------------------------------------- |
| Qwen3-Omni | thinker-only, ci-50, c=8 | 50        | 0      | 44.530         | 46.846           | 52.694        | 53.180        | 0.9                            | 40.0            | 2025             | 21684.0            | 1084218             | 0.167          | PR #411 [H200, c=8, max_tokens=256] |
| Qwen3-Omni | thinker-talker, ci-50, c=8 | 50        | 0      | 40.423         | 40.063           | 63.088        | 81.046        | 1.0                            | 41.0            | 2050             | 21684.0            | 1084218             | 0.193          | PR #411 [H200, c=8, max_tokens=256] |
| Qwen3-Omni | thinker-only, ci-50, c=8 | 50        | 0      | 38.408         | 40.320           | 44.883        | 45.764        | 1.1                            | 44.0            | 2181             | 21684.0            | 1084218             | 0.194          | PR #411 [H100, c=8, max_tokens=256] |
| Qwen3-Omni | thinker-talker, ci-50, c=8 | 50        | 0      | 33.618         | 33.365           | 46.380        | 58.321        | 1.3                            | 42.0            | 2123             | 21684.0            | 1084218             | 0.230          | local [H100, c=8, max_tokens=256] |


Talker WER

| Model      | Config                    | evaluated | skipped | wer_corpus | wer_per_sample_mean | wer_per_sample_p95 | wer_per_sample_max | n_above_50_pct_wer | rtf_mean | audio_duration_mean_s | Source                                                              |
| ---------- | ------------------------- | --------- | ------- | ---------- | ------------------- | ------------------ | ------------------ | ------------------ | -------- | --------------------- | ------------------------------------------------------------------- |
| Qwen3-Omni | thinker-talker, ci-50, c=8 | 50/50     | 0       | 2.58%      | 18.87%              | 155.00%            | 200.00%            | 6                  | 6.3183   | 11.750                | PR #411 [H200, c=8, max_tokens=256] |
| Qwen3-Omni | thinker-talker, ci-50, c=8 | 50/50     | 0       | 2.29%      | 11.94%              | 100.00%            | 200.00%            | 4                  | 6.0412   | 12.356                | local [H100, c=8, max_tokens=256] |

Local v1 Pipeline Result (this workspace, 2026-05-01)

Accuracy

| Model      | Config                   | accuracy | correct | failed | mc_fallback | Source                                               |
| ---------- | ------------------------ | -------- | ------- | ------ | ----------- | ---------------------------------------------------- |
| Qwen3-Omni | thinker-only, ci-50, c=8 | 68.00%   | 34/50   | 0      | 0           | local v1 sweep [H200, ci-50, c=8, max_tokens=256]   |

Speed

| Model      | Config                   | completed | failed | latency_mean_s | latency_median_s | latency_p95_s | latency_p99_s | output_tok_per_req_s | output_tokens_mean | output_tokens_total | prompt_tokens_mean | prompt_tokens_total | throughput_qps | Source                                               |
| ---------- | ------------------------ | --------- | ------ | -------------- | ---------------- | ------------- | ------------- | ------------------------------ | --------------- | ---------------- | ------------------ | ------------------- | -------------- | ---------------------------------------------------- |
| Qwen3-Omni | thinker-only, ci-50, c=8 | 50        | 0      | 133.245        | 137.354          | 155.201       | 159.106       | 0.3                            | 43              | 2172             | 21684              | 1084218             | 0.058          | local v1 sweep [H200, ci-50, c=8, max_tokens=256]   |

Local Qwen3.5-Omni S2T Result (this workspace, 2026-07-08/09)

Accuracy

| Model        | Config                                              | accuracy | correct | failed | mc_fallback | Source                                                                                        |
| ------------ | --------------------------------------------------- | -------- | ------- | ------ | ----------- | --------------------------------------------------------------------------------------------- |
| Qwen3.5-Omni | thinker-only, ci-50, c=8                            | 72.00%   | 36/50   | 0      | 0           | local H20 GPU6, qwen35_s2t_align/sglang-qwen35-videoamme-c8-20260708-183040                   |
| Qwen3.5-Omni | thinker-only, ci-50, c=8, stream=True, video_fps=1  | 64.00%   | 32/50   | 0      | 0           | local H20 GPU6, qwen35_s2t_align/sglang-qwen35-videoamme-c8-fps1-ttft-20260708-185632         |
| Qwen3.5-Omni | thinker-only, ci-50, c=8, video_fps=1, server TTFT  | 66.00%   | 33/50   | 0      | 0           | local H20 GPU7, qwen35_s2t_align/sglang-videoamme-c8-fps1-postmedia-20260709-081734/ci50_c8  |
| Qwen3.5-Omni | thinker-only, ci-50, c=8, video_fps=1, no qwen35 cache metadata | 66.00% | 33/50 | 0 | 0 | local H20 GPU7, qwen35_s2t_align/sglang-videoamme-c8-no-cachemeta-20260709-085815/ci50_c8 |

Speed

| Model        | Config                   | completed | failed | latency_mean_s | latency_median_s | latency_p95_s | latency_p99_s | output_tok_per_req_s | output_tokens_mean | output_tokens_total | prompt_tokens_mean | prompt_tokens_total | throughput_qps | Source                                                                         |
| ------------ | ------------------------ | --------- | ------ | -------------- | ---------------- | ------------- | ------------- | -------------------- | ------------------ | ------------------- | ------------------ | ------------------- | -------------- | ------------------------------------------------------------------------------ |
| Qwen3.5-Omni | thinker-only, ci-50, c=8 | 50        | 0      | 15.389         | 14.565           | 23.419        | 33.100        | 0.4                  | 7.0                | 345                 | 14763.0            | 738170              | 0.505          | local H20 GPU6, max_tokens=256, fps=2, max_frames=128, max_pixels=401408       |
| Qwen3.5-Omni | thinker-only, ci-50, c=8, video_fps=1 | 50 | 0 | 9.854 | 10.025 | 13.968 | 17.006 | 0.7 | 7.3 | 364 | 11275.0 | 563752 | 0.778 | local H20 GPU7, max_tokens=256, max_frames=128, max_pixels=401408, server TTFT profile |
| Qwen3.5-Omni | thinker-only, ci-50, c=8, video_fps=1, no qwen35 cache metadata | 50 | 0 | 9.829 | 9.729 | 14.504 | 15.531 | 0.7 | 7.1 | 357 | 11275.0 | 563752 | 0.795 | local H20 GPU7, max_tokens=256, max_frames=128, max_pixels=401408, qwen35 cache metadata removed |

Server-Side First-Token TTFT / Speed (video_fps=1)

Warmup sample: `002-1` outside the measured profile window; measured set:
Video-AMME ci-50; max_concurrency=8; non-streaming HTTP responses; thinker CUDA
graph on; thinker torch.compile on; radix cache disabled; max_tokens=256;
max_frames=128; max_pixels=401408. The no-cache-metadata run removes qwen35
`media_cache_keys` / fake media token id propagation while keeping radix cache
disabled.

| Model        | Config                                             | completed | failed | accuracy | latency_mean_s | latency_median_s | latency_p95_s | latency_p99_s | throughput_qps | output_tokens_total | prompt_tokens_total | Source                                                                                      |
| ------------ | -------------------------------------------------- | --------- | ------ | -------- | -------------- | ---------------- | ------------- | ------------- | -------------- | ------------------- | ------------------- | ------------------------------------------------------------------------------------------- |
| Qwen3.5-Omni | thinker-only, ci-50, c=8                           | 50        | 0      | 66.00%   | 9.854          | 10.025           | 13.968        | 17.006        | 0.778          | 364                 | 563752              | local H20 GPU7, qwen35_s2t_align/sglang-videoamme-c8-fps1-postmedia-20260709-081734/ci50_c8 |
| Qwen3.5-Omni | thinker-only, ci-50, c=8, no qwen35 cache metadata | 50        | 0      | 66.00%   | 9.829          | 9.729            | 14.504        | 15.531        | 0.795          | 357                 | 563752              | local H20 GPU7, qwen35_s2t_align/sglang-videoamme-c8-no-cachemeta-20260709-085815/ci50_c8 |

First-token profiler summary:

| Run                       | Metric                    | count | mean_ms  | p50_ms   | p95_ms    | max_ms    |
| ------------------------- | ------------------------- | ----- | -------- | -------- | --------- | --------- |
| server TTFT baseline      | request_to_first_token    | 50    | 7810.796 | 7745.798 | 11222.975 | 12080.812 |
| server TTFT baseline      | post_media_to_first_token | 50    | 6073.107 | 6658.167 | 8614.775  | 8858.777  |
| server TTFT baseline      | prefill_to_first_token    | 50    | 1329.771 | 1339.071 | 1994.968  | 2065.155  |
| no qwen35 cache metadata  | request_to_first_token    | 50    | 8020.857 | 7623.649 | 11630.533 | 11934.907 |
| no qwen35 cache metadata  | post_media_to_first_token | 50    | 6137.068 | 6478.907 | 8421.332  | 9031.351  |
| no qwen35 cache metadata  | prefill_to_first_token    | 50    | 1244.348 | 1330.549 | 1622.105  | 1795.375  |

Stage profile summary:

| Run                       | Stage         | Interval                                                   | avg_ms   | p50_ms   | p95_ms   | max_ms   |
| ------------------------- | ------------- | ---------------------------------------------------------- | -------- | -------- | -------- | -------- |
| server TTFT baseline      | preprocessing | stage_input_received->stage_complete                       | 2035.973 | 1710.155 | 4519.683 | 5982.146 |
| server TTFT baseline      | preprocessing | preprocess_media_load_start->preprocess_media_load_end     | 515.858  | 504.705  | 759.564  | 849.651  |
| server TTFT baseline      | preprocessing | preprocess_hf_processor_start->preprocess_hf_processor_end | 268.152  | 288.980  | 325.965  | 334.985  |
| server TTFT baseline      | image_encoder | stage_input_received->stage_complete                       | 2484.427 | 2374.044 | 4314.467 | 5039.534 |
| server TTFT baseline      | audio_encoder | stage_input_received->stage_complete                       | 3258.032 | 3195.976 | 5602.437 | 7134.064 |
| server TTFT baseline      | mm_aggregate  | stage_input_received->stage_complete                       | 3264.520 | 3195.684 | 5602.417 | 7134.757 |
| server TTFT baseline      | thinker       | scheduler_prefill_start->scheduler_first_emit              | 1329.771 | 1339.070 | 1994.968 | 2065.155 |
| server TTFT baseline      | decode        | stage_input_received->stage_complete                       | 1.605    | 0.615    | 1.407    | 34.441   |
| no qwen35 cache metadata  | preprocessing | stage_input_received->stage_complete                       | 2191.674 | 1667.591 | 4674.976 | 6120.545 |
| no qwen35 cache metadata  | preprocessing | preprocess_media_load_start->preprocess_media_load_end     | 518.366  | 512.874  | 751.186  | 834.064  |
| no qwen35 cache metadata  | preprocessing | preprocess_hf_processor_start->preprocess_hf_processor_end | 289.592  | 305.867  | 354.259  | 400.634  |
| no qwen35 cache metadata  | image_encoder | stage_input_received->stage_complete                       | 2510.664 | 2438.798 | 4196.444 | 5827.525 |
| no qwen35 cache metadata  | audio_encoder | stage_input_received->stage_complete                       | 2927.244 | 3188.097 | 4747.820 | 6467.760 |
| no qwen35 cache metadata  | mm_aggregate  | stage_input_received->stage_complete                       | 3293.897 | 3392.545 | 4747.876 | 6467.785 |
| no qwen35 cache metadata  | thinker       | scheduler_prefill_start->scheduler_first_emit              | 1244.348 | 1330.549 | 1622.105 | 1795.375 |
| no qwen35 cache metadata  | decode        | stage_input_received->stage_complete                       | 0.975    | 0.556    | 1.619    | 15.917   |

Streaming TTFT / Speed (video_fps=1)

| Model        | Config                                             | completed | failed | latency_mean_s | latency_median_s | latency_p95_s | latency_p99_s | text_ttft_mean_s | text_ttft_median_s | text_ttft_p95_s | text_ttft_p99_s | output_tok_per_req_s | output_tokens_mean | output_tokens_total | prompt_tokens_mean | prompt_tokens_total | throughput_qps | Source                                                                         |
| ------------ | -------------------------------------------------- | --------- | ------ | -------------- | ---------------- | ------------- | ------------- | ---------------- | ------------------ | --------------- | --------------- | -------------------- | ------------------ | ------------------- | ------------------ | ------------------- | -------------- | ------------------------------------------------------------------------------ |
| Qwen3.5-Omni | thinker-only, ci-50, c=8, stream=True, video_fps=1 | 50        | 0      | 10.514         | 10.332           | 15.499        | 17.005        | 8.1495           | 8.1984             | 10.8445         | 12.2999         | 0.6                  | 7.0                | 340                 | 11275.0            | 563752              | 0.744          | local H20 GPU6, max_tokens=256, max_frames=128, max_pixels=401408              |

Single Request First-Token TTFT / Speed (video_fps=1)

Warmup sample: `002-1`; measured sample: `001-1`; single concurrency;
thinker CUDA graph on; thinker torch.compile on; radix cache disabled;
max_tokens=256; max_frames=128; max_pixels=401408.

Metric notes:

- `latency_mean_s` is client-side end-to-end request latency for the measured
  request.
- `request_to_first_token_ms` is server-side profiler time from the first
  recorded request event, typically coordinator admission or preprocessing input
  receipt, to `scheduler_first_emit`, the first sampled token observed by the
  thinker scheduler. It includes preprocessing, media load / video decode /
  frame sampling / resize, HF processor, cross-process payload transfer,
  image/audio encoder, multimodal aggregation, thinker request build, and
  thinker prefill to first token. It does not include client-side HTTP time,
  detokenization/decode stage output, SSE delivery, or later tokens.
- `post_media_to_first_token_ms` starts from `preprocess_hf_processor_start`,
  after media loading / video decode / frame sampling / resize and before the HF
  processor, then ends at the same first sampled token event.
- `prefill_to_first_token_ms` starts from thinker `scheduler_prefill_start` and
  isolates the thinker prefill-to-first-token portion.
- Stage profile rows with `stage_input_received->stage_complete` are stage wall
  times. Aggregation stages can include waiting for upstream encoder results;
  for example `mm_aggregate` is not pure compute time.

| Model        | Config                                      | completed | failed | latency_mean_s | output_tokens_total | prompt_tokens_total | request_to_first_token_ms | post_media_to_first_token_ms | prefill_to_first_token_ms | Source                                                                                       |
| ------------ | ------------------------------------------- | --------- | ------ | -------------- | ------------------- | ------------------- | ------------------------- | ---------------------------- | ------------------------ | -------------------------------------------------------------------------------------------- |
| Qwen3.5-Omni | thinker-only, sample=001-1, warmup=002-1    | 1         | 0      | 1.978          | 4                   | 8698                | 1959.392                  | 1483.578                     | 401.589                  | local H20 GPU7, qwen35_s2t_align/sglang-single-001-1-fps1-postmedia-20260709-080452          |

Single Request Stage Profile (video_fps=1, sample=001-1)

| Stage         | Interval                                                   | total_ms |
| ------------- | ---------------------------------------------------------- | -------- |
| preprocessing | stage_input_received->stage_complete                       | 738.749  |
| preprocessing | preprocess_media_load_start->preprocess_media_load_end     | 473.605  |
| preprocessing | preprocess_hf_processor_start->preprocess_hf_processor_end | 231.383  |
| image_encoder | stage_input_received->stage_complete                       | 517.433  |
| audio_encoder | stage_input_received->stage_complete                       | 516.063  |
| mm_aggregate  | stage_input_received->stage_complete                       | 516.008  |
| thinker       | scheduler_prefill_start->scheduler_first_emit              | 401.589  |
| decode        | stage_input_received->stage_complete                       | 0.392    |

Single Request Critical Path To First Token (video_fps=1, sample=001-1)

This table reconciles `post_media_to_first_token_ms` with the stage profile.
It starts at `preprocess_hf_processor_start` and follows the critical path to
`scheduler_first_emit`. Stage wall times are not directly additive because
encoder stages run in parallel and `mm_aggregate` waits for upstream results.

| Segment                                                      | delta_ms |
| ------------------------------------------------------------ | -------- |
| preprocess_hf_processor_start -> preprocess_hf_processor_end | 231.383  |
| preprocess_hf_processor_end -> preprocessing stage_complete  | 31.870   |
| preprocessing stage_complete -> audio_encoder input          | 293.475  |
| audio_encoder input -> audio_encoder stage_complete          | 516.063  |
| audio_encoder stage_complete -> mm_aggregate stage_complete  | 0.558    |
| mm_aggregate stage_complete -> thinker input                 | 0.114    |
| thinker input -> scheduler_prefill_start                     | 8.526    |
| scheduler_prefill_start -> scheduler_first_emit              | 401.589  |
| **post_media_to_first_token_ms**                             | **1483.578** |

Qwen3.5-Omni Four-Stage Optimization Result (H20, 2026-07-11)

Accepted configuration: preprocessed media input, native SGLang vision encoder
with grouped SDPA, BF16 encoder payloads, segmented SHM relay with mmap reads,
Hopper FA3 thinker attention, 8192-token chunked prefill, CUDA graph on,
torch.compile on, radix cache off, image-encoder batch dedup off.

Single-request profile (`002-1` warmup, `001-1` measured):

| Mode               | latency_s | request_to_first_token_ms | post_media_to_first_token_ms | preprocess_ms | hf_processor_ms | image_encoder_ms | audio_encoder_ms | prefill_execute_ms |
| ------------------ | --------- | ------------------------- | ---------------------------- | ------------- | --------------- | ---------------- | ---------------- | ------------------ |
| cold media/default | 1.082     | 1057.454                  | 923.235                      | 254.945       | 94.296          | 351.698          | 394.459          | 376.524            |
| warm media mean    | 0.891     | 865.893                   | 856.121                      | 123.359       | 90.701          | 299.712          | 343.007          | 365.531            |

CI-50 summary:

| concurrency | completed | failed | accuracy | latency_mean_s | latency_p95_s | throughput_qps | request_to_first_token_mean_ms | Source |
| ----------- | --------- | ------ | -------- | -------------- | ------------- | -------------- | ------------------------------ | ------ |
| 1           | 50        | 0      | 58.00%   | 1.050          | 1.163         | n/a            | n/a                            | `qwen35_s2t_align/sglang-final-accepted-20260711-025518/ci50-c1.json` |
| 8           | 50        | 0      | 58.00%   | 5.131          | 7.699         | 1.502          | 4217.125                       | `qwen35_s2t_align/sglang-final-accepted-20260711-025518/ci50-c8` |

Accuracy under concurrency is not deterministic for this branch. Repeated
identical-request and CI-50 runs can change answer distributions even on the
pre-optimization `fc34ac` baseline, with radix cache and image batch dedup both
disabled. Use repeated runs or vLLM alignment to evaluate accuracy changes;
single c=8 accuracy should not be treated as a stable performance signal.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.benchmarker.utils import wait_for_service
from benchmarks.dataset.videomme import (
    DEFAULT_VIDEOAMME_REPO_ID as _VIDEOAMME_DEFAULT_REPO,
)
from benchmarks.dataset.videomme import VideoAMMESample, load_videoamme_samples
from benchmarks.eval.benchmark_omni_videomme import (
    VideoEvalConfig,
    add_video_eval_args,
    run_video_eval,
    video_eval_config_from_args,
)
from benchmarks.metrics.performance import print_speed_summary
from benchmarks.metrics.video import print_videomme_accuracy_summary
from benchmarks.metrics.wer import print_wer_summary
from benchmarks.tasks.video_understanding import VIDEOAMME_REQUEST_TEXT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)


async def run_videoamme_eval(
    config: VideoEvalConfig,
    *,
    samples: list[VideoAMMESample] | None = None,
    compute_wer: bool = True,
) -> dict:
    return await run_video_eval(
        config,
        samples=samples,
        load_samples=load_videoamme_samples,
        task_label="Video-AMME",
        output_filename="videoamme_results.json",
        audio_output_dir_default="results/videoamme_audio",
        enable_audio_input=True,
        fixed_prompt=VIDEOAMME_REQUEST_TEXT,
        compute_wer=compute_wer,
    )


def _config_from_args(args: argparse.Namespace) -> VideoEvalConfig:
    return video_eval_config_from_args(args)


async def benchmark(args: argparse.Namespace) -> dict:
    config = _config_from_args(args)
    results = await run_videoamme_eval(config)
    print_videomme_accuracy_summary(
        results["summary"],
        config.model,
        title="Video-AMME Accuracy",
    )
    print_speed_summary(
        results["speed"],
        config.model,
        config.max_concurrency,
        title="Video-AMME Speed",
    )
    if "wer" in results:
        print_wer_summary(results["wer"]["summary"], config.model)
    return results


def _add_preprocessed_media_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--preprocessed-video-dir",
        type=str,
        default=None,
        help=(
            "Directory of predecoded/sampled/resized video .pt files. When set, "
            "send preprocessed_videos instead of raw video paths."
        ),
    )
    parser.add_argument(
        "--preprocessed-audio-dir",
        type=str,
        default=None,
        help=(
            "Directory of predecoded 16 kHz audio .pt files. When set, send "
            "preprocessed_audios instead of raw audio paths."
        ),
    )
    parser.add_argument(
        "--reuse-preprocessed-media",
        action="store_true",
        help=(
            "Keep path-backed preprocessed video/audio objects in the "
            "preprocessor CPU process for reuse across repeated requests."
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Video-AMME benchmark for video + audio question models."
    )
    add_video_eval_args(
        parser,
        repo_help=(
            "HuggingFace dataset repo for Video-AMME. "
            f"Defaults to {_VIDEOAMME_DEFAULT_REPO}."
        ),
    )
    _add_preprocessed_media_args(parser)
    args = parser.parse_args()

    wait_for_service(args.base_url or f"http://{args.host}:{args.port}")
    asyncio.run(benchmark(args))


if __name__ == "__main__":
    main()
