# Qwen3.5-Omni Stage Boundary Bottleneck Ledger

生成时间 UTC：`2026-06-21T02:00:23.077432+00:00`。
工作目录：`/home/gangouyu/sglang-omni`。

这份台账把 stage interaction 的每一条边界转换为可复核的瓶颈判定：
当前是否是瓶颈、证据数字是什么、对优化和 headline 的约束是什么。

## 1. Gate

| Gate | Value |
| --- | ---: |
| Ready | True |
| Checks | 12/12 |
| Required failures | 0 |
| Ledger rows | 37 |
| Pressure transition rows | 11 |
| Recommended SGLang window | `c4-c8` |
| Saturation boundary | `c16` |
| vLLM c8 scope | `offline_diagnostic_until_online_ingress_artifacts` |

## 2. Reviewer 读法

- `healthy` 表示该 stage 边界不是当前优化优先级。
- `queue_limited` 表示并发压力主要进入 admission/queue，不等于实际 preprocessing compute 变慢。
- `contention_regression` 表示反例 recipe，不能当作优化结论。
- `prompt_feed_limited` 表示 vLLM offline runner 的 prompt/feed 限制，不能外推为 online serving parity。
- `diagnostic_only` 表示只用于定位瓶颈转移，不能直接提升 headline。

## 3. Boundary Ledger

| ID | Runtime | Case | Boundary | Status | Verdict | Evidence | Decision | Scope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| boundary-01 | sglang | Video-AMME ci-50 c=1 | request_admission_to_preprocessing | healthy | not_current_bottleneck | preproc_lifecycle=294.8ms; actual_preproc=n/a; queue_est=n/a; top_stage=talker_ar 444/1516ms | Keep current recipe; this boundary is not the limiting stage under this condition. | SGLang stage connection health |
| boundary-02 | sglang | Video-AMME ci-50 c=1 | talker_to_code2wav_stream | healthy | not_current_bottleneck | hop_p95=15.5ms; talker_p95=1516.5ms; decode_avg=n/a | No tuning priority on the stream hop; keep watching p95 while optimizing admission and Talker cadence. | SGLang stage connection health |
| boundary-03 | sglang | Video-AMME ci-50 c=1 | code2wav_collect_to_decode | healthy | not_current_bottleneck | decode_avg/p95=14.3/17.8ms; collect_minus_decode=12.1ms | Do not optimize vocoder decode first; collect wait and upstream chunk cadence dominate the window. | SGLang stage connection health |
| boundary-04 | sglang | Video-AMME ci-50 c=2 | request_admission_to_preprocessing | healthy | not_current_bottleneck | preproc_lifecycle=334.1ms; actual_preproc=n/a; queue_est=n/a; top_stage=talker_ar 526/2024ms | Keep current recipe; this boundary is not the limiting stage under this condition. | SGLang stage connection health |
| boundary-05 | sglang | Video-AMME ci-50 c=2 | talker_to_code2wav_stream | healthy | not_current_bottleneck | hop_p95=16.1ms; talker_p95=2024.2ms; decode_avg=n/a | No tuning priority on the stream hop; keep watching p95 while optimizing admission and Talker cadence. | SGLang stage connection health |
| boundary-06 | sglang | Video-AMME ci-50 c=2 | code2wav_collect_to_decode | healthy | not_current_bottleneck | decode_avg/p95=16.0/20.2ms; collect_minus_decode=15.9ms | Do not optimize vocoder decode first; collect wait and upstream chunk cadence dominate the window. | SGLang stage connection health |
| boundary-07 | sglang | Video-AMME ci-50 c=4 | request_admission_to_preprocessing | healthy | not_current_bottleneck | preproc_lifecycle=486.9ms; actual_preproc=295.8ms; queue_est=191.1ms; top_stage=talker_ar 663/2628ms | Keep current recipe; this boundary is not the limiting stage under this condition. | SGLang stage connection health |
| boundary-08 | sglang | Video-AMME ci-50 c=4 | talker_to_code2wav_stream | healthy | not_current_bottleneck | hop_p95=17.8ms; talker_p95=2627.9ms; decode_avg=n/a | No tuning priority on the stream hop; keep watching p95 while optimizing admission and Talker cadence. | SGLang stage connection health |
| boundary-09 | sglang | Video-AMME ci-50 c=4 | code2wav_collect_to_decode | healthy | not_current_bottleneck | decode_avg/p95=16.0/21.8ms; collect_minus_decode=29.6ms | Do not optimize vocoder decode first; collect wait and upstream chunk cadence dominate the window. | SGLang stage connection health |
| boundary-10 | sglang | Video-AMME ci-50 c=8 | request_admission_to_preprocessing | queue_limited | admission_queue_bottleneck | preproc_lifecycle=1226.6ms; actual_preproc=289.2ms; queue_est=937.4ms; top_stage=preprocessing 1227/2164ms | Use c4-c8 as serving window; treat c16 as saturation boundary and avoid claiming more concurrency as better. | SGLang high-concurrency limit / serving-window decision |
| boundary-11 | sglang | Video-AMME ci-50 c=8 | talker_to_code2wav_stream | healthy | not_current_bottleneck | hop_p95=20.4ms; talker_p95=4418.0ms; decode_avg=n/a | No tuning priority on the stream hop; keep watching p95 while optimizing admission and Talker cadence. | SGLang stage connection health |
| boundary-12 | sglang | Video-AMME ci-50 c=8 | code2wav_collect_to_decode | healthy | not_current_bottleneck | decode_avg/p95=17.2/25.9ms; collect_minus_decode=50.9ms | Do not optimize vocoder decode first; collect wait and upstream chunk cadence dominate the window. | SGLang stage connection health |
| boundary-13 | sglang | Video-AMME ci-50 c=16 | request_admission_to_preprocessing | queue_limited | admission_queue_bottleneck | preproc_lifecycle=4395.1ms; actual_preproc=304.6ms; queue_est=4090.5ms; top_stage=preprocessing 4395/5884ms | Use c4-c8 as serving window; treat c16 as saturation boundary and avoid claiming more concurrency as better. | SGLang high-concurrency limit / serving-window decision |
| boundary-14 | sglang | Video-AMME ci-50 c=16 | talker_to_code2wav_stream | healthy | not_current_bottleneck | hop_p95=19.7ms; talker_p95=3636.5ms; decode_avg=n/a | No tuning priority on the stream hop; keep watching p95 while optimizing admission and Talker cadence. | SGLang stage connection health |
| boundary-15 | sglang | Video-AMME ci-50 c=16 | code2wav_collect_to_decode | healthy | not_current_bottleneck | decode_avg/p95=16.7/23.9ms; collect_minus_decode=44.3ms | Do not optimize vocoder decode first; collect wait and upstream chunk cadence dominate the window. | SGLang stage connection health |
| boundary-16 | sglang | Video-AMME ci-50 c=8 | preprocessing_to_encoder_thinker | contention_regression | negative_optimization | baseline_qps=2.540; candidate_qps=1.642; delta=-35.4% | Keep preprocessing concurrency at 1 unless placement/admission is redesigned; preproc=2 is a negative recipe. | SGLang anti-recipe guardrail |
| boundary-17 | sglang | synthetic_short c=1 | talker_to_code2wav_stream | healthy | not_current_bottleneck | hop_p95=14.9ms; talker_p95=903.7ms; decode_avg=12.6ms | No tuning priority on the stream hop; keep watching p95 while optimizing admission and Talker cadence. | SGLang stage connection health |
| boundary-18 | sglang | synthetic_short c=4 | talker_to_code2wav_stream | healthy | not_current_bottleneck | hop_p95=20.1ms; talker_p95=2028.8ms; decode_avg=16.5ms | No tuning priority on the stream hop; keep watching p95 while optimizing admission and Talker cadence. | SGLang stage connection health |
| boundary-19 | sglang | synthetic_short c=8 | talker_to_code2wav_stream | healthy | not_current_bottleneck | hop_p95=21.2ms; talker_p95=2501.6ms; decode_avg=15.3ms | No tuning priority on the stream hop; keep watching p95 while optimizing admission and Talker cadence. | SGLang stage connection health |
| boundary-20 | sglang | synthetic_long c=1 | talker_to_code2wav_stream | healthy | not_current_bottleneck | hop_p95=15.0ms; talker_p95=9383.9ms; decode_avg=12.7ms | No tuning priority on the stream hop; keep watching p95 while optimizing admission and Talker cadence. | SGLang stage connection health |
| boundary-21 | sglang | synthetic_long c=4 | talker_to_code2wav_stream | healthy | not_current_bottleneck | hop_p95=20.4ms; talker_p95=17947.2ms; decode_avg=14.0ms | No tuning priority on the stream hop; keep watching p95 while optimizing admission and Talker cadence. | SGLang stage connection health |
| boundary-22 | sglang | synthetic_long c=8 | talker_to_code2wav_stream | healthy | not_current_bottleneck | hop_p95=24.0ms; talker_p95=26198.6ms; decode_avg=14.5ms | No tuning priority on the stream hop; keep watching p95 while optimizing admission and Talker cadence. | SGLang stage connection health |
| boundary-23 | vllm | vLLM-c1 | thinker_to_talker | healthy | not_current_bottleneck | thinker_to_talker_p95=1.0ms; diagnosis=n/a | Keep current recipe; this boundary is not the limiting stage under this condition. | vLLM stage connection health |
| boundary-24 | vllm | vLLM-c1 | talker_to_code2wav | healthy | not_current_bottleneck | drain_p95=16.0ms; feed_to_first_codec_p95=77.5ms; diagnosis=n/a | Keep current recipe; this boundary is not the limiting stage under this condition. | vLLM stage connection health |
| boundary-25 | vllm | vLLM-c1 | runner_to_engine_admission | healthy | not_current_bottleneck | admission_avg/p95=0.0/0.0ms; runner_overhead=n/a; engine_qps=n/a | Keep current recipe; this boundary is not the limiting stage under this condition. | vLLM stage connection health |
| boundary-26 | vllm | vLLM-c4 | thinker_to_talker | healthy | not_current_bottleneck | thinker_to_talker_p95=1.0ms; diagnosis=prompt_feed_limited | Keep current recipe; this boundary is not the limiting stage under this condition. | strict c4 baseline comparison |
| boundary-27 | vllm | vLLM-c4 | talker_to_code2wav | healthy | not_current_bottleneck | drain_p95=17.5ms; feed_to_first_codec_p95=77.7ms; diagnosis=prompt_feed_limited | Keep current recipe; this boundary is not the limiting stage under this condition. | strict c4 baseline comparison |
| boundary-28 | vllm | vLLM-c4 | runner_to_engine_admission | prompt_feed_limited | offline_runner_prompt_feed_bottleneck | admission_avg/p95=15110.8/19135.8ms; runner_overhead=76.7%; engine_qps=0.1536 | Use as vLLM offline runner diagnosis; do not compare c8 wall time as online serving parity. | strict c4 baseline comparison |
| boundary-29 | vllm | vLLM-c8 | thinker_to_talker | healthy | not_current_bottleneck | thinker_to_talker_p95=1.0ms; diagnosis=prompt_feed_limited | Keep current recipe; this boundary is not the limiting stage under this condition. | vLLM stage connection health |
| boundary-30 | vllm | vLLM-c8 | talker_to_code2wav | healthy | not_current_bottleneck | drain_p95=16.0ms; feed_to_first_codec_p95=70.0ms; diagnosis=prompt_feed_limited | Keep current recipe; this boundary is not the limiting stage under this condition. | vLLM stage connection health |
| boundary-31 | vllm | vLLM-c8 | runner_to_engine_admission | prompt_feed_limited | offline_runner_prompt_feed_bottleneck | admission_avg/p95=33314.0/43972.7ms; runner_overhead=81.8%; engine_qps=0.1622 | Use as vLLM offline runner diagnosis; do not compare c8 wall time as online serving parity. | offline admission diagnosis |
| boundary-32 | vllm | vLLM-c8-prebuild-w1 | thinker_to_talker | healthy | not_current_bottleneck | thinker_to_talker_p95=4.0ms; diagnosis=engine_or_workload_limited | Keep current recipe; this boundary is not the limiting stage under this condition. | offline diagnostic only |
| boundary-33 | vllm | vLLM-c8-prebuild-w1 | talker_to_code2wav | bottleneck | observed_bottleneck | drain_p95=88.7ms; feed_to_first_codec_p95=549.0ms; diagnosis=engine_or_workload_limited | Treat as a post-admission tail exposed by diagnostic mode; isolate before making a headline claim. | offline diagnostic only |
| boundary-34 | vllm | vLLM-c8-prebuild-w1 | runner_to_engine_admission | diagnostic_only | offline_diagnostic_not_parity | admission_avg/p95=4439.7/5425.0ms; runner_overhead=77.6%; engine_qps=0.5391 | Keep as offline diagnostic; do not promote c8 parity without online ingress artifacts. | offline diagnostic only |
| boundary-35 | vllm | vLLM-c8-prebuild-w4 | thinker_to_talker | healthy | not_current_bottleneck | thinker_to_talker_p95=3.9ms; diagnosis=engine_or_workload_limited | Monitor engine/talker tail after prompt feed is removed; keep caveat attached. | offline diagnostic only |
| boundary-36 | vllm | vLLM-c8-prebuild-w4 | talker_to_code2wav | watch | tail_watch | drain_p95=123.4ms; feed_to_first_codec_p95=509.7ms; diagnosis=engine_or_workload_limited | Monitor engine/talker tail after prompt feed is removed; keep caveat attached. | offline diagnostic only |
| boundary-37 | vllm | vLLM-c8-prebuild-w4 | runner_to_engine_admission | diagnostic_only | offline_diagnostic_not_parity | admission_avg/p95=4089.0/4891.5ms; runner_overhead=65.6%; engine_qps=0.5360 | Keep as offline diagnostic; do not promote c8 parity without online ingress artifacts. | offline diagnostic only |

## 4. Pressure Propagation Matrix

这张表回答的是：并发、文本长度或 vLLM offline runner 形态变化时，压力沿哪个 stage 边界传导。
它只使用已审计 summary/interaction 数据派生相邻档位 delta，不引入新的 benchmark 数字。

| ID | Runtime | Workload | Transition | Axis | Verdict | Evidence | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pressure-sglang-c1-c2 | sglang | Video-AMME ci-50 | c1 -> c2 | concurrency | scales_without_boundary_bottleneck | QPS +73.0%; latency_p95 +29.8%; RTF_p95 -4.4%; queue_delta n/a; queue_share_after n/a; hop_p95_delta +0.6ms; decode_p95_delta +2.4ms | Throughput improves without moving the main bottleneck to stage handoff or decode. |
| pressure-sglang-c2-c4 | sglang | Video-AMME ci-50 | c2 -> c4 | concurrency | scales_without_boundary_bottleneck | QPS +54.8%; latency_p95 +16.3%; RTF_p95 +29.4%; queue_delta n/a; queue_share_after 39.3%; hop_p95_delta +1.7ms; decode_p95_delta +1.6ms | Throughput improves without moving the main bottleneck to stage handoff or decode. |
| pressure-sglang-c4-c8 | sglang | Video-AMME ci-50 | c4 -> c8 | concurrency | usable_high_concurrency_window | QPS +24.8%; latency_p95 +61.1%; RTF_p95 +75.8%; queue_delta +746.3ms; queue_share_after 76.4%; hop_p95_delta +2.6ms; decode_p95_delta +4.1ms | Keep c8 as the throughput-oriented serving edge; queue pressure is visible but throughput still improves. |
| pressure-sglang-c8-c16 | sglang | Video-AMME ci-50 | c8 -> c16 | concurrency | saturation_boundary | QPS -5.2%; latency_p95 +34.1%; RTF_p95 +137.0%; queue_delta +3153.1ms; queue_share_after 93.1%; hop_p95_delta -0.7ms; decode_p95_delta -2.0ms | Do not use c16 as the serving optimum: throughput falls while admission queue and p95 RTF rise. |
| pressure-synthetic-short-c1-c4 | sglang | synthetic_short | c1 -> c4 | short_text_concurrency | short_text_scales_below_realtime | QPS +92.2%; latency_p95 +122.5%; RTF_p95_after 0.4; talker_p95_delta +1125.1ms; hop_p95_delta +5.2ms; decode_avg_delta +3.9ms | Synthetic speech remains below real-time; the handoff delta stays small, so length pressure maps to Talker cadence rather than code2wav decode. |
| pressure-synthetic-short-c4-c8 | sglang | synthetic_short | c4 -> c8 | short_text_concurrency | short_text_scales_below_realtime | QPS +34.5%; latency_p95 +37.5%; RTF_p95_after 0.7; talker_p95_delta +472.8ms; hop_p95_delta +1.1ms; decode_avg_delta -1.2ms | Synthetic speech remains below real-time; the handoff delta stays small, so length pressure maps to Talker cadence rather than code2wav decode. |
| pressure-synthetic-long-c1-c4 | sglang | synthetic_long | c1 -> c4 | long_text_concurrency | long_text_realtime_guard_holds | QPS +108.3%; latency_p95 +90.4%; RTF_p95_after 0.3; talker_p95_delta +8563.3ms; hop_p95_delta +5.4ms; decode_avg_delta +1.3ms | Synthetic speech remains below real-time; the handoff delta stays small, so length pressure maps to Talker cadence rather than code2wav decode. |
| pressure-synthetic-long-c4-c8 | sglang | synthetic_long | c4 -> c8 | long_text_concurrency | long_text_realtime_guard_holds | QPS +33.5%; latency_p95 +46.0%; RTF_p95_after 0.5; talker_p95_delta +8251.4ms; hop_p95_delta +3.6ms; decode_avg_delta +0.5ms | Synthetic speech remains below real-time; the handoff delta stays small, so length pressure maps to Talker cadence rather than code2wav decode. |
| pressure-vllm-vLLM-c4-to-vLLM-c8 | vllm | Video-AMME ci-50 offline | vLLM-c4 -> vLLM-c8 | original_concurrency | offline_prompt_feed_limited | wall_qps +5.6%; engine_qps +5.6%; admission_p95 +129.8%; runner_overhead +5.1pp; talker_drain_p95_delta -1.5ms; after_diagnosis=prompt_feed_limited | Do not use original c8 wall QPS as online parity; admission span grows sharply. |
| pressure-vllm-vLLM-c8-to-vLLM-c8-prebuild-w4 | vllm | Video-AMME ci-50 offline | vLLM-c8 -> vLLM-c8-prebuild-w4 | prebuild_prompt_feed | diagnostic_bottleneck_shift | wall_qps +31.1%; engine_qps +230.4%; admission_p95 -88.9%; runner_overhead -16.2pp; talker_drain_p95_delta +107.4ms; after_diagnosis=engine_or_workload_limited | Prebuild removes most admission span and exposes later engine/talker tail; keep offline caveat. |
| pressure-vllm-vLLM-c8-prebuild-w1-to-vLLM-c8-prebuild-w4 | vllm | Video-AMME ci-50 offline | vLLM-c8-prebuild-w1 -> vLLM-c8-prebuild-w4 | prebuild_worker_parallelism | runner_parallelism_helps_wall_not_engine | wall_qps +49.8%; engine_qps -0.6%; admission_p95 -9.8%; runner_overhead -12.0pp; talker_drain_p95_delta +34.7ms; after_diagnosis=engine_or_workload_limited | w4 improves runner wall time while engine QPS stays flat; not a serving-parity headline. |

## 5. Machine Checks

| Status | Required | Check | Evidence |
| --- | --- | --- | --- |
| PASS | yes | interaction rows are represented | ledger_rows=37, source_rows=37 |
| PASS | yes | SGLang c8/c16 queue boundary is explicit | c8_queue_pct=30.594223237597905, c16_queue_pct=67.43293768545993 |
| PASS | yes | SGLang stream hop stays non-bottleneck | all SGLang talker_to_code2wav_stream rows are healthy |
| PASS | yes | SGLang code2wav decode stays non-bottleneck | all SGLang code2wav_collect_to_decode rows are healthy |
| PASS | yes | preprocessing concurrency anti-recipe is preserved | preproc=2 row is marked contention_regression |
| PASS | yes | long synthetic c8 remains faster than real time | long_c8_rtf=0.4932 |
| PASS | yes | vLLM c8 prompt-feed limit is explicit | vLLM-c8 runner_to_engine_admission is prompt_feed_limited |
| PASS | yes | vLLM prebuild w4 stays diagnostic-only | vllm_w4_scope=offline_diagnostic_only |
| PASS | yes | every ledger row has decision evidence and scope | decision/evidence/claim_scope populated for all rows; malformed_evidence_tokens=[] |
| PASS | yes | claim verifier remains green | claims_passed=True, failed=0 |
| PASS | yes | pressure propagation matrix covers concurrency, length, and vLLM diagnostics | pressure_rows=11, required_ids_present=['pressure-sglang-c1-c2', 'pressure-sglang-c2-c4', 'pressure-sglang-c4-c8', 'pressure-sglang-c8-c16']... |
| PASS | yes | pressure propagation conclusions preserve serving-window and diagnostic boundaries | c8_to_c16_qps_delta=-5.236220472440945; c8_to_c16_queue_delta=3153.075; long_c8_rtf_p95=0.5001; vllm_prebuild_verdict=diagnostic_bottleneck_shift |
