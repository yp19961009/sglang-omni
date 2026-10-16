# Qwen3.5-Omni Pressure × Stage Heatmap

状态：证据门已就绪；更新后的目标不再等待 6.21 晚间，后续变更必须重跑 full audit。
工作目录：`/home/gangouyu/sglang-omni`。

用途：把单并发/高并发、短/长文本语音、vLLM original/prebuild 诊断压到同一张 stage 热力表。
这份报告只聚合已经审计通过的 JSON，不引入新的 benchmark 数字。

## 1. Gate

| Gate | Value |
| --- | ---: |
| ready | `True` |
| rows | `15` |
| SGLang Video-AMME rows | `5` |
| synthetic rows | `6` |
| vLLM rows | `4` |
| checks | `11/11` |
| required failures | `0` |

## 2. Heatmap 总览

| Pressure | Runtime | Key Metrics | Admission / Preprocess | Talker | Handoff | Code2wav | Runner / Engine | Bottleneck | Decision | Caveat |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Video-AMME c=1 | sglang | QPS=0.760; lat_p95=2.406s; RTF_p95=2.0199 | preproc_lifecycle=294.8ms / 22.4%; queue=n/a / n/a | talker=444.1ms / 33.7%; top_stage=talker_ar 444/1516ms | talker->code2wav hop_p95=15.5ms; handoff is not the current bottleneck | decode=14.3ms / 1.1%; collect=26.4ms | n/a | talker_ar_tail | latency-first guard | Do not describe low-concurrency Talker tail as high-concurrency queue saturation. |
| Video-AMME c=2 | sglang | QPS=1.315; lat_p95=3.124s; RTF_p95=1.9308 | preproc_lifecycle=334.1ms / 22.2%; queue=n/a / n/a | talker=525.8ms / 34.9%; top_stage=talker_ar 526/2024ms | talker->code2wav hop_p95=16.1ms; handoff is not the current bottleneck | decode=16.0ms / 1.1%; collect=31.9ms | n/a | talker_ar_tail | latency-first guard | Do not describe low-concurrency Talker tail as high-concurrency queue saturation. |
| Video-AMME c=4 | sglang | QPS=2.036; lat_p95=3.633s; RTF_p95=2.4982 | preproc_lifecycle=486.9ms / 25.2%; queue=191.1ms / 9.9% | talker=663.1ms / 34.4%; top_stage=talker_ar 663/2628ms | talker->code2wav hop_p95=17.8ms; handoff is not the current bottleneck | decode=16.0ms / 0.8%; collect=45.6ms | n/a | talker_ar_tail | balanced serving | Do not mix stress c4 and strict warmed c4 metrics. |
| Video-AMME c=8 | sglang | QPS=2.540; lat_p95=5.853s; RTF_p95=4.3924 | preproc_lifecycle=1226.6ms / 40.0%; queue=937.4ms / 30.6% | talker=982.7ms / 32.1%; top_stage=preprocessing 1227/2164ms | talker->code2wav hop_p95=20.4ms; handoff is not the current bottleneck | decode=17.2ms / 0.6%; collect=68.1ms | n/a | admission_queue_plus_talker_tail | throughput edge | Do not widen preprocessing concurrency without redesigning admission and placement. |
| Video-AMME c=16 | sglang | QPS=2.407; lat_p95=7.846s; RTF_p95=10.4087 | preproc_lifecycle=4395.1ms / 72.5%; queue=4090.5ms / 67.4% | talker=815.6ms / 13.4%; top_stage=preprocessing 4395/5884ms | talker->code2wav hop_p95=19.7ms; handoff is not the current bottleneck | decode=16.7ms / 0.3%; collect=61.0ms | n/a | saturation_boundary | saturation boundary | Do not present c16 as the high-concurrency optimum. |
| Synthetic short text c=1 | sglang | QPS=1.154; lat_p95=0.924s; RTF_p95=0.2112; words=12; audio_mean=4.2s | synthetic speech isolates thinker/talker/code2wav; no Video-AMME preprocessing queue claim | talker=849.0ms / 98.0%; RTF below 1.0 | talker->code2wav hop_p95=14.9ms; handoff delta stays small versus Talker cadence | decode=12.6ms / 1.5%; collect=27.0ms | n/a | faster_than_realtime | short-text speech guardrail | Do not replace full-set or Video-AMME headline with synthetic-only evidence. |
| Synthetic short text c=4 | sglang | QPS=2.218; lat_p95=2.056s; RTF_p95=0.4388; words=12; audio_mean=4.3s | synthetic speech isolates thinker/talker/code2wav; no Video-AMME preprocessing queue claim | talker=1722.6ms / 97.4%; RTF below 1.0 | talker->code2wav hop_p95=20.1ms; handoff delta stays small versus Talker cadence | decode=16.5ms / 0.9%; collect=57.9ms | n/a | faster_than_realtime | short-text speech guardrail | Do not replace full-set or Video-AMME headline with synthetic-only evidence. |
| Synthetic short text c=8 | sglang | QPS=2.983; lat_p95=2.828s; RTF_p95=0.7440; words=12; audio_mean=4.3s | synthetic speech isolates thinker/talker/code2wav; no Video-AMME preprocessing queue claim | talker=2205.9ms / 83.6%; RTF below 1.0 | talker->code2wav hop_p95=21.2ms; handoff delta stays small versus Talker cadence | decode=15.3ms / 0.6%; collect=73.4ms | n/a | faster_than_realtime | short-text speech guardrail | Do not replace full-set or Video-AMME headline with synthetic-only evidence. |
| Synthetic long text c=1 | sglang | QPS=0.109; lat_p95=9.465s; RTF_p95=0.1776; words=139; audio_mean=51.9s | synthetic speech isolates thinker/talker/code2wav; no Video-AMME preprocessing queue claim | talker=9091.2ms / 99.2%; RTF below 1.0 | talker->code2wav hop_p95=15.0ms; handoff delta stays small versus Talker cadence | decode=12.7ms / 0.1%; collect=27.8ms | n/a | faster_than_realtime | long-text speech guardrail | Do not replace full-set or Video-AMME headline with synthetic-only evidence. |
| Synthetic long text c=4 | sglang | QPS=0.227; lat_p95=18.025s; RTF_p95=0.3373; words=139; audio_mean=52.6s | synthetic speech isolates thinker/talker/code2wav; no Video-AMME preprocessing queue claim | talker=17463.9ms / 99.5%; RTF below 1.0 | talker->code2wav hop_p95=20.4ms; handoff delta stays small versus Talker cadence | decode=14.0ms / 0.1%; collect=56.1ms | n/a | faster_than_realtime | long-text speech guardrail | Do not replace full-set or Video-AMME headline with synthetic-only evidence. |
| Synthetic long text c=8 | sglang | QPS=0.303; lat_p95=26.318s; RTF_p95=0.5001; words=139; audio_mean=52.3s | synthetic speech isolates thinker/talker/code2wav; no Video-AMME preprocessing queue claim | talker=25572.3ms / 99.1%; RTF below 1.0 | talker->code2wav hop_p95=24.0ms; handoff delta stays small versus Talker cadence | decode=14.5ms / 0.1%; collect=81.7ms | n/a | long_speech_talker_ar_dominant_but_faster_than_realtime | long-text speech guardrail | Do not replace full-set or Video-AMME headline with synthetic-only evidence. |
| vLLM-c4 | vllm | QPS=0.154; lat_p95=3.525s; RTF_p95=3.0717 | runner_overhead=76.7%; admission_p95=19135.8ms | thinker->talker p95=1.0ms; talker->code2wav drain_p95=17.5ms | prompt-feed dominates before engine boundaries | talker/code2wav drain_p95=17.5ms | runner_QPS=0.1536; engine_QPS=0.1536 | prompt_feed_limited | offline prompt-feed diagnostic | Do not promote offline diagnostic rows to online serving parity without online ingress plus WER/ASR. |
| vLLM-c8 | vllm | QPS=0.162; lat_p95=3.260s; RTF_p95=3.1987 | runner_overhead=81.8%; admission_p95=43972.7ms | thinker->talker p95=1.0ms; talker->code2wav drain_p95=16.0ms | prompt-feed dominates before engine boundaries | talker/code2wav drain_p95=16.0ms | runner_QPS=0.1622; engine_QPS=0.1622 | prompt_feed_limited | offline prompt-feed diagnostic | Do not promote offline diagnostic rows to online serving parity without online ingress plus WER/ASR. |
| vLLM-c8-prebuild-w1 | vllm | QPS=0.539; lat_p95=7.009s; RTF_p95=6.2581 | runner_overhead=77.6%; admission_p95=5425.0ms | thinker->talker p95=4.0ms; talker->code2wav drain_p95=88.7ms | prebuild removes most admission span and exposes later engine/talker tail | talker/code2wav drain_p95=88.7ms | runner_QPS=0.1420; engine_QPS=0.5391 | engine_or_workload_limited | optimized offline diagnostic | Do not promote offline diagnostic rows to online serving parity without online ingress plus WER/ASR. |
| vLLM-c8-prebuild-w4 | vllm | QPS=0.536; lat_p95=7.730s; RTF_p95=7.0869 | runner_overhead=65.6%; admission_p95=4891.5ms | thinker->talker p95=3.9ms; talker->code2wav drain_p95=123.4ms | prebuild removes most admission span and exposes later engine/talker tail | talker/code2wav drain_p95=123.4ms | runner_QPS=0.2127; engine_QPS=0.5360 | engine_or_workload_limited | optimized offline diagnostic | Do not promote offline diagnostic rows to online serving parity without online ingress plus WER/ASR. |

## 3. 复核入口

| Pressure | Evidence | Rerun Command IDs |
| --- | --- | --- |
| Video-AMME c=1 | results/qwen35_sglang_mr8_stress_20260619/benchmark_audio_50_c1_warm_profile_skipwer/videoamme_results.json, results/qwen35_report_audit_20260619/stage_latency_budget.json, results/qwen35_report_audit_20260619/stage_boundary_bottleneck_ledger.json | `sglang_videoamme_stress`, `build_stage_latency_budget`, `build_stage_boundary_bottleneck_ledger` |
| Video-AMME c=2 | results/qwen35_sglang_mr8_stress_20260619/benchmark_audio_50_c2_warm_profile_skipwer/videoamme_results.json, results/qwen35_report_audit_20260619/stage_latency_budget.json, results/qwen35_report_audit_20260619/stage_boundary_bottleneck_ledger.json | `sglang_videoamme_stress`, `build_stage_latency_budget`, `build_stage_boundary_bottleneck_ledger` |
| Video-AMME c=4 | results/qwen35_sglang_mr8_stress_20260619/benchmark_audio_50_c4_profile_skipwer/videoamme_results.json, results/qwen35_report_audit_20260619/stage_latency_budget.json, results/qwen35_report_audit_20260619/stage_boundary_bottleneck_ledger.json | `sglang_videoamme_stress`, `build_stage_latency_budget`, `build_stage_boundary_bottleneck_ledger` |
| Video-AMME c=8 | results/qwen35_sglang_mr8_stress_20260619/benchmark_audio_50_c8_profile_skipwer/videoamme_results.json, results/qwen35_report_audit_20260619/stage_latency_budget.json, results/qwen35_report_audit_20260619/stage_boundary_bottleneck_ledger.json | `sglang_videoamme_stress`, `build_stage_latency_budget`, `build_stage_boundary_bottleneck_ledger` |
| Video-AMME c=16 | results/qwen35_sglang_mr8_stress_20260619/benchmark_audio_50_c16_profile_skipwer/videoamme_results.json, results/qwen35_report_audit_20260619/stage_latency_budget.json, results/qwen35_report_audit_20260619/stage_boundary_bottleneck_ledger.json | `sglang_videoamme_stress`, `build_stage_latency_budget`, `build_stage_boundary_bottleneck_ledger` |
| Synthetic short text c=1 | results/qwen35_synthetic_speech_20260619/short_c1/synthetic_speech_results.json, results/qwen35_report_audit_20260619/stage_latency_budget.json | `sglang_synthetic_text_to_speech`, `build_stage_latency_budget`, `build_tail_confidence_appendix` |
| Synthetic short text c=4 | results/qwen35_synthetic_speech_20260619/short_c4/synthetic_speech_results.json, results/qwen35_report_audit_20260619/stage_latency_budget.json | `sglang_synthetic_text_to_speech`, `build_stage_latency_budget`, `build_tail_confidence_appendix` |
| Synthetic short text c=8 | results/qwen35_synthetic_speech_20260619/short_c8/synthetic_speech_results.json, results/qwen35_report_audit_20260619/stage_latency_budget.json | `sglang_synthetic_text_to_speech`, `build_stage_latency_budget`, `build_tail_confidence_appendix` |
| Synthetic long text c=1 | results/qwen35_synthetic_speech_20260619/long_c1/synthetic_speech_results.json, results/qwen35_report_audit_20260619/stage_latency_budget.json | `sglang_synthetic_text_to_speech`, `build_stage_latency_budget`, `build_tail_confidence_appendix` |
| Synthetic long text c=4 | results/qwen35_synthetic_speech_20260619/long_c4/synthetic_speech_results.json, results/qwen35_report_audit_20260619/stage_latency_budget.json | `sglang_synthetic_text_to_speech`, `build_stage_latency_budget`, `build_tail_confidence_appendix` |
| Synthetic long text c=8 | results/qwen35_synthetic_speech_20260619/long_c8/synthetic_speech_results.json, results/qwen35_report_audit_20260619/stage_latency_budget.json | `sglang_synthetic_text_to_speech`, `build_stage_latency_budget`, `build_tail_confidence_appendix` |
| vLLM-c4 | results/qwen35_vllm_videoamme_ci50_official_talker_compile_c4_20260618_202106/benchmark_audio_50_c4_offline_compile/videoamme_results.json, results/qwen35_report_audit_20260619/vllm_admission_diagnosis.json, results/qwen35_report_audit_20260619/stage_latency_budget.json | `vllm_c4_original`, `summarize_vllm_log_stages`, `diagnose_vllm_admission`, `build_stage_latency_budget` |
| vLLM-c8 | results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_20260619_20260619_222434/benchmark_audio_50_c8_offline_compile/videoamme_results.json, results/qwen35_report_audit_20260619/vllm_admission_diagnosis.json, results/qwen35_report_audit_20260619/stage_latency_budget.json | `vllm_c8_original`, `summarize_vllm_log_stages`, `diagnose_vllm_admission`, `build_stage_latency_budget` |
| vLLM-c8-prebuild-w1 | results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuild_20260620_002020/benchmark_audio_50_c8_offline_compile/videoamme_results.json, results/qwen35_report_audit_20260619/vllm_admission_diagnosis.json, results/qwen35_report_audit_20260619/stage_latency_budget.json | `summarize_vllm_log_stages`, `diagnose_vllm_admission`, `build_stage_latency_budget` |
| vLLM-c8-prebuild-w4 | results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuildw4_20260620_005346/benchmark_audio_50_c8_offline_compile/videoamme_results.json, results/qwen35_report_audit_20260619/vllm_admission_diagnosis.json, results/qwen35_report_audit_20260619/stage_latency_budget.json | `vllm_c8_prebuild_w4`, `summarize_vllm_log_stages`, `diagnose_vllm_admission`, `build_stage_latency_budget` |

## 4. 读法

- SGLang c=1/c=2：低并发主要看 Talker AR tail，不能说成 admission queue 饱和。
- SGLang c=8：吞吐达到当前峰值，但 queue 已显性化；这是当前 high-concurrency serving edge。
- SGLang c=16：吞吐低于 c=8 且 queue/RTF tail 上升，是 saturation boundary。
- synthetic short/long：用来隔离 thinker/talker/code2wav；长文本 c=8 仍快于实时，优先瓶颈不是 vocoder decode。
- vLLM original c=8：offline runner prompt build/feed 限制 admission；prebuild w4 是 optimized offline diagnostic，不是 online serving parity。

## 5. 机器证据

- `results/qwen35_report_audit_20260619/pressure_stage_heatmap.json`
- `results/qwen35_report_audit_20260619/stage_latency_budget.json`
- `results/qwen35_report_audit_20260619/stage_boundary_bottleneck_ledger.json`
- `results/qwen35_report_audit_20260619/stage_reproduction_drilldown.json`
- `results/qwen35_report_audit_20260619/tail_confidence_appendix.json`
- Rebuild command ID: `build_stage_route_decision_matrix`
