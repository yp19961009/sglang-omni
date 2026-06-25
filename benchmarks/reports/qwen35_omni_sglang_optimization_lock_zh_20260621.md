# Qwen3.5-Omni SGLang 优化锁定矩阵

生成时间 UTC：`2026-06-21T02:00:21.435465+00:00`。
工作目录：`/home/gangouyu/sglang-omni`。

这页专门锁定 SGLang-Omni 当前 best recipe、推荐运行窗口、stage
连接健康度和反例实验。目的不是新增 benchmark 数字，而是防止 reviewer
误以为 SGLang 侧只是偶然跑快，或把负优化配置误当成下一步主线。

## 1. 锁定结论

- ready：`True`，checks：`26/26`，required failures：`0`。
- SGLang image：`frankleeeee/sglang-omni:dev`。
- SGLang image id：`sha256:be7e72126f525c3767008a73de16f400f974a09db431ded3c52bd48370941a84`。
- 推荐窗口：c4-c8 warmed serving; c8 peak throughput; c16 saturation boundary。
- recipe contract：compiled/graph SGLang recipe with serial preprocessing and 16GiB preprocessing cache。
- c=8 是当前吞吐峰值；c=16 是饱和边界，不作为推荐 serving 点。
- `PREPROCESSING_MAX_CONCURRENCY=2/4` 已被锁为反例，不应作为默认优化方向。

## 2. Gate 明细

| Status | Required | Gate | Evidence |
| --- | --- | --- | --- |
| PASS | yes | SGLang image locked | image=frankleeeee/sglang-omni:dev, id=sha256:be7e72126f525c3767008a73de16f400f974a09db431ded3c52bd48370941a84, created=2026-02-05T09:30:27.304070315Z |
| PASS | yes | launch script present | /home/gangouyu/sglang-omni/examples/launch_qwen35_omni_speech_server_container.sh |
| PASS | yes | launch defaults to serial preprocessing | examples/launch_qwen35_omni_speech_server_container.sh contains 'PREPROCESSING_MAX_CONCURRENCY="${PREPROCESSING_MAX_CONCURRENCY:-1}"' |
| PASS | yes | launch exposes code2wav compile toggle | examples/launch_qwen35_omni_speech_server_container.sh contains 'NO_CODE2WAV_TORCH_COMPILE="${NO_CODE2WAV_TORCH_COMPILE:-1}"' |
| PASS | yes | launch exposes TorchDynamo override | examples/launch_qwen35_omni_speech_server_container.sh contains 'export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"' |
| PASS | yes | launch appends EXTRA_ARGS | examples/launch_qwen35_omni_speech_server_container.sh contains 'server_args+=("${extra_args_array[@]}")' |
| PASS | yes | performance recipe keeps code2wav compile on | main report / reproduction checklist contains 'NO_CODE2WAV_TORCH_COMPILE=0' |
| PASS | yes | performance recipe keeps TorchDynamo on | main report / reproduction checklist contains 'TORCHDYNAMO_DISABLE=0' |
| PASS | yes | performance recipe sets video preprocessing cache | main report / reproduction checklist contains 'SGLANG_OMNI_VIDEO_PREPROCESS_CACHE_MAX_BYTES=17179869184' |
| PASS | yes | performance recipe enables Thinker CUDA graph | main report / reproduction checklist contains '--thinker-cuda-graph on' |
| PASS | yes | performance recipe enables Talker CUDA graph | main report / reproduction checklist contains '--talker-cuda-graph on' |
| PASS | yes | performance recipe enables Talker torch compile | main report / reproduction checklist contains '--talker-torch-compile on' |
| PASS | yes | performance recipe uses max-running 8 | main report / reproduction checklist / launch script contains '--thinker-max-running-requests 8 --talker-max-running-requests 8' |
| PASS | yes | stress sweep covers c1/c2/c4/c8/c16 | concurrency=[1, 2, 4, 8, 16] |
| PASS | yes | stress quality is stable | acc=c1=0.700,c2=0.700,c4=0.700,c8=0.700,c16=0.700; WER=c1=0.0385,c2=0.0385,c4=0.0385,c8=0.0323,c16=0.0288 |
| PASS | yes | c8 is the current throughput peak | c4=2.0360, c8=2.5400, c16=2.4070 QPS |
| PASS | yes | c16 is a saturation boundary | c8_qps=2.5400, c16_qps=2.4070, c8_lat=3.064s, c16_lat=6.066s |
| PASS | yes | code2wav decode is not the stress bottleneck | decode_p95=c1=17.8ms,c2=20.2ms,c4=21.8ms,c8=25.9ms,c16=23.9ms |
| PASS | yes | talker to code2wav stream handoff is healthy | hop_p95=c1=15.5ms,c2=16.1ms,c4=17.8ms,c8=20.4ms,c16=19.7ms |
| PASS | yes | preprocessing compute stays stable while lifecycle queues | actual_avg_max=304.6ms, c8_queue_gap=937.4ms, c16_queue_gap=4090.5ms |
| PASS | yes | preproc=2 is locked as a negative optimization | baseline_qps=2.5400, preproc2_qps=1.6420, baseline_lat=3.064s, preproc2_lat=4.579s |
| PASS | yes | preproc=4 failure boundary is locked | acceptance matrix row PREPROCESSING_MAX_CONCURRENCY=4 at c=8 is PASS |
| PASS | yes | synthetic short and long speech regimes are covered | synthetic_rows=6 |
| PASS | yes | long synthetic c8 remains faster than real time | audio=52.3s, lat=25.799s, rtf=0.4932 |
| PASS | yes | synthetic speech handoff remains healthy | max_hop_p95=24.0ms, max_decode_p95=24.3ms |
| PASS | yes | stage interaction flags are locked | stage_interactions={'total_interactions': 37, 'status_counts': {'healthy': 28, 'queue_limited': 2, 'contention_regression': 1, 'prompt_feed_limited': 2, 'bottleneck': 1, 'diagnostic_only': 2, 'watch': 1}, 'sglang_talker_to_code2wav_healthy': True, 'sglang_code2wav_decode_not_bottleneck': True, 'vllm_original_c8_prompt_feed_limited': True, 'preprocessing_parallelism_regresses': True} |

## 3. 必须锁定的 SGLang 优化开关

| Switch / evidence | 用途 | 来源 |
| --- | --- | --- |
| `NO_CODE2WAV_TORCH_COMPILE=0; TORCHDYNAMO_DISABLE=0` | keep code2wav compile path enabled for performance runs | `benchmarks/reports/qwen35_omni_stress_performance_plan_20260621.md` |
| `--thinker-cuda-graph on; --talker-cuda-graph on; --talker-torch-compile on` | lock warmed Thinker/Talker graph and compile path | `benchmarks/reports/qwen35_omni_stress_performance_plan_20260621.md` |
| `--thinker-max-running-requests 8; --talker-max-running-requests 8` | expose the current c4-c8 operating window and c16 saturation boundary | `benchmarks/reports/qwen35_omni_stress_performance_plan_20260621.md` |
| `SGLANG_OMNI_VIDEO_PREPROCESS_CACHE_MAX_BYTES=17179869184; SGLANG_OMNI_VIDEO_PREPROCESS_CACHE_MAX_ENTRIES=64` | stabilize repeated Video-AMME preprocessing during stress sweeps | `benchmarks/reports/qwen35_omni_stress_performance_plan_20260621.md` |
| `PREPROCESSING_MAX_CONCURRENCY=1` | current safe admission point; wider preprocessing is a measured anti-recipe | `examples/launch_qwen35_omni_speech_server_container.sh` |

## 4. SGLang 压力窗口锁定表

| c | Accuracy | WER | Lat mean | Lat p95 | RTF mean | QPS | Audio throughput | 使用边界 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 70.0% | 3.85% | 1.316s | 2.406s | 1.0490 | 0.7600 | 1.4870 | recommended serving window |
| 2 | 70.0% | 3.85% | 1.508s | 3.124s | 1.0816 | 1.3150 | 2.7450 | recommended serving window |
| 4 | 70.0% | 3.85% | 1.929s | 3.633s | 1.4015 | 2.0360 | 4.0790 | recommended serving window |
| 8 | 70.0% | 3.23% | 3.064s | 5.853s | 2.2141 | 2.5400 | 5.3720 | current peak throughput |
| 16 | 70.0% | 2.88% | 6.066s | 7.846s | 4.8489 | 2.4070 | 4.7590 | saturation boundary |

## 5. Stage 连接锁定表

| c | Top stage | Preproc lifecycle avg/p95 | Talker avg/p95 | Code2wav decode avg/p95 | Talker->Code2wav hop p95 | 解释 |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | talker_ar 444/1516ms | 294.8ms/305.7ms | 444.1ms/1516.5ms | 14.3ms/17.8ms | 15.5ms | talker tail dominates |
| 2 | talker_ar 526/2024ms | 334.1ms/541.7ms | 525.8ms/2024.2ms | 16.0ms/20.2ms | 16.1ms | talker tail dominates |
| 4 | talker_ar 663/2628ms | 486.9ms/955.6ms | 663.1ms/2627.9ms | 16.0ms/21.8ms | 17.8ms | talker tail dominates |
| 8 | preprocessing 1227/2164ms | 1226.6ms/2163.7ms | 982.7ms/4418.0ms | 17.2ms/25.9ms | 20.4ms | preprocessing lifecycle queue joins the tail |
| 16 | preprocessing 4395/5884ms | 4395.1ms/5884.0ms | 815.6ms/3636.5ms | 16.7ms/23.9ms | 19.7ms | queue-dominated saturation |

## 6. 对外使用规则

- 可以说：SGLang 当前 best recipe 是 compiled/graph path + 16GiB preprocessing cache + serial preprocessing admission。
- 可以说：在当前 8x H20、Video-AMME ci-50、warmed pressure sweep 下，c=8 是吞吐峰值，c=16 是压力边界。
- 可以说：stage 间连接不是主要瓶颈；主要瓶颈来自 high-concurrency preprocessing lifecycle queue 和 talker AR tail。
- 禁止说：简单扩大 preprocessing 并发能优化当前 recipe；preproc=2/4 已经被反例锁定。
- 禁止把没有通过本页 JSON、claims、coverage、preflight、final readiness 的新 SGLang 数字替换进报告。

## 7. 机器证据

- environment_snapshot：`/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/environment_snapshot.json`
- tables_summary：`/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/tables_summary.json`
- claims_verification：`/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/claims_verification.json`
- acceptance_matrix：`/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/acceptance_matrix.json`
- stage_interaction_summary：`/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/stage_interaction_summary.json`
- launch_script：`/home/gangouyu/sglang-omni/examples/launch_qwen35_omni_speech_server_container.sh`
- main_report：`/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_stress_performance_plan_20260621.md`
- reproduction_checklist：`/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_reproduction_checklist_zh_20260621.md`
