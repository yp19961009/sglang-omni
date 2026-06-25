# Qwen3.5-Omni Runtime Image Contract

生成时间 UTC：`2026-06-21T02:00:21.587696+00:00`。
工作目录：`/home/gangouyu/sglang-omni`。

这页把 SGLang-Omni 与 vLLM-Omni 的镜像、digest、硬件环境、优化开关和
可对外声明边界合并成一个 handoff contract。它不新增 benchmark 口径，
只把已有审计证据压成 reviewer 可以快速核对的一页。

## 1. 结论

- ready：`True`，checks：`12/12`，required failures：`0`。
- GPU contract：`8x NVIDIA H20 / CUDA 12.8`。
- SGLang image：`frankleeeee/sglang-omni:dev`。
- SGLang image id：`sha256:be7e72126f525c3767008a73de16f400f974a09db431ded3c52bd48370941a84`。
- vLLM image：`tongyi-duanwu-registry-vpc.cn-beijing.cr.aliyuncs.com/dashscope/dashllm:cuda129_cp312_test_vl_13589`。
- vLLM image id：`sha256:e71dc281e0882896f81e3471206312b7c31ba4b6ab56ed9def0d4f1392a8c4ba`。
- SGLang scope：`c4-c8 warmed serving; c8 peak throughput; c16 saturation boundary`。
- vLLM strict scope：`optimized warmed c4 apples-to-apples headline only`。
- vLLM c8 scope：`prebuild w4 is optimized offline diagnostic, not online parity`。

## 2. Runtime Matrix

| Runtime | Image | Image ID | Created | Optimization Gate | Claim Scope |
| --- | --- | --- | --- | ---: | --- |
| SGLang-Omni | `frankleeeee/sglang-omni:dev` | `sha256:be7e72126f525c3767008a73de16f400f974a09db431ded3c52bd48370941a84` | `2026-02-05T09:30:27.304070315Z` | 26/26 | c4-c8 warmed serving; c8 peak throughput; c16 saturation boundary |
| vLLM-Omni | `tongyi-duanwu-registry-vpc.cn-beijing.cr.aliyuncs.com/dashscope/dashllm:cuda129_cp312_test_vl_13589` | `sha256:e71dc281e0882896f81e3471206312b7c31ba4b6ab56ed9def0d4f1392a8c4ba` | `2026-06-05T17:46:16.879316827+08:00` | 22/22 | optimized warmed c4 apples-to-apples headline only; prebuild w4 is optimized offline diagnostic, not online parity |

## 3. Gate 明细

| Status | Required | Gate | Evidence |
| --- | --- | --- | --- |
| PASS | yes | environment snapshot ready | environment_summary={'ready': True, 'claims': {'passed': True, 'total_checks': 17, 'failed_checks': 0}, 'coverage': {'total_requirements': 34, 'passed': 34, 'missing': 0, 'complete': True}, 'preflight': {'total_checks': 62, 'required_failures': 0, 'warnings': 1, 'ready': True}, 'manifest': {'total_records': 196, 'missing_records': 0, 'file_records': 194, 'directory_records': 2}}; preflight_pending_in_full_audit=False |
| PASS | yes | 8x H20 CUDA environment captured | count=8, cuda=12.8, first_gpu=NVIDIA H20 |
| PASS | yes | SGLang image digest locked | env=frankleeeee/sglang-omni:dev sha256:be7e72126f525c3767008a73de16f400f974a09db431ded3c52bd48370941a84; lock=frankleeeee/sglang-omni:dev sha256:be7e72126f525c3767008a73de16f400f974a09db431ded3c52bd48370941a84 |
| PASS | yes | vLLM image digest captured | env=tongyi-duanwu-registry-vpc.cn-beijing.cr.aliyuncs.com/dashscope/dashllm:cuda129_cp312_test_vl_13589 sha256:e71dc281e0882896f81e3471206312b7c31ba4b6ab56ed9def0d4f1392a8c4ba; lock=tongyi-duanwu-registry-vpc.cn-beijing.cr.aliyuncs.com/dashscope/dashllm:cuda129_cp312_test_vl_13589 sha256:e71dc281e0882896f81e3471206312b7c31ba4b6ab56ed9def0d4f1392a8c4ba |
| PASS | yes | SGLang optimization lock ready | sglang_optimization_lock={'ready': True, 'checks_total': 26, 'checks_passed': 26, 'required_failures': 0, 'sglang_image': 'frankleeeee/sglang-omni:dev', 'sglang_image_id': 'sha256:be7e72126f525c3767008a73de16f400f974a09db431ded3c52bd48370941a84', 'recommended_window': 'c4-c8 warmed serving; c8 peak throughput; c16 saturation boundary', 'recipe_contract': 'compiled/graph SGLang recipe with serial preprocessing and 16GiB preprocessing cache'} |
| PASS | yes | vLLM optimization lock ready | vllm_optimization_lock={'ready': True, 'checks_total': 22, 'checks_passed': 22, 'required_failures': 0, 'vllm_image': 'tongyi-duanwu-registry-vpc.cn-beijing.cr.aliyuncs.com/dashscope/dashllm:cuda129_cp312_test_vl_13589', 'vllm_image_id': 'sha256:e71dc281e0882896f81e3471206312b7c31ba4b6ab56ed9def0d4f1392a8c4ba', 'strict_c4_contract': 'optimized warmed c4 apples-to-apples headline only', 'c8_contract': 'prebuild w4 is optimized offline diagnostic, not online parity'} |
| PASS | yes | SGLang optimized recipe switches preserved | missing= |
| PASS | yes | vLLM optimized recipe switches preserved | missing= |
| PASS | yes | reproduction commands cover runtime recipes | commands=63, repro_ready=True, missing=[] |
| PASS | yes | SGLang serving scope explicit | recommended_window=c4-c8 warmed serving; c8 peak throughput; c16 saturation boundary |
| PASS | yes | vLLM strict c4 scope explicit | strict_c4_contract=optimized warmed c4 apples-to-apples headline only |
| PASS | yes | vLLM c8 online caveat explicit | vllm_online_parity_protocol={'ready': True, 'checks_total': 18, 'checks_passed': 18, 'required_failures': 0, 'current_package_safe': True, 'online_parity_proven': False, 'upgrade_decision': 'do_not_promote_c8_parity_without_online_ingress_artifacts', 'required_artifacts_total': 6} |

## 4. SGLang 必须保留的优化开关

| Switch | 用途 | 来源 |
| --- | --- | --- |
| `NO_CODE2WAV_TORCH_COMPILE=0; TORCHDYNAMO_DISABLE=0` | keep code2wav compile path enabled for performance runs | `benchmarks/reports/qwen35_omni_stress_performance_plan_20260621.md` |
| `--thinker-cuda-graph on; --talker-cuda-graph on; --talker-torch-compile on` | lock warmed Thinker/Talker graph and compile path | `benchmarks/reports/qwen35_omni_stress_performance_plan_20260621.md` |
| `--thinker-max-running-requests 8; --talker-max-running-requests 8` | expose the current c4-c8 operating window and c16 saturation boundary | `benchmarks/reports/qwen35_omni_stress_performance_plan_20260621.md` |
| `SGLANG_OMNI_VIDEO_PREPROCESS_CACHE_MAX_BYTES=17179869184; SGLANG_OMNI_VIDEO_PREPROCESS_CACHE_MAX_ENTRIES=64` | stabilize repeated Video-AMME preprocessing during stress sweeps | `benchmarks/reports/qwen35_omni_stress_performance_plan_20260621.md` |
| `PREPROCESSING_MAX_CONCURRENCY=1` | current safe admission point; wider preprocessing is a measured anti-recipe | `examples/launch_qwen35_omni_speech_server_container.sh` |

## 5. vLLM 必须保留的优化开关

| Switch / evidence | 用途 | 来源 |
| --- | --- | --- |
| `VLLM_ENABLE_TORCH_COMPILE=True` | avoid conservative eager baseline | `results/qwen35_vllm_videoamme_ci50_opt_20260618_162319/run_vllm_videoamme_ci5_offline_compile.sh` |
| `enforce_eager=False` | engine runs on compile/graph path | `vLLM run logs` |
| `FULL_AND_PIECEWISE CUDA graph` | lock optimized cudagraph mode | `vLLM run logs` |
| `enable_prefix_caching=True; enable_chunked_prefill=True` | lock vLLM-Omni prefill/cache behavior | `vLLM run logs` |
| `VLLM_HIDDEN_BUFFER_BACKEND=shm; VLLM_HIDDEN_BUFFER_FAST_TRANSFER=True` | lock inter-stage shared-memory transfer path | `results/qwen35_vllm_videoamme_ci50_opt_20260618_162319/run_vllm_videoamme_ci5_offline_compile.sh` |
| `VLLM_OMNI_ENABLE_ENCODER_TORCH_COMPILE=True; VLLM_OMNI_ENABLE_ENCODER_BATCH=True` | lock optimized multimodal encoder path | `results/qwen35_vllm_videoamme_ci50_opt_20260618_162319/run_vllm_videoamme_ci5_offline_compile.sh` |
| `--prebuild-prompts --prebuild-workers 4` | strongest current c=8 offline diagnostic | `results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuildw4_20260620_005346/run.log` |

## 6. 对外声明规则

- If either image digest changes, rerun the relevant optimization lock and full audit before replacing headline numbers.
- If any required optimization switch is removed, treat that run as a new baseline instead of the current optimized contract.
- SGLang c4-c8 is the current serving window; c16 is saturation evidence, not the recommended operating point.
- vLLM c8 prebuild w4 is an optimized offline diagnostic until online ingress, WER/ASR, and engine/talker boundary artifacts are collected.

## 7. 复现命令覆盖

本 contract 要求以下 reproduction command ids 存在于 `repro_command_manifest.json`：

- `launch_sglang_optimized`
- `sglang_synthetic_text_to_speech`
- `sglang_videoamme_stress`
- `vllm_c1_original`
- `vllm_c8_original`
- `vllm_c8_prebuild_w4`

## 8. 机器证据

- environment_snapshot：`/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/environment_snapshot.json`
- sglang_optimization_lock：`/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/sglang_optimization_lock.json`
- vllm_optimization_lock：`/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/vllm_optimization_lock.json`
- vllm_online_parity_protocol：`/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/vllm_online_parity_protocol.json`
- repro_command_manifest：`/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/repro_command_manifest.json`
