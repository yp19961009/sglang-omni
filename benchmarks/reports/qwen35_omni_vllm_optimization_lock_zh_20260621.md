# Qwen3.5-Omni vLLM 优化锁定矩阵

生成时间 UTC：`2026-06-21T02:00:21.497079+00:00`。
工作目录：`/home/gangouyu/sglang-omni`。

这页专门锁定 vLLM baseline 的镜像、run wrapper、compile/cudagraph/cache
优化开关和 c=8 prebuild diagnostic 边界。目的不是新增 benchmark 口径，
而是防止 reviewer 误以为报告拿 SGLang 优化版去比较一个保守 vLLM baseline。

## 1. 锁定结论

- ready：`True`，checks：`22/22`，required failures：`0`。
- vLLM image：`tongyi-duanwu-registry-vpc.cn-beijing.cr.aliyuncs.com/dashscope/dashllm:cuda129_cp312_test_vl_13589`。
- vLLM image id：`sha256:e71dc281e0882896f81e3471206312b7c31ba4b6ab56ed9def0d4f1392a8c4ba`。
- strict headline 只使用 warmed c=4 apples-to-apples 对比。
- c=8 prebuild w4 是当前最强 vLLM offline diagnostic，不是 online serving parity。

## 2. Gate 明细

| Status | Required | Gate | Evidence |
| --- | --- | --- | --- |
| PASS | yes | vLLM image locked | image=tongyi-duanwu-registry-vpc.cn-beijing.cr.aliyuncs.com/dashscope/dashllm:cuda129_cp312_test_vl_13589, id=sha256:e71dc281e0882896f81e3471206312b7c31ba4b6ab56ed9def0d4f1392a8c4ba, created=2026-06-05T17:46:16.879316827+08:00 |
| PASS | yes | vLLM wrapper present | /home/gangouyu/sglang-omni/results/qwen35_vllm_videoamme_ci50_opt_20260618_162319/run_vllm_videoamme_ci5_offline_compile.sh |
| PASS | yes | vLLM runner present | /home/gangouyu/sglang-omni/results/qwen35_vllm_videoamme_ci50_opt_20260618_162319/vllm_videoamme_runner.py |
| PASS | yes | wrapper uses locked image | results/qwen35_vllm_videoamme_ci50_opt_20260618_162319/run_vllm_videoamme_ci5_offline_compile.sh contains 'tongyi-duanwu-registry-vpc.cn-beijing.cr.aliyuncs.com/dashscope/dashllm:cuda129_cp312_test_vl_13589' |
| PASS | yes | wrapper enables torch compile | results/qwen35_vllm_videoamme_ci50_opt_20260618_162319/run_vllm_videoamme_ci5_offline_compile.sh contains 'VLLM_ENABLE_TORCH_COMPILE=True' |
| PASS | yes | wrapper enables hidden-buffer fast transfer | results/qwen35_vllm_videoamme_ci50_opt_20260618_162319/run_vllm_videoamme_ci5_offline_compile.sh contains 'VLLM_HIDDEN_BUFFER_FAST_TRANSFER=True' |
| PASS | yes | wrapper enables shared-memory hidden buffer | results/qwen35_vllm_videoamme_ci50_opt_20260618_162319/run_vllm_videoamme_ci5_offline_compile.sh contains 'VLLM_HIDDEN_BUFFER_BACKEND=shm' |
| PASS | yes | wrapper enables encoder torch compile | results/qwen35_vllm_videoamme_ci50_opt_20260618_162319/run_vllm_videoamme_ci5_offline_compile.sh contains 'VLLM_OMNI_ENABLE_ENCODER_TORCH_COMPILE=True' |
| PASS | yes | wrapper enables encoder batching | results/qwen35_vllm_videoamme_ci50_opt_20260618_162319/run_vllm_videoamme_ci5_offline_compile.sh contains 'VLLM_OMNI_ENABLE_ENCODER_BATCH=True' |
| PASS | yes | wrapper reuses thinker preprocessing for talker | results/qwen35_vllm_videoamme_ci50_opt_20260618_162319/run_vllm_videoamme_ci5_offline_compile.sh contains 'VLLM_OMNI_TALKER_REUSE_THINKER_PREPROCESS=True' |
| PASS | yes | runner supports prebuild prompts | results/qwen35_vllm_videoamme_ci50_opt_20260618_162319/vllm_videoamme_runner.py contains '--prebuild-prompts' |
| PASS | yes | runner supports prebuild workers | results/qwen35_vllm_videoamme_ci50_opt_20260618_162319/vllm_videoamme_runner.py contains '--prebuild-workers' |
| PASS | yes | log proves torch.compile preflight | vLLM run logs contains 'preflight torch.compile ok' |
| PASS | yes | log proves enforce_eager false | vLLM run logs contains "'enforce_eager': False" |
| PASS | yes | log proves VLLM compile mode | vLLM run logs contains 'CompilationMode.VLLM_COMPILE' |
| PASS | yes | log proves FULL_AND_PIECEWISE CUDA graph | vLLM run logs contains 'FULL_AND_PIECEWISE' |
| PASS | yes | log proves chunked prefill | vLLM run logs contains "'enable_chunked_prefill': True" |
| PASS | yes | log proves prefix caching | vLLM run logs contains "'enable_prefix_caching': True" |
| PASS | yes | w4 diagnostic uses prebuild args | results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuildw4_20260620_005346/run.log contains 'EXTRA_ARGS=--prebuild-prompts --prebuild-workers 4' |
| PASS | yes | w4 diagnostic logs prebuilt batches | results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuildw4_20260620_005346/run.log contains 'Prebuilt batch 0 prompts' |
| PASS | yes | original c8 prompt-feed diagnosis locked | label=vLLM-c8, diagnosis=prompt_feed_limited, admission_p95=43972.7ms |
| PASS | yes | prebuild w4 improves runner wall | c8_runner_qps=0.1622, w4_runner_qps=0.2127, w4_diagnosis=engine_or_workload_limited |

## 3. 必须锁定的 vLLM 优化开关

| Switch / evidence | 用途 | 来源 |
| --- | --- | --- |
| `VLLM_ENABLE_TORCH_COMPILE=True` | avoid conservative eager baseline | `results/qwen35_vllm_videoamme_ci50_opt_20260618_162319/run_vllm_videoamme_ci5_offline_compile.sh` |
| `enforce_eager=False` | engine runs on compile/graph path | `vLLM run logs` |
| `FULL_AND_PIECEWISE CUDA graph` | lock optimized cudagraph mode | `vLLM run logs` |
| `enable_prefix_caching=True; enable_chunked_prefill=True` | lock vLLM-Omni prefill/cache behavior | `vLLM run logs` |
| `VLLM_HIDDEN_BUFFER_BACKEND=shm; VLLM_HIDDEN_BUFFER_FAST_TRANSFER=True` | lock inter-stage shared-memory transfer path | `results/qwen35_vllm_videoamme_ci50_opt_20260618_162319/run_vllm_videoamme_ci5_offline_compile.sh` |
| `VLLM_OMNI_ENABLE_ENCODER_TORCH_COMPILE=True; VLLM_OMNI_ENABLE_ENCODER_BATCH=True` | lock optimized multimodal encoder path | `results/qwen35_vllm_videoamme_ci50_opt_20260618_162319/run_vllm_videoamme_ci5_offline_compile.sh` |
| `--prebuild-prompts --prebuild-workers 4` | strongest current c=8 offline diagnostic | `results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuildw4_20260620_005346/run.log` |

## 4. vLLM case 锁定表

| Case | c | Runner QPS | Engine QPS | Admission p95 | Runner overhead | Diagnosis | 使用边界 |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| vLLM-c4 | 4 | 0.1536 | 0.1536 | 19135.8ms | 76.7% | prompt_feed_limited | strict warmed baseline |
| vLLM-c8 | 8 | 0.1622 | 0.1622 | 43972.7ms | 81.8% | prompt_feed_limited | prompt-feed diagnostic |
| vLLM-c8-prebuild-w4 | 8 | 0.2127 | 0.5360 | 4891.5ms | 65.6% | engine_or_workload_limited | optimized offline diagnostic |

## 5. 对外使用规则

- 可以说：vLLM baseline 使用 Qwen3.5-capable 镜像，且 compile、CUDA graph、prefix/chunked prefill、shared-memory transfer 和 encoder compile/batch 等优化路径均有 run script 或 log 证据。
- 可以说：c=8 prebuild w4 已经把 offline runner 的 prompt-feed admission 问题明显缓解，是当前最强 vLLM offline diagnostic。
- 禁止说：已经完成严格 vLLM c=8 online serving parity；该结论需要 online ingress、同口径 WER/ASR 和 engine/talker boundary 复核。
- 禁止把缺少这些开关的新 vLLM 重跑结果直接替换为报告 baseline；必须先刷新本页 JSON、runtime contract、confidence ledger、final readiness 和 full audit。

## 6. 机器证据

- environment_snapshot：`/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/environment_snapshot.json`
- vllm_admission_diagnosis：`/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/vllm_admission_diagnosis.json`
- run_script：`/home/gangouyu/sglang-omni/results/qwen35_vllm_videoamme_ci50_opt_20260618_162319/run_vllm_videoamme_ci5_offline_compile.sh`
- runner：`/home/gangouyu/sglang-omni/results/qwen35_vllm_videoamme_ci50_opt_20260618_162319/vllm_videoamme_runner.py`
- c4_log：`/home/gangouyu/sglang-omni/results/qwen35_vllm_videoamme_ci50_official_talker_compile_c4_20260618_202106/run.log`
- c8_log：`/home/gangouyu/sglang-omni/results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_20260619_20260619_222434/run.log`
- c8_prebuild_w4_log：`/home/gangouyu/sglang-omni/results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuildw4_20260620_005346/run.log`
