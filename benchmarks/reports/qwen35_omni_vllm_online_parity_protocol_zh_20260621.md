# Qwen3.5-Omni vLLM c=8 Online Parity 升级协议

生成时间 UTC：`2026-06-21T02:00:21.542929+00:00`。
工作目录：`/home/gangouyu/sglang-omni`。

这份协议只定义如何把当前 vLLM c=8 证据从 offline diagnostic 升级为 strict
online serving parity。它不把当前报告的 vLLM c=8 prebuild w4 结果提升为
online parity；当前可分享 headline 仍是 warmed c=4 的严格横向对比。

## 1. 当前裁决

| Gate | Value |
| --- | ---: |
| Protocol ready | `True` |
| Checks | `18/18` |
| Required failures | `0` |
| Current package safe | `True` |
| Online parity proven | `False` |
| Upgrade decision | `do_not_promote_c8_parity_without_online_ingress_artifacts` |

当前结论：vLLM original c=8 仍是 prompt-feed/admission limited；prebuild w4
是优化后的 offline diagnostic，用来定位 runner prompt 构造/投喂瓶颈和后续
engine/talker-side tail，不可直接写成 online serving parity。

## 2. SGLang c=8 对照目标

| Metric | Value |
| --- | ---: |
| n | `50` |
| accuracy | `70.0%` |
| WER corpus | `3.2%` |
| latency mean / p95 | `3.064s` / `5.853s` |
| RTF mean / p95 | `2.2141` / `4.3925` |
| throughput QPS | `2.540` |

## 3. 升级为 strict c=8 parity 的替换 Gate

| Gate | Required value |
| --- | --- |
| sample contract | Video-AMME ci-50, max_tokens=256, temperature=0, c=8, skip first warmup batch/request window. |
| completed / failed | `>= 50` / `<= 0` |
| latency mean max | `3.217s` |
| latency p95 max | `6.146s` |
| RTF mean max | `2.3248` |
| RTF p95 max | `4.6121` |
| throughput QPS min | `2.413` |
| accuracy min | `68.0%` |
| WER corpus max | `4.2%` |
| online ingress | HTTP/OpenAI-compatible serving ingress, not offline engine runner |
| quality path | same WER/ASR scoring path after serving latency measurement |
| stage profile | request admission, preprocessing, thinker/talker, talker/code2wav, code2wav decode all visible |

## 4. 必需新增 Artifact

| Artifact | Requirement |
| --- | --- |
| vLLM online serving launch log | same image digest, compile/cache knobs, c=8 admission capacity, and warmup are logged. |
| vLLM online HTTP ingress result JSON | Video-AMME ci-50 requests are sent through OpenAI-compatible ingress, not offline engine calls. |
| vLLM online stage/tail profile | request admission, preprocessing, thinker/talker, talker/code2wav, and code2wav decode boundaries are measurable. |
| vLLM online WER/ASR JSON | same audio outputs are scored with the same Whisper/ASR path after serving latency is measured. |
| updated SGLang c=8 control run or frozen control hash | SGLang comparison target is either rerun in the same window or explicitly pinned by manifest hash. |
| parity verifier JSON | machine gate compares vLLM online c=8 against SGLang c=8 thresholds before any headline replacement. |

## 5. 当前证据边界

- strict headline：SGLang warmed c=4 vs optimized vLLM warmed c=4.
- vLLM c=8 status：optimized offline diagnostic only
- 禁止把 `prebuild w4`、`engine_qps` 或 offline runner wall QPS 写成 online serving parity。
- 如果合作方给出新的 vLLM online c=8 结果，先按本页 gate 生成 parity verifier JSON，再决定是否替换主报告数字。

## 6. 协议自检

| Status | Required | Check | Evidence |
| --- | --- | --- | --- |
| PASS | yes | vLLM image is recorded | image=tongyi-duanwu-registry-vpc.cn-beijing.cr.aliyuncs.com/dashscope/dashllm:cuda129_cp312_test_vl_13589, id=sha256:e71dc281e0882896f81e3471206312b7c31ba4b6ab56ed9def0d4f1392a8c4ba |
| PASS | yes | vLLM optimization lock ready | vllm_optimization_lock={'ready': True, 'checks_total': 22, 'checks_passed': 22, 'required_failures': 0, 'vllm_image': 'tongyi-duanwu-registry-vpc.cn-beijing.cr.aliyuncs.com/dashscope/dashllm:cuda129_cp312_test_vl_13589', 'vllm_image_id': 'sha256:e71dc281e0882896f81e3471206312b7c31ba4b6ab56ed9def0d4f1392a8c4ba', 'strict_c4_contract': 'optimized warmed c4 apples-to-apples headline only', 'c8_contract': 'prebuild w4 is optimized offline diagnostic, not online parity'} |
| PASS | yes | original c8 remains prompt-feed/admission limited | vLLM-c8 diagnosis=prompt_feed_limited, span_avg_ms=33314.0 |
| PASS | yes | prebuild w4 is diagnostic, not parity | vLLM-c8-prebuild-w4 diagnosis=engine_or_workload_limited, engine_qps=0.5360 |
| PASS | yes | prebuild w4 improves prompt build wall | w1_prompt_wall=249.343s, w4_prompt_wall=129.220s |
| PASS | yes | prebuild w4 improves runner wall | w1_runner_wall=352.126s, w4_runner_wall=235.085s |
| PASS | yes | stage summary records current c8 boundary | stage_interactions={'total_interactions': 37, 'status_counts': {'healthy': 28, 'queue_limited': 2, 'contention_regression': 1, 'prompt_feed_limited': 2, 'bottleneck': 1, 'diagnostic_only': 2, 'watch': 1}, 'sglang_talker_to_code2wav_healthy': True, 'sglang_code2wav_decode_not_bottleneck': True, 'vllm_original_c8_prompt_feed_limited': True, 'preprocessing_parallelism_regresses': True} |
| PASS | yes | SGLang c8 target row is present | sglang_c8={'concurrency': 8, 'n': 50, 'accuracy': 0.7, 'latency_mean_s': 3.064, 'latency_p95_s': 5.853, 'rtf_mean': 2.2141, 'rtf_p95': 4.3925, 'throughput_qps': 2.54, 'audio_throughput_s_per_s': 5.372, 'wer_corpus': 0.03225806451612903, 'result_json': '/home/gangouyu/sglang-omni/results/qwen35_sglang_mr8_stress_20260619/benchmark_audio_50_c8_profile_skipwer/videoamme_results.json', 'wer_json': '/home/gangouyu/sglang-omni/results/qwen35_sglang_mr8_stress_20260619/benchmark_audio_50_c8_profile_skipwer/whisper_large_v3_local_wer.json'} |
| PASS | yes | wrapper keeps vLLM compile enabled | results/qwen35_vllm_videoamme_ci50_opt_20260618_162319/run_vllm_videoamme_ci5_offline_compile.sh contains 'VLLM_ENABLE_TORCH_COMPILE=True' |
| PASS | yes | wrapper keeps optimized cache/transfer path | results/qwen35_vllm_videoamme_ci50_opt_20260618_162319/run_vllm_videoamme_ci5_offline_compile.sh contains 'VLLM_HIDDEN_BUFFER_FAST_TRANSFER=True' |
| PASS | yes | runner supports prebuilt prompts | results/qwen35_vllm_videoamme_ci50_opt_20260618_162319/vllm_videoamme_runner.py contains '--prebuild-prompts' |
| PASS | yes | final delivery note forbids online parity claim | benchmarks/reports/qwen35_omni_final_share_delivery_note_zh_20260621.md contains 'vLLM c=8 prebuild w4 不是 online serving parity' |
| PASS | yes | runtime contract separates online parity | benchmarks/reports/qwen35_omni_runtime_comparison_contract_zh_20260621.md contains '不能当 online parity' |
| PASS | yes | vLLM lock states diagnostic boundary | benchmarks/reports/qwen35_omni_vllm_optimization_lock_zh_20260621.md contains '不是 online serving parity' |
| PASS | yes | caveat matrix blocks c8 headline promotion | benchmarks/reports/qwen35_omni_caveat_adjudication_matrix_zh_20260621.md contains 'online parity' |
| PASS | yes | main report keeps online caveat visible | benchmarks/reports/qwen35_omni_stress_performance_plan_20260621.md contains 'online serving parity' |
| PASS | yes | replacement gates are declared | replacement_gates={'sample_contract': 'Video-AMME ci-50, max_tokens=256, temperature=0, c=8, skip first warmup batch/request window.', 'online_ingress_required': True, 'same_quality_path_required': True, 'minimum_completed': 50, 'maximum_failed': 0, 'latency_mean_s_max_for_parity': 3.2172, 'latency_p95_s_max_for_parity': 6.1456, 'rtf_mean_max_for_parity': 2.3248, 'rtf_p95_max_for_parity': 4.6121, 'throughput_qps_min_for_parity': 2.413, 'accuracy_min_for_parity': 0.68, 'wer_corpus_max_for_parity': 0.0423, 'stage_boundary_required': ['online request admission/queue time', 'input preprocessing lifecycle', 'encoder/thinker boundary', 'thinker->talker handoff', 'talker codec cadence', 'talker->code2wav drain', 'code2wav collect/decode']} |
| PASS | yes | current package does not overclaim online parity | current_package_safe=True, online_parity_proven=False |
