# Qwen3.5-Omni 复跑耗时/算力预算

生成时间 UTC：`2026-06-21T02:00:25.387202+00:00`。
工作目录：`/home/gangouyu/sglang-omni`。

这张表给合作方估算复跑排期和算力占用。计时列是从已归档结果反推的
`timed benchmark lower bound`：SGLang 用 `n / throughput_qps`，vLLM 用
`vllm_admission_diagnosis.wall_time_s`。它不包含 server launch/warmup/WER/ASR、
镜像或模型下载、缓存填充、人工审阅和结果打包时间，因此实际预约窗口应在下界上留 buffer。

## 1. 总览

| Item | Value |
| --- | --- |
| ready | `True` |
| budget rows | `9` |
| measured detail rows | `16` |
| timed rows | `6` |
| unmeasured/audit rows | `3` |
| command refs | `12` |
| required command ids present | `True` |
| total timed lower bound | `1592.8s` |
| 8-GPU equivalent lower bound | `3.540 GPUh` |
| SGLang Video-AMME lower bound | `168.8s` |
| SGLang synthetic lower bound | `161.5s` |
| vLLM lower bound | `1262.5s` |

## 2. 预算矩阵

| Budget | Command IDs | Runs / requests | Timed lower bound | 8-GPU lower bound | GPU scope | Evidence | Boundary |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| 收包 quickcheck 与 evidence-query smoke | `validate_share_bundle_receiver_smoke` | 1 / 0 | not timed | n/a | host-only package validation | `results/qwen35_report_audit_20260619/share_package_validation.json`; `results/qwen35_report_audit_20260619/share_package_receiver_smoke_validation.json`; `benchmarks/eval/qwen35_omni_evidence_query_cards_smoke.sh` | audit-only packaging check; no model-serving benchmark is rerun |
| 完整报告审计再生成 | `run_full_audit` | 1 / 0 | not timed | n/a | host audit pipeline | `results/qwen35_report_audit_20260619/audit_run_summary.json`; `results/qwen35_report_audit_20260619/final_readiness_audit.json`; `results/qwen35_report_audit_20260619/manifest.json` | audit-only, no benchmark rerun; it regenerates reports and package validation evidence |
| SGLang Video-AMME ci-50 c=1/2/4/8/16 | `launch_sglang_optimized`, `sglang_videoamme_stress` | 5 / 250 | 168.8s | 0.375 GPUh | 8x H20 SGLang serving session | `results/qwen35_report_audit_20260619/tables_summary.json`; `results/qwen35_report_audit_20260619/headline_scorecard.json`; `results/qwen35_report_audit_20260619/stage_latency_budget.json` | 不包含 server launch/warmup/WER/ASR；c=16 是 saturation boundary，不是默认 serving point |
| SGLang short/long synthetic speech c=1/4/8 | `launch_sglang_optimized`, `sglang_synthetic_text_to_speech` | 6 / 72 | 161.5s | 0.359 GPUh | 8x H20 SGLang serving session | `results/qwen35_report_audit_20260619/length_regime_coverage.json`; `results/qwen35_report_audit_20260619/share_charts/synthetic_short_long_speech.csv`; `results/qwen35_synthetic_speech_20260619/*/synthetic_speech_results.json` | 不包含 server launch/warmup/WER/ASR；short=74 chars/12 words, long=944 chars/139 words |
| SGLang saved-output WER/ASR recompute | `check_wer_asr_path`, `sglang_recompute_wer` | 1 / 250 | not timed | n/a | ASR GPU or cached Whisper path | `results/qwen35_sglang_mr8_stress_20260619/benchmark_audio_50_c*_*/whisper_large_v3_local_wer.json`; `benchmarks/reports/qwen35_omni_collaborator_rerun_validation_sheet_zh_20260621.md` | WER/ASR is quality validation, not serving throughput; avoid co-running with serving benchmark |
| vLLM-c1 offline benchmark | `vllm_c1_original` | 1 / 50 | 393.7s | 0.875 GPUh | 8x H20 vLLM offline runner | `results/qwen35_vllm_videoamme_ci50_offline_compile_c1_mns8_20260619_20260619_220617/benchmark_audio_50_c1_offline_compile/videoamme_results.json`; `results/qwen35_vllm_videoamme_ci50_offline_compile_c1_mns8_20260619_20260619_220617/run.log`; `results/qwen35_report_audit_20260619/vllm_admission_diagnosis.json` | offline diagnostic; not online serving parity |
| vLLM-c4 offline benchmark | `vllm_c4_original` | 1 / 50 | 325.5s | 0.723 GPUh | 8x H20 vLLM offline runner | `results/qwen35_vllm_videoamme_ci50_official_talker_compile_c4_20260618_202106/benchmark_audio_50_c4_offline_compile/videoamme_results.json`; `results/qwen35_vllm_videoamme_ci50_official_talker_compile_c4_20260618_202106/run.log`; `results/qwen35_report_audit_20260619/vllm_admission_diagnosis.json` | strict headline slice |
| vLLM-c8 offline benchmark | `vllm_c8_original` | 1 / 50 | 308.2s | 0.685 GPUh | 8x H20 vLLM offline runner | `results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_20260619_20260619_222434/benchmark_audio_50_c8_offline_compile/videoamme_results.json`; `results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_20260619_20260619_222434/run.log`; `results/qwen35_report_audit_20260619/vllm_admission_diagnosis.json` | offline diagnostic; not online serving parity |
| vLLM-c8-prebuild-w4 offline benchmark | `vllm_c8_prebuild_w4` | 1 / 50 | 235.1s | 0.522 GPUh | 8x H20 vLLM offline runner | `results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuildw4_20260620_005346/benchmark_audio_50_c8_offline_compile/videoamme_results.json`; `results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuildw4_20260620_005346/run.log`; `results/qwen35_report_audit_20260619/vllm_admission_diagnosis.json` | offline diagnostic; not online serving parity |

## 3. 计时明细

| Detail | Runtime | Scenario | C | Requests | Timed lower bound | QPS / wall QPS | Lat p95 / RTF p95 | Evidence |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| sglang_videoamme_c1 | sglang | Video-AMME ci-50 | 1 | 50 | 65.8s | 0.760 | 2.406s / 2.0198 | `results/qwen35_sglang_mr8_stress_20260619/benchmark_audio_50_c1_warm_profile_skipwer/videoamme_results.json`; `results/qwen35_sglang_mr8_stress_20260619/benchmark_audio_50_c1_warm_profile_skipwer/whisper_large_v3_local_wer.json` |
| sglang_videoamme_c2 | sglang | Video-AMME ci-50 | 2 | 50 | 38.0s | 1.315 | 3.124s / 1.9309 | `results/qwen35_sglang_mr8_stress_20260619/benchmark_audio_50_c2_warm_profile_skipwer/videoamme_results.json`; `results/qwen35_sglang_mr8_stress_20260619/benchmark_audio_50_c2_warm_profile_skipwer/whisper_large_v3_local_wer.json` |
| sglang_videoamme_c4 | sglang | Video-AMME ci-50 | 4 | 50 | 24.6s | 2.036 | 3.633s / 2.4983 | `results/qwen35_sglang_mr8_stress_20260619/benchmark_audio_50_c4_profile_skipwer/videoamme_results.json`; `results/qwen35_sglang_mr8_stress_20260619/benchmark_audio_50_c4_profile_skipwer/whisper_large_v3_local_wer.json` |
| sglang_videoamme_c8 | sglang | Video-AMME ci-50 | 8 | 50 | 19.7s | 2.540 | 5.853s / 4.3925 | `results/qwen35_sglang_mr8_stress_20260619/benchmark_audio_50_c8_profile_skipwer/videoamme_results.json`; `results/qwen35_sglang_mr8_stress_20260619/benchmark_audio_50_c8_profile_skipwer/whisper_large_v3_local_wer.json` |
| sglang_videoamme_c16 | sglang | Video-AMME ci-50 | 16 | 50 | 20.8s | 2.407 | 7.846s / 10.4087 | `results/qwen35_sglang_mr8_stress_20260619/benchmark_audio_50_c16_profile_skipwer/videoamme_results.json`; `results/qwen35_sglang_mr8_stress_20260619/benchmark_audio_50_c16_profile_skipwer/whisper_large_v3_local_wer.json` |
| sglang_synthetic_short_c1 | sglang | synthetic short | 1 | 16 | 13.9s | 1.154 | 0.924s / 0.2112 | `results/qwen35_synthetic_speech_20260619/short_c1/synthetic_speech_results.json` |
| sglang_synthetic_short_c4 | sglang | synthetic short | 4 | 16 | 7.2s | 2.218 | 2.056s / 0.4388 | `results/qwen35_synthetic_speech_20260619/short_c4/synthetic_speech_results.json` |
| sglang_synthetic_short_c8 | sglang | synthetic short | 8 | 16 | 5.4s | 2.983 | 2.828s / 0.7440 | `results/qwen35_synthetic_speech_20260619/short_c8/synthetic_speech_results.json` |
| sglang_synthetic_long_c1 | sglang | synthetic long | 1 | 8 | 73.4s | 0.109 | 9.465s / 0.1776 | `results/qwen35_synthetic_speech_20260619/long_c1/synthetic_speech_results.json` |
| sglang_synthetic_long_c4 | sglang | synthetic long | 4 | 8 | 35.2s | 0.227 | 18.025s / 0.3373 | `results/qwen35_synthetic_speech_20260619/long_c4/synthetic_speech_results.json` |
| sglang_synthetic_long_c8 | sglang | synthetic long | 8 | 8 | 26.4s | 0.303 | 26.318s / 0.5001 | `results/qwen35_synthetic_speech_20260619/long_c8/synthetic_speech_results.json` |
| vllm-c1 | vllm | vLLM-c1 | 1 | 50 | 393.7s | 0.127 | runner overhead 0.0% | `results/qwen35_vllm_videoamme_ci50_offline_compile_c1_mns8_20260619_20260619_220617/benchmark_audio_50_c1_offline_compile/videoamme_results.json`; `results/qwen35_vllm_videoamme_ci50_offline_compile_c1_mns8_20260619_20260619_220617/run.log` |
| vllm-c4 | vllm | vLLM-c4 | 4 | 50 | 325.5s | 0.154 | runner overhead 76.7% | `results/qwen35_vllm_videoamme_ci50_official_talker_compile_c4_20260618_202106/benchmark_audio_50_c4_offline_compile/videoamme_results.json`; `results/qwen35_vllm_videoamme_ci50_official_talker_compile_c4_20260618_202106/run.log` |
| vllm-c8 | vllm | vLLM-c8 | 8 | 50 | 308.2s | 0.162 | runner overhead 81.8% | `results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_20260619_20260619_222434/benchmark_audio_50_c8_offline_compile/videoamme_results.json`; `results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_20260619_20260619_222434/run.log` |
| vllm-c8-prebuild-w1 | vllm | vLLM-c8-prebuild-w1 | 8 | 50 | 352.1s | 0.142 | runner overhead 77.6% | `results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuild_20260620_002020/benchmark_audio_50_c8_offline_compile/videoamme_results.json`; `results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuild_20260620_002020/run.log` |
| vllm-c8-prebuild-w4 | vllm | vLLM-c8-prebuild-w4 | 8 | 50 | 235.1s | 0.213 | runner overhead 65.6% | `results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuildw4_20260620_005346/benchmark_audio_50_c8_offline_compile/videoamme_results.json`; `results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuildw4_20260620_005346/run.log` |

## 4. 查询命令

```bash
jq '.summary' results/qwen35_report_audit_20260619/rerun_time_budget.json
jq '.rows[] | {budget_id, command_ids, timed_wall_lower_bound_s, boundary}' results/qwen35_report_audit_20260619/rerun_time_budget.json
jq '.detail_rows[] | select(.runtime=="sglang") | {detail_id, concurrency, timed_wall_lower_bound_s, throughput_qps}' results/qwen35_report_audit_20260619/rerun_time_budget.json
```

## 5. 使用边界

- `rerun_time_budget.json` 是排期预算和复跑解释证据，不替代 `rerun_acceptance_contract.json` 的 34 项返回证据硬合同。
- SGLang 预算默认共用同一个 8 卡 serving session；若每个压力点都冷启动，实际 wall time 会高于表中下界。
- vLLM c=8 与 c=8 prebuild w4 是 offline diagnostic；升级成 online parity headline 前仍要按 `qwen35_omni_vllm_online_parity_protocol_zh_20260621.md` 补在线入口证据。
- WER/ASR 复算没有纳入 timed benchmark lower bound；合作方如果要求重新跑 WER，请单独预约 ASR GPU 或确认本地 Whisper cache。
