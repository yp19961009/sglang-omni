# Qwen3.5-Omni 数字来源索引

状态：证据门已就绪；更新后的目标不再等待 6.21 晚间，后续变更必须重跑 full audit。
工作目录：`/home/gangouyu/sglang-omni`。
用途：把分享时最容易被追问的 headline 数字、SGLang pressure 数字、stage 连接数字、
vLLM c=8 诊断数字和反例数字映射到机器证据，避免现场只凭记忆引用。

关联材料：

- 主报告：
  `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_stress_performance_plan_20260621.md`
- 压力条件总表：
  `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_pressure_matrix_zh_20260621.md`
- Stage 指标字典：
  `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_stage_metric_dictionary_zh_20260621.md`
- 外部复现 handoff runbook：
  `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_external_handoff_runbook_zh_20260621.md`
- 合作方复跑验收表：
  `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_collaborator_rerun_validation_sheet_zh_20260621.md`
- headline scorecard：
  `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/headline_scorecard.json`
- metric provenance index：
  `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/metric_provenance_index.json`
- claim metric crosswalk：
  `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/claim_metric_crosswalk.json`
- objective requirement crosswalk：
  `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/objective_requirement_crosswalk.json`
- acceptance matrix：
  `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/acceptance_matrix.json`
- stage interaction summary：
  `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/stage_interaction_summary.json`
- vLLM admission diagnosis：
  `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/vllm_admission_diagnosis.json`

## 1. 读法

优先引用顺序：

1. 对外口径先看 `confidence_ledger.json` 和中文简报，确认这句话能不能说。
2. 数字来源看本文件；如果本文件和刚重跑的 JSON 不一致，以重跑后的 JSON 为准。
3. 需要原始 artifact 时看 `manifest.json` 的 SHA-256 和下表中的 raw path。
4. 重跑后必须重新跑 full audit，再更新本文件和主报告中的对应数字。

## 2. Headline 数字

| 要引用的结论 | 推荐数字 | 主机器来源 | 原始 artifact | 现场说法 |
| --- | --- | --- | --- | --- |
| warmed c=4 SGLang 至少和 vLLM 相当，并且 latency/RTF 更优 | SGLang latency 1.743/3.328s，RTF 1.3536/2.4023；vLLM latency 2.093/3.525s，RTF 1.4677/3.0717 | `headline_scorecard.json` -> `strict_c4_comparison`; `claims_verification.json` -> `SGLang warmed c4 beats vLLM warmed c4 latency/RTF` | SGLang `results/qwen35_sglang_subtalker_seedfix_compile_mr4_ci50_c4_20260618_181046/benchmark_audio_50_c4_warm_profile_no_wer/videoamme_results.json`; vLLM `results/qwen35_vllm_videoamme_ci50_official_talker_compile_c4_20260618_202106/benchmark_audio_50_c4_offline_compile/videoamme_results.json` | “warmed c=4 严格对比里，SGLang latency 和 RTF 均优于优化版 vLLM。” |
| warmed c=4 质量没有退化 | SGLang accuracy 67.4%，WER 4.12%；vLLM accuracy 63.0%，WER 7.44% | `headline_scorecard.json` -> `strict_c4_comparison`; `claims_verification.json` -> `SGLang warmed c4 preserves accuracy/WER vs vLLM` | SGLang/vLLM `videoamme_results.json` + `whisper_large_v3_wer.json` | “速度提升不是用质量换来的。” |
| SGLang 当前推荐窗口 | c=4-c=8；c=8 是吞吐峰值 | `headline_scorecard.json` -> `sglang_stress.throughput_peak`; `acceptance_matrix.json` -> `recommended_peak_throughput` | `results/qwen35_sglang_mr8_stress_20260619/benchmark_audio_50_c*_*/videoamme_results.json` | “c=8 是当前 recipe 的最高吞吐点，c=16 是压力边界。” |
| 短/长文本输入覆盖 | short=74 chars / 12 words；long=944 chars / 139 words | `tables_summary.json` -> `synthetic_speech.target_chars/target_words`; `share_charts/synthetic_short_long_speech.csv` | `results/qwen35_synthetic_speech_20260619/short_c*/synthetic_speech_results.json`; `results/qwen35_synthetic_speech_20260619/long_c*/synthetic_speech_results.json` | “长短文口径来自实际 synthetic text-to-speech artifact，不是手写说明。” |
| long c=8 仍快于实时 | audio mean 52.3s，latency mean 25.799s，RTF 0.4932 | `headline_scorecard.json` -> `synthetic_long_c8`; `tables_summary.json` -> `synthetic_speech` | `results/qwen35_synthetic_speech_20260619/long_c8/synthetic_speech_results.json` | “长文本/长语音 guardrail 仍快于实时。” |

## 3. SGLang Pressure 数字

| Pressure | 推荐数字 | 主机器来源 | 复核字段 | 解释边界 |
| --- | ---: | --- | --- | --- |
| c=1 低并发基线 | QPS 0.760，latency 1.316/2.406s，WER 3.85% | `tables_summary.json` | `tables.sglang_stress` row `concurrency=1` | 用来隔离排队影响，不是吞吐 headline。 |
| c=2 低并发基线 | QPS 1.315，latency 1.508/3.124s，WER 3.85% | `tables_summary.json` | `tables.sglang_stress` row `concurrency=2` | 仍主要是 talker tail。 |
| c=4 推荐窗口 | QPS 2.036，latency 1.929/3.633s，WER 3.85% | `tables_summary.json`; `headline_scorecard.json` | `tables.sglang_stress` row `concurrency=4` | 与 vLLM 的严格横向对比也在 c=4。 |
| c=8 吞吐峰值 | QPS 2.540，latency 3.064/5.853s，WER 3.23% | `headline_scorecard.json`; `claims_verification.json` | `sglang_stress.throughput_peak`; claim `SGLang stress c8 is throughput peak` | 当前 recipe 的最高吞吐点。 |
| c=16 压力边界 | QPS 2.407，latency 6.066/7.846s，WER 2.88% | `headline_scorecard.json`; `acceptance_matrix.json` | `sglang_stress.c16_vs_c8_qps_delta_pct`; row `not_recommended_saturation` | 可运行但不推荐默认服务点。 |

## 4. Stage 连接数字

| 问题 | 推荐数字 | 主机器来源 | 现场解释 |
| --- | --- | --- | --- |
| c=8/c=16 的 preprocessing 是不是算子变慢 | c=8 lifecycle 1227/2164ms，但 actual preprocess 289/336ms；c=16 lifecycle 4395/5884ms，但 actual preprocess 305/341ms | `tables_summary.json` -> `sglang_preprocessing_split`; `stage_interaction_summary.json` -> `request_admission_to_preprocessing` | 主要是 admission/queue，不是预处理 compute 突然变慢。 |
| `talker_ar -> code2wav` 是否卡住 | SGLang hop p95：c=1 15.5ms，c=2 16.1ms，c=4 17.8ms，c=8 20.4ms，c=16 19.7ms | `stage_interaction_summary.json` -> `talker_to_code2wav_stream`; `claims_verification.json` -> `code2wav decode and talker->code2wav hop are not bottlenecks` | stage handoff 健康，不是 c=8/c=16 主瓶颈。 |
| code2wav decode 是否是瓶颈 | decode p95：c=1 17.8ms，c=2 20.2ms，c=4 21.8ms，c=8 25.9ms，c=16 23.9ms | `claims_verification.json`; `stage_interaction_summary.json` -> `code2wav_collect_to_decode` | decode 是十几到二十几毫秒量级，优先级低于 talker AR 和 admission。 |
| long speech 的 tail 在哪 | long c=8 talker avg/p95 25572/26199ms，code2wav decode avg 14.5ms，hop p95 24.0ms | `stage_interaction_summary.json` -> `synthetic_long c=8`; `tables_summary.json` -> `synthetic_stage_breakdown` | 长语音主要是 talker AR 和 chunk cadence，不是 vocoder compute。 |

## 5. vLLM c=8 诊断数字

| vLLM pressure | 推荐数字 | 主机器来源 | 解释边界 |
| --- | ---: | --- | --- |
| c=4 optimized baseline | runner/engine QPS 0.1536，admission span 15.11/19.14s | `vllm_admission_diagnosis.json` row `vLLM-c4` | 用于 warmed c=4 严格横向 baseline；offline runner prompt-feed 很重。 |
| c=8 original | runner/engine QPS 0.1622，admission span 33.31/43.97s，runner overhead 81.8% | `vllm_admission_diagnosis.json` row `vLLM-c8`; `claims_verification.json` -> `vLLM c8 offline runner is prompt-feed/admission limited` | 诊断为 prompt-feed/admission limited，不能直接当 online parity。 |
| c=8 prebuild w1 | runner QPS 0.1420，engine QPS 0.5391，admission span 4.44/5.43s | `vllm_admission_diagnosis.json` row `vLLM-c8-prebuild-w1` | 证明 prebuild 能把 admission span 降下来，但 runner wall 仍慢。 |
| c=8 prebuild w4 | runner QPS 0.2127，engine QPS 0.5360，admission span 4.09/4.89s | `vllm_admission_diagnosis.json` row `vLLM-c8-prebuild-w4`; `headline_scorecard.json` -> `vllm_c8_diagnostics.prebuild_w4` | 当前最强 offline diagnostic；仍不是 online serving parity。 |

## 6. 反例数字

| 反例 | 推荐数字 | 主机器来源 | 现场解释 |
| --- | --- | --- | --- |
| naive preproc=2 | QPS 1.642 vs baseline c=8 QPS 2.540，latency 4.579/6.313s vs 3.064/5.853s | `tables_summary.json` -> `preprocessing_concurrency`; `claims_verification.json` -> `naive preprocessing parallelism regresses or fails` | 朴素放大 preprocessing 会把 queue 问题变成资源争用。 |
| naive preproc=4 | 43/50 useful，7 failed/OOM | `tables_summary.json` -> `preprocessing_concurrency`; `acceptance_matrix.json` -> `anti_recipe_failure` | 不能作为当前 recipe。 |
| c=16 默认服务点 | QPS 从 c=8 的 2.540 降到 2.407，RTF/tail 变差 | `headline_scorecard.json`; `acceptance_matrix.json` -> `not_recommended_saturation` | c=16 是压力边界，不是推荐默认点。 |

## 7. 重生成命令

完整审计：

```bash
cd /home/gangouyu/sglang-omni

python3 -m benchmarks.eval.run_qwen35_omni_report_audit \
  --root /home/gangouyu/sglang-omni \
  --summary-output results/qwen35_report_audit_20260619/audit_run_summary.json
```

单独重生成关键数字来源：

```bash
python3 -m benchmarks.eval.build_qwen35_omni_headline_scorecard \
  --root /home/gangouyu/sglang-omni \
  --json-output results/qwen35_report_audit_20260619/headline_scorecard.json

python3 -m benchmarks.eval.summarize_qwen35_omni_report_artifacts \
  --root /home/gangouyu/sglang-omni \
  --check-only \
  --json-output results/qwen35_report_audit_20260619/tables_summary.json

python3 -m benchmarks.eval.summarize_qwen35_stage_interactions \
  --root /home/gangouyu/sglang-omni \
  --json-output results/qwen35_report_audit_20260619/stage_interaction_summary.json

python3 -m benchmarks.eval.diagnose_vllm_offline_admission \
  --case vLLM-c4 \
  results/qwen35_vllm_videoamme_ci50_official_talker_compile_c4_20260618_202106/benchmark_audio_50_c4_offline_compile/videoamme_results.json \
  results/qwen35_vllm_videoamme_ci50_official_talker_compile_c4_20260618_202106/run.log 4 \
  --case vLLM-c8 \
  results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_20260619_20260619_222434/benchmark_audio_50_c8_offline_compile/videoamme_results.json \
  results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_20260619_20260619_222434/run.log 8 \
  --case vLLM-c8-prebuild-w1 \
  results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuild_20260620_002020/benchmark_audio_50_c8_offline_compile/videoamme_results.json \
  results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuild_20260620_002020/run.log 8 \
  --case vLLM-c8-prebuild-w4 \
  results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuildw4_20260620_005346/benchmark_audio_50_c8_offline_compile/videoamme_results.json \
  results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuildw4_20260620_005346/run.log 8 \
  --json-output results/qwen35_report_audit_20260619/vllm_admission_diagnosis.json
```

说明：现场优先跑 full audit；单独重生成时，上面的命令会覆盖当前
`vllm_admission_diagnosis.json`。

## 8. 复现 gate

期望 gate：

- full audit `ok=true`
- claims `17/17`
- coverage `34/34`
- preflight `62` checks, `0` required failures
- manifest current `196` records, minimum `180`, `0` missing
- SGLang optimization lock `26/26`
- repro command manifest `63` commands / `7` phases
- metric provenance index `ready=true`
- stage reproduction drilldown `ready=true`
- stage route decision matrix `ready=true`
- claim metric crosswalk `ready=true`
- objective requirement crosswalk `ready=true`
- headline scorecard `9/9`
- acceptance matrix `17/17`
- confidence ledger `12/12`
