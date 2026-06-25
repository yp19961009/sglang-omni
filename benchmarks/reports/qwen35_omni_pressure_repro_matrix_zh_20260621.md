# Qwen3.5-Omni 压力条件复现快速矩阵

状态：证据门已就绪；更新后的目标不再等待 6.21 晚间，后续变更必须重跑 full audit。
工作目录：`/home/gangouyu/sglang-omni`。
用途：按压力条件查复跑入口，避免 reviewer 在完整复现清单、stage drilldown 和 command manifest 之间来回翻。

## 1. 使用方法

所有命令正文以 `results/qwen35_report_audit_20260619/repro_command_manifest.json` 为准。本文只引用
稳定的 command id、关键证据文件和替换边界。查看某条命令：

```bash
cd /home/gangouyu/sglang-omni
jq -r '.commands[] | select(.id=="sglang_videoamme_stress") | .command' \
  results/qwen35_report_audit_20260619/repro_command_manifest.json
```

复跑任何性能数字后，都要再跑 `run_full_audit`；只跑 raw benchmark 不足以替换报告数字。

## 2. 快速矩阵

| 目标 / 压力条件 | command id | 运行位置 | 首要证据 | 预期形态 | 是否可替换 headline |
| --- | --- | --- | --- | --- | --- |
| 收包快检 | `bash benchmarks/eval/qwen35_omni_receiver_quickcheck.sh` | host | `share_package_validation.json`、`share_package_receiver_smoke_validation.json` | checksum OK；tarball `17/17`；receiver smoke `17/17`；extracted-only `13/13`；standalone `8/8` | 否，只证明包可接收 |
| 全量审计 | `run_full_audit` | host | `audit_run_summary.json`、`final_readiness_audit.json` | `ok=true`；final readiness `49/49`；required failures 为 `0` | 是，所有替换评审的前置门 |
| SGLang 服务启动 | `launch_sglang_optimized` | SGLang container | `runtime_image_contract.json`、`sglang_optimization_lock.json` | compiled/graph recipe、16GiB preprocessing cache、c4-c8 serving window | 否，是复跑前置条件 |
| 严格 SGLang vs vLLM headline | `sglang_videoamme_stress` + `sglang_recompute_wer` + `vllm_c4_original` + `build_headline_scorecard` + `build_runtime_comparison_contract` | container + host | `headline_scorecard.json`、`runtime_comparison_contract.json`、`claims_verification.json` | warmed c=4 SGLang latency/RTF 优于优化 vLLM，accuracy/WER 不退化 | 只有同硬件、同镜像、同模型/cache、同 ASR/WER 且 full audit 全绿 |
| SGLang 低并发 c=1/2/4 | `sglang_videoamme_stress` | SGLang container | `tables_summary.json`、`stage_latency_budget.json`、`stage_interaction_summary.json` | serving window；主要 tail 是 `talker_ar`，handoff 健康 | 可确认形态；不单独替换 strict headline |
| SGLang 高并发 c=8 | `sglang_videoamme_stress` | SGLang container | `headline_scorecard.json`、`serving_capacity_matrix.json`、`stage_boundary_bottleneck_ledger.json` | 当前吞吐峰值；admission/queue 开始显性化；quality 稳定 | 同环境全绿后可替换容量数字 |
| SGLang c=16 饱和边界 | `sglang_videoamme_stress` | SGLang container | `serving_capacity_matrix.json`、`stage_latency_budget.json`、`regime_decision_matrix.json` | 吞吐低于 c=8；queue/admission 加重；不推荐默认 | 否，除非新增证据推翻 saturation verdict |
| 短文本 + 语音 c=1/4/8 | `sglang_synthetic_text_to_speech` | SGLang container | `length_regime_coverage.json`、`tables_summary.json`、`stage_latency_budget.json` | 固定 short prompt 为 74 chars / 12 words；handoff 仍健康 | 只替换 short guardrail，不能替换 Video-AMME headline |
| 长文本 + 语音 c=1/4/8 | `sglang_synthetic_text_to_speech` | SGLang container | `length_regime_coverage.json`、`tail_confidence_appendix.json`、`stage_latency_budget.json` | 固定 long prompt 为 944 chars / 139 words；long c=8 RTF 小于 1 | 只替换 long guardrail，不能替换 Video-AMME headline |
| SGLang WER/ASR | `check_wer_asr_path` + `sglang_recompute_wer` | host + container | `claims_verification.json`、`headline_scorecard.json`、WER JSON | host Whisper cache 缺失是 optional warning；必须保留同一 ASR/WER 口径 | quality 证据必需；不能单独替换吞吐 |
| vLLM strict c=4 baseline | `vllm_c4_original` + `summarize_vllm_log_stages` + `diagnose_vllm_admission` | host | `vllm_log_stage_summary.json`、`vllm_admission_diagnosis.json`、`runtime_image_contract.json` | Qwen3.5-capable image；compile/CUDA graph/cache/shared-memory/encoder 证据保留 | 同环境全绿后可替换 strict baseline |
| vLLM original c=8 | `vllm_c8_original` + `diagnose_vllm_admission` | host | `vllm_admission_diagnosis.json`、`vllm_online_parity_protocol.json` | prompt build/feed admission limited | 否，不能写成 online parity |
| vLLM c=8 prebuild w4 | `vllm_c8_prebuild_w4` + `diagnose_vllm_admission` + `build_vllm_online_parity_protocol` | host | `optimization_candidate_ledger.json`、`vllm_optimization_lock.json`、`vllm_online_parity_protocol.json` | optimized offline diagnostic；runner wall 改善；仍非 online parity | 否，除非补齐 online ingress、WER/ASR 和 stage evidence |
| stage breakdown 再生成 | `build_stage_interactions` + `build_stage_latency_budget` + `build_stage_boundary_bottleneck_ledger` + `build_stage_causal_graph` + `build_stage_reproduction_drilldown` | host | `stage_interaction_summary.json`、`stage_latency_budget.json`、`stage_boundary_bottleneck_ledger.json`、`stage_causal_graph.json` | handoff 健康；code2wav decode 不是主瓶颈；c8/c16 queue/admission 解释一致 | 只有 full audit、claim/crosswalk/route gate 同步全绿 |
| 复跑验收 | `build_rerun_acceptance_contract` + `build_rerun_delta_triage` | host | `rerun_acceptance_contract.json`、`rerun_delta_triage.json` | 18 rules、34 return evidence files、27 command return rows | 替换评审必须通过 |

## 3. 必跑顺序

1. 先跑 `bash benchmarks/eval/qwen35_omni_receiver_quickcheck.sh`，确认包本身没坏。
2. 跑 `run_full_audit`，确认当前证据链是完成态。
3. SGLang 复跑前先启动 `launch_sglang_optimized`。
4. 按压力条件跑 SGLang / vLLM command id。
5. 跑 WER/ASR 或记录 `check_wer_asr_path` 的 optional warning。
6. 跑 `run_full_audit` 重新生成表格、图、stage JSON、crosswalk、share bundle 和 release seal。
7. 用 `qwen35_omni_collaborator_rerun_validation_sheet_zh_20260621.md` 填写复跑结果。
8. 用 `qwen35_omni_rerun_delta_triage_zh_20260621.md` 判断差异是环境、admission、talker、handoff、code2wav、WER 还是 vLLM parity 边界。

## 4. 快查命令

列出核心复跑命令：

```bash
cd /home/gangouyu/sglang-omni
jq -r '.commands[]
  | select(.id as $id
    | ["run_full_audit","launch_sglang_optimized","sglang_videoamme_stress",
       "sglang_synthetic_text_to_speech","check_wer_asr_path","sglang_recompute_wer",
       "vllm_c4_original","vllm_c8_original","vllm_c8_prebuild_w4"] | index($id))
  | [.id, .where, .phase, .purpose] | @tsv' \
  results/qwen35_report_audit_20260619/repro_command_manifest.json
```

查看当前总门禁：

```bash
cd /home/gangouyu/sglang-omni
jq '{ok, final_readiness, share_package_validation,
     share_package_receiver_smoke_validation, receiver_quickcheck_contract,
     share_release_seal, final_completion_audit}' \
  results/qwen35_report_audit_20260619/audit_run_summary.json
```

验证复跑后是否允许替换：

```bash
cd /home/gangouyu/sglang-omni
jq '{runtime_image_contract, rerun_acceptance_contract,
     vllm_online_parity_protocol, final_readiness}' \
  results/qwen35_report_audit_20260619/audit_run_summary.json
```

## 5. 替换红线

- 硬件、SGLang/vLLM image digest、模型路径、Video-AMME cache 或 ASR/WER 口径任一不同，只能作为外部复核附录。
- `run_full_audit`、claim gate、acceptance gate、confidence gate、stage gate 或 share package validation 任一 required gate 失败，不替换公开数字。
- vLLM c=8 prebuild w4 只证明 offline diagnostic；没有 online serving ingress、同口径 WER/ASR 和 stage evidence 前，不升级为 online parity。
- c=16 是 saturation boundary，不是当前推荐默认点。
- 只改 Markdown 或图表、没有同步 JSON evidence、manifest、tarball checksum 和 receiver quickcheck，属于 silent replacement risk。
