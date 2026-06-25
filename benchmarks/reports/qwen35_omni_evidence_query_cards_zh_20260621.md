# Qwen3.5-Omni Evidence Query Cards

状态：2026-06-21 evidence-ready 交付稿；更新后的目标不再等待 6.21 晚上。
工作目录：`/home/gangouyu/sglang-omni`。
用途：给答辩、合作方复核和现场追问使用。每张卡片都给出可复制的 `jq` 查询，
直接从机器证据证明核心结论。
一键执行这些查询并做核心断言时，可运行：
`bash benchmarks/eval/qwen35_omni_evidence_query_cards_smoke.sh`。
host 仓库态和解包 bundle 态也可显式指定：
`bash benchmarks/eval/qwen35_omni_evidence_query_cards_smoke.sh --root "$HOST_REPO" --mode host`；
解包后运行
`bash "$BUNDLE_ROOT/benchmarks/eval/qwen35_omni_evidence_query_cards_smoke.sh" --root "$BUNDLE_ROOT" --mode portable`。
这条 smoke 的 `PASS`/skip 摘要打印在 stdout；`--output` 只保存查询卡 bash block
的 JSON/文本正文。

## 1. 总门禁卡

证明当前包和报告是完成态 evidence，而不是只改了 Markdown：

```bash
cd /home/gangouyu/sglang-omni
jq '{ok, final_readiness, share_package_validation,
     share_package_receiver_smoke_validation, receiver_quickcheck_contract,
     share_release_seal, final_completion_audit}' \
  results/qwen35_report_audit_20260619/audit_run_summary.json
```

期望：`ok=true`，final readiness `49/49`，tarball validation `17/17`，receiver quickcheck
contract `15/15`，release seal `14/14`。更新后的目标不再等待 6.21 晚上；
`completion_allowed_now=true` 表示可以进入最终完成裁决。

## 2. strict c=4 SGLang vs vLLM 卡

证明严格 headline 只来自 warmed c=4 apples-to-apples 对比：

```bash
cd /home/gangouyu/sglang-omni
jq '{scope: .strict_c4_comparison.scope,
     sglang: (.strict_c4_comparison.sglang
       | {accuracy, latency_mean_s, latency_p95_s, rtf_mean, rtf_p95, wer_corpus}),
     vllm: (.strict_c4_comparison.vllm
       | {accuracy, latency_mean_s, latency_p95_s, rtf_mean, rtf_p95, wer_corpus}),
     relative_sglang_lower_pct: .strict_c4_comparison.relative_sglang_lower_pct,
     checks: .summary.checks}' \
  results/qwen35_report_audit_20260619/headline_scorecard.json
```

期望：SGLang latency/RTF 更低；accuracy 不低于 vLLM；WER 不高于 vLLM；
`strict_c4_sglang_latency_rtf_win=true` 且 `strict_c4_accuracy_wer_preserved=true`。

## 3. SGLang c=8 吞吐峰值卡

证明 c=8 是当前 SGLang 压测峰值，c=16 是饱和边界：

```bash
cd /home/gangouyu/sglang-omni
jq '{throughput_peak: (.sglang_stress.throughput_peak
       | {concurrency, throughput_qps, latency_mean_s, rtf_mean, accuracy, wer_corpus}),
     c16_vs_c8_qps_delta_pct: .sglang_stress.c16_vs_c8_qps_delta_pct,
     rows: [.sglang_stress.rows[]
       | {concurrency, throughput_qps, latency_mean_s, rtf_mean, accuracy, wer_corpus}],
     checks: .summary.checks}' \
  results/qwen35_report_audit_20260619/headline_scorecard.json
```

期望：`throughput_peak.concurrency=8`；c=16 相对 c=8 的 QPS delta 为负；
accuracy/WER 没有用质量换吞吐。

## 4. 短/长文本语音卡

证明短/长文本输入的固定 prompt 形态、机器 coverage gate，以及 long c=8 仍快于实时：

```bash
cd /home/gangouyu/sglang-omni
jq '{summary: .summary}' \
  results/qwen35_report_audit_20260619/length_regime_coverage.json
jq '.tables.synthetic_speech[]
  | {scenario, concurrency, n, target_chars, target_words,
     audio_duration_mean_s, latency_mean_s, rtf_mean, rtf_p95, throughput_qps}' \
  results/qwen35_report_audit_20260619/tables_summary.json
```

期望：length coverage `10/10`、rows=`7`；short prompt 为 74 chars / 12 words；
long prompt 为 944 chars / 139 words；long c=8 的 RTF p95 小于 1；handoff/decode
guard p95 均在几十毫秒级。

## 5. stage 连接健康卡

证明 stage 之间的连接没有隐藏瓶颈，且主要瓶颈归因清楚：

```bash
cd /home/gangouyu/sglang-omni
jq '.summary' results/qwen35_report_audit_20260619/stage_interaction_summary.json
```

期望：`sglang_talker_to_code2wav_healthy=true`，
`sglang_code2wav_decode_not_bottleneck=true`，
`vllm_original_c8_prompt_feed_limited=true`，
`preprocessing_parallelism_regresses=true`。

## 6. stage budget 和 boundary 卡

证明 c=8/c=16 的 queue/admission 压力、long c=8 RTF、vLLM c=8 诊断和 boundary ledger
都已经机器化：

```bash
cd /home/gangouyu/sglang-omni
jq '{latency_budget: .summary}' \
  results/qwen35_report_audit_20260619/stage_latency_budget.json
jq '{boundary_ledger: .summary}' \
  results/qwen35_report_audit_20260619/stage_boundary_bottleneck_ledger.json
```

期望：stage latency budget `12/12`；boundary ledger `12/12`；c=8/c=16 queue 占比和
`recommended_sglang_window=c4-c8`、`saturation_boundary=c16` 都存在。

## 7. vLLM baseline 强度卡

证明 vLLM 不是弱 baseline，且 c=8 prebuild w4 只作为 offline diagnostic：

```bash
cd /home/gangouyu/sglang-omni
jq '{summary: .summary,
     required_switches: [.required_switches[] | {switch, purpose}]}' \
  results/qwen35_report_audit_20260619/vllm_optimization_lock.json
jq '{summary: .summary,
     current_boundary: .current_boundary,
     required_online_artifacts: .required_online_artifacts}' \
  results/qwen35_report_audit_20260619/vllm_online_parity_protocol.json
```

期望：vLLM optimization lock `22/22`；compile/CUDA graph、prefix/chunked prefill、
shared-memory transfer、encoder compile/batch 和 prebuild w4 都能看到；
`online_parity_proven=false`。

## 8. vLLM c=8 diagnostic 卡

证明 original c=8 是 prompt-feed/admission limited，prebuild w4 是 optimized offline
diagnostic：

```bash
cd /home/gangouyu/sglang-omni
jq '{original: (.vllm_c8_diagnostics.original
       | {label, runner_overhead_pct_wall, batch_admission_span_avg_ms,
          diagnosis, prompt_feed_limited}),
     prebuild_w4: (.vllm_c8_diagnostics.prebuild_w4
       | {label, runner_qps, engine_qps, batch_admission_span_avg_ms,
          diagnosis, prompt_feed_limited})}' \
  results/qwen35_report_audit_20260619/headline_scorecard.json
```

期望：original c=8 `prompt_feed_limited=true`；prebuild w4 runner/engine 时钟分离；
不把该结果写成 online serving parity。

## 9. 复跑替换边界卡

证明外部复跑能否替换 headline 有机器合同约束：

```bash
cd /home/gangouyu/sglang-omni
jq '{summary: .summary,
     rules_total: .summary.rules_total,
     replacement_scope: .summary.replacement_scope,
     default_decision: .summary.default_decision}' \
  results/qwen35_report_audit_20260619/rerun_acceptance_contract.json
jq '{summary: .summary}' \
  results/qwen35_report_audit_20260619/rerun_delta_triage.json
```

期望：rerun acceptance contract `17/17`；18 rules；34 return evidence files；
27 command return rows；默认策略是同环境全绿前只确认形态，不静默替换 headline。

## 10. 分享包自证卡

证明 tarball、receiver smoke、extracted-only 和 standalone validation 都保护了接收方路径：

```bash
cd /home/gangouyu/sglang-omni
jq '{package: .summary}' \
  results/qwen35_report_audit_20260619/share_package_validation.json
jq '{receiver_smoke: .summary}' \
  results/qwen35_report_audit_20260619/share_package_receiver_smoke_validation.json
jq '{contract: .summary}' \
  results/qwen35_report_audit_20260619/receiver_quickcheck_contract.json
```

期望：tarball validation `17/17`；receiver smoke `17/17`；receiver quickcheck contract
`15/15`，且 stage dictionary crosswalk needles 为 `8`；asset quality offenders 为空；WER/ASR path guard 存在。

## 11. stage quick reproduction route 卡

证明现场答辩的 5 条 stage quick route 能从问题、stage row、metric row 一路追到
rerun command 和 jq 查询：

```bash
cd /home/gangouyu/sglang-omni
jq '{summary: (.summary
      | {ready, checks_passed, checks_total, required_failures,
         quick_reproduction_routes_total}),
     quick_routes: [.quick_reproduction_map[]
       | {question, runtime, case_name: .["case"], stage_row_id,
          metric_row_id, bottleneck_verdict, first_rerun_command_id,
          stage_query, metric_query}]}' \
  results/qwen35_report_audit_20260619/stage_reproduction_drilldown.json
```

期望：stage reproduction drilldown `17/17`；`quick_reproduction_routes_total=5`；
5 条答辩 quick route 均有 `stage_row_id`、`metric_row_id`、
`first_rerun_command_id`、`stage_query` 和 `metric_query`，可继续查
`stage_drilldown_index.json` 与 `metric_provenance_index.json`。

## 12. 一致性和 preflight alias 防漂移卡

证明公开报告、机器证据、manifest expected-gates、preflight sidecar 和 evidence-query 路由没有漂移：

```bash
cd /home/gangouyu/sglang-omni
jq '{summary: (.summary
      | {ready, checks_passed, checks_total, required_failures,
         public_stale_hits, machine_stale_hits,
         university_review_packet_checks,
         university_review_packet_glossary_missing,
         manifest_expected_gate_ready, manifest_expected_gate_unexpected_fields,
         preflight_alias_ready, preflight_alias_present,
         preflight_alias_mismatches, preflight_repro_in_manifest,
         preflight_alias_in_manifest, evidence_smoke_route_missing})}' \
  results/qwen35_report_audit_20260619/share_consistency_guard.json
```

期望：share consistency guard `17/17`；public/machine stale hits 都为 `0`；
university review packet gate 为 `14/14 checks`，术语口径缺失为空；
manifest expected-gates 没有旧 `total_records` 模糊字段；`preflight_repro.json`
在 manifest 内，`preflight.json` 不进 manifest 且如存在必须与官方 preflight 字节一致；
host/portable evidence-query 路由没有缺口。

## 13. SGLang 优化锁和 current-best recipe 卡

证明 SGLang 自身不是弱 base，而是已经锁定 compiled/graph recipe、c4-c8 推荐窗口、
c8 吞吐峰值、c16 饱和边界和 preprocessing 并发反例：

```bash
cd /home/gangouyu/sglang-omni
jq '{summary: .summary,
     recipe_switches: [.recipe_switches[] | {switch, purpose}],
     stress_rows: [.stress_rows[] | {concurrency, throughput_qps, latency_mean_s,
       latency_p95_s, rtf_mean, rtf_p95, accuracy, wer_corpus}]}' \
  results/qwen35_report_audit_20260619/sglang_optimization_lock.json
jq '{summary: .summary,
     sglang_rows: [.rows[]
       | select(.runtime == "sglang")
       | {candidate_id, decision_class, decision, rationale, replacement_rule}]}' \
  results/qwen35_report_audit_20260619/optimization_candidate_ledger.json
```

期望：SGLang optimization lock `26/26`；recipe switches 至少包含 Thinker/Talker
CUDA graph、Talker torch compile、max-running=8、16GiB preprocessing cache 和
`PREPROCESSING_MAX_CONCURRENCY=1`；优化 ledger 中
`sglang_current_best_measured_recipe` 是当前 accepted recipe，preproc=2/4 是两个
rejected anti-recipe；c=8 QPS 高于 c=16。

## 14. 公开文档质量和 semantic count drift 卡

证明公开文档质量门禁不仅检查 hash/table/token/duplicate heading，也会检查类似
“N 条红线”标题和实际 bullet 数不一致的语义漂移：

```bash
cd /home/gangouyu/sglang-omni
jq '{hard_gate: .summary.hard_gates.public_doc_quality_guard,
     public_quality_check: [.checks[]
       | select(.name == "public share docs quality guard")
       | {status, required, evidence}]}' \
  results/qwen35_report_audit_20260619/final_readiness_audit.json
```

期望：hard gate 包含 `semantic count drift`；public quality check 为 PASS；
evidence 中 `semantic_offenders=[]`，同时 hash/table/token/duplicate-heading
offenders 都为空。

## 15. 复跑耗时/算力预算卡

证明合作方复跑排期预算不是口头估计，而是能追到 command IDs、timed lower bound 和
WER/ASR 未计时边界：

```bash
cd /home/gangouyu/sglang-omni
jq '{summary: .summary,
     budget_rows: [.rows[]
       | {budget_id, command_ids, runs, requests,
          timed_wall_lower_bound_s, equivalent_gpu_hours_lower_bound,
          boundary}]}' \
  results/qwen35_report_audit_20260619/rerun_time_budget.json
```

期望：rerun time budget `ready=true`；9 个 budget row；6 个 timed row；
required failures 为 `0`；总计时段下界约 `1592.8s`，8-GPU 等效下界约
`3.540 GPUh`；并明确不包含 server launch/warmup/WER/ASR/downloads/cache
population/manual review。

## 16. 复跑命令引用不断链卡

证明公开报告、结构化证据和 `repro_command_manifest.json` 里的复跑命令 ID 没有断链，
reviewer 看到的 stage / vLLM / package 命令都能回到同一份 manifest：

```bash
cd /home/gangouyu/sglang-omni
jq '{summary: (.summary
      | {ready, checks_passed, checks_total, required_failures,
         manifest_commands_total, structured_artifacts_total,
         structured_artifacts_with_refs_total, structured_command_refs_total,
         structured_unique_command_refs_total, unresolved_command_refs_total,
         critical_command_doc_refs_total, missing_critical_doc_command_ids})}' \
  results/qwen35_report_audit_20260619/command_reference_hygiene.json
```

期望：command reference hygiene `6/6`；manifest commands=`63`；
structured artifacts with refs=`9/9`；structured unique command refs=`63`；
unresolved command refs=`0`；critical command doc refs 大于 `0`；
missing critical command IDs 为空。
