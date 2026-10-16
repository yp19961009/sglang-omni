# Qwen3.5-Omni 最终状态摘要

生成时间 UTC：`2026-06-21T02:00:21.270039+00:00`。
工作目录：`/home/gangouyu/sglang-omni`。

这是一页给合作方或内部 reviewer 的状态摘要。数字来自当前审计 JSON；若和其他
文档不一致，以 `audit_run_summary.json`、`objective_completion_audit.json` 和
`final_readiness_audit.json`、`final_completion_audit.json` 为准。

## 1. 当前结论

- 当前 evidence package 可以作为分享版本发送，但必须携带 caveat。
- 原始目标逐项审计为 `share_ready_with_documented_caveats=true`，不是把长线 goal 标记完成。
- 严格 headline 是 SGLang-Omni Qwen3.5 在 warmed c=4 上优于优化版 vLLM c=4，且 accuracy/WER 不退化。
- 高并发推荐窗口是 c=4 到 c=8；c=16 是压力边界，不是默认服务点。
- ci-50/stress/synthetic 证据不能直接外推到完整线上流量；更大 Video-AMME 或真实线上流量需要同口径复跑和 gate 全绿。

## 2. 机器 Gate

| Gate | Status | Evidence |
| --- | --- | --- |
| full audit | `PASS` | `audit_run_summary.json` |
| objective completion | `PASS` | `17` rows, `0` required failures |
| final readiness | `PASS` | `49/49` checks |
| final completion audit | `PASS` | `13` checks, `0` tracking, `0` required failures, completion_allowed_now=`True`, blockers=`[]` |
| runtime fairness contract | `PASS` | `9/9` checks, warmed c=4 only, offline_diagnostic_not_online_parity |
| public doc quality guard | `PASS` | `public_doc_quality_guard`=`no bare hashes / malformed tables / malformed tokens / duplicate headings / semantic count drift`; hash/table/token/duplicate-heading/semantic-count offenders all empty |
| manifest | `PASS` | `196` records, `0` missing |
| repro command manifest | `PASS` | `63` commands, `7` phases |
| slide asset map | `PASS` | `10` rows, `5/5` checks |
| share bundle manifest | `PASS` | `122` records, `0` missing required |
| share consistency guard | `PASS` | `17/17` checks, stale public/machine hits=`0/0`, embedded leaks=`0`, current identity mismatches=`0`; full post-validation hash chain is sealed by adjacent release seal |
| runtime image contract | `PASS` | `12/12` checks, SGLang `c4-c8 warmed serving; c8 peak throughput; c16 saturation boundary`, vLLM `prebuild w4 is optimized offline diagnostic, not online parity` |
| rerun acceptance contract | `PASS` | `17/17` checks, `18` rules, `34` return evidence files, `27` command matrix rows |
| rerun time budget | `PASS` | `9` rows, `6` timed rows, timed lower bound=`1592.844095361441s`, 8-GPU lower bound=`3.5396535452476465 GPUh` |
| rerun delta triage | `PASS` | `19` symptoms, `8/8` checks |
| final checkpoint watchlist | `PASS` | `24/24` checks, `7` watch items |
| stage latency budget | `PASS` | `12/12` checks, SGLang `5` / synthetic `6` / vLLM `4` rows |
| length regime coverage | `PASS` | `10/10` checks, `7` rows, long c=8 RTF p95=`0.5001`, max hop/decode p95=`24.031/24.269ms` |
| stage boundary bottleneck ledger | `PASS` | `12/12` checks, `37` boundary rows |
| stage causal graph JSON | `PASS` | `7/7` checks, `7` edges, `5` drilldown rows |
| share tarball | `PASS` | checksum `results/qwen35_report_audit_20260619/qwen35_omni_share_bundle_20260621.tar.gz.sha256` |
| share package validation | `PASS` | `17/17` checks, `0` required failures, `report_quality_offenders=[]`, `chart_quality_offenders=[]` |
| receiver smoke validation | `PASS` | `17/17` checks, receiver_smoke_ready=true, nested extracted-only `13/13` |
| extracted-only validation | `PASS` | `13/13` checks, extracted_only=true, `report_quality_offenders=[]`, `chart_quality_offenders=[]` |
| external standalone validation | `PASS` | `8/8` checks, bundled validator, nested extracted-only `13` checks |
| receiver quickcheck contract | `PASS` | `15/15` checks, `4` receiver JSONs, `8` public docs, `6` completion-gate docs, `3` WER/ASR docs, `6` evidence-smoke docs |
| collaborator return check | `PASS` | `16/16` checks, `34/34` return files, `27/27` command rows, decision=`eligible_for_current_scope_headline_replacement_review` |
| share release seal | `PASS` | `14/14` checks, receiver_smoke_ready=true, forbidden_tarball_members=`[]`, send_decision=`send_tarball_with_adjacent_seal_and_caveats` |

## 3. 发送文件

- 便捷 tarball：`results/qwen35_report_audit_20260619/qwen35_omni_share_bundle_20260621.tar.gz`
- tarball checksum：`results/qwen35_report_audit_20260619/qwen35_omni_share_bundle_20260621.tar.gz.sha256`
- 随包逐文件 hash 清单：tarball 内 `PACKAGE_FILE_SHA256SUMS.txt`
- 伴随 package manifest：`results/qwen35_report_audit_20260619/share_bundle_package_manifest.json`
- 伴随 package validation：`results/qwen35_report_audit_20260619/share_package_validation.json`
- 伴随 extracted validation：`results/qwen35_report_audit_20260619/share_package_validation_extracted.json`
- 伴随 receiver smoke validation：`results/qwen35_report_audit_20260619/share_package_receiver_smoke_validation.json`
- 伴随 external standalone validation：`results/qwen35_report_audit_20260619/share_package_external_standalone_validation.json`
- 伴随 receiver quickcheck contract JSON：`results/qwen35_report_audit_20260619/receiver_quickcheck_contract.json`
- 伴随 collaborator return check JSON：`results/qwen35_report_audit_20260619/collaborator_return_check.json`
- 伴随 final completion audit JSON：`results/qwen35_report_audit_20260619/final_completion_audit.json`
- 伴随 final completion audit 报告：`benchmarks/reports/qwen35_omni_final_completion_audit_zh_20260621.md`
- 伴随 share release seal JSON：`results/qwen35_report_audit_20260619/share_release_seal.json`
- 伴随 share release seal 报告：`benchmarks/reports/qwen35_omni_share_release_seal_zh_20260621.md`
- 主报告：`benchmarks/reports/qwen35_omni_stress_performance_plan_20260621.md`
- 分享索引：`benchmarks/reports/qwen35_omni_share_package_index_zh_20260621.md`
- 高校合作方审阅会议包：`benchmarks/reports/qwen35_omni_university_review_packet_zh_20260621.md`
- 高校合作方审阅会议包 JSON：`results/qwen35_report_audit_20260619/university_review_packet.json`
- 分享一致性 guard：`benchmarks/reports/qwen35_omni_share_consistency_guard_zh_20260621.md`
- 分享一致性 guard JSON：`results/qwen35_report_audit_20260619/share_consistency_guard.json`
- 接收方路径手册：`benchmarks/reports/qwen35_omni_receiver_package_path_map_zh_20260621.md`
- 接收方命令卡：`benchmarks/reports/qwen35_omni_receiver_command_card_zh_20260621.md`
- 接收方 quickcheck contract：`benchmarks/reports/qwen35_omni_receiver_quickcheck_contract_zh_20260621.md`
- 一页式 scorecard：`benchmarks/reports/qwen35_omni_one_page_scorecard_zh_20260621.md`
- Deck 图表资产映射：`benchmarks/reports/qwen35_omni_slide_asset_map_zh_20260621.md`
- 图表来源一致性报告：`benchmarks/reports/qwen35_omni_chart_source_consistency_zh_20260621.md`
- 图表来源一致性 JSON：`results/qwen35_report_audit_20260619/chart_source_consistency.json`
- 决策矩阵：`benchmarks/reports/qwen35_omni_regime_decision_matrix_zh_20260621.md`
- 决策矩阵 JSON：`results/qwen35_report_audit_20260619/regime_decision_matrix.json`
- 原始需求证据映射：`benchmarks/reports/qwen35_omni_requirement_evidence_map_zh_20260621.md`
- 压力条件总表：`benchmarks/reports/qwen35_omni_pressure_matrix_zh_20260621.md`
- 指标来源索引：`benchmarks/reports/qwen35_omni_metric_source_map_zh_20260621.md`
- 公平对比合同：`benchmarks/reports/qwen35_omni_runtime_comparison_contract_zh_20260621.md`
- 公平对比合同 JSON：`results/qwen35_report_audit_20260619/runtime_comparison_contract.json`
- Runtime image contract：`benchmarks/reports/qwen35_omni_runtime_image_contract_zh_20260621.md`
- 复跑验收阈值合同：`benchmarks/reports/qwen35_omni_rerun_acceptance_contract_zh_20260621.md`
- 复跑耗时/算力预算：`benchmarks/reports/qwen35_omni_rerun_time_budget_zh_20260621.md`
- 复跑耗时/算力预算 JSON：`results/qwen35_report_audit_20260619/rerun_time_budget.json`
- 复跑差异定位矩阵：`benchmarks/reports/qwen35_omni_rerun_delta_triage_zh_20260621.md`
- SGLang 优化锁定矩阵：`benchmarks/reports/qwen35_omni_sglang_optimization_lock_zh_20260621.md`
- vLLM 优化锁定矩阵：`benchmarks/reports/qwen35_omni_vllm_optimization_lock_zh_20260621.md`
- 优化候选裁决 ledger：`benchmarks/reports/qwen35_omni_optimization_candidate_ledger_zh_20260621.md`
- 优化候选裁决 JSON：`results/qwen35_report_audit_20260619/optimization_candidate_ledger.json`
- Caveat 裁决矩阵 JSON：`results/qwen35_report_audit_20260619/caveat_adjudication_matrix.json`
- vLLM c=8 online parity 升级协议：`benchmarks/reports/qwen35_omni_vllm_online_parity_protocol_zh_20260621.md`
- 最终 checkpoint watchlist：`benchmarks/reports/qwen35_omni_final_checkpoint_watchlist_zh_20260621.md`
- Stage latency budget：`benchmarks/reports/qwen35_omni_stage_latency_budget_zh_20260621.md`
- 长短输入/输出 coverage：`benchmarks/reports/qwen35_omni_length_regime_coverage_zh_20260621.md`
- 长短输入/输出 coverage JSON：`results/qwen35_report_audit_20260619/length_regime_coverage.json`
- Stage boundary bottleneck ledger：`benchmarks/reports/qwen35_omni_stage_boundary_bottleneck_ledger_zh_20260621.md`
- Stage 因果图：`benchmarks/reports/qwen35_omni_stage_causal_graph_zh_20260621.md`
- Stage 因果图 JSON：`results/qwen35_report_audit_20260619/stage_causal_graph.json`
- Stage 指标字典：`benchmarks/reports/qwen35_omni_stage_metric_dictionary_zh_20260621.md`
- Caveat 裁决矩阵：`benchmarks/reports/qwen35_omni_caveat_adjudication_matrix_zh_20260621.md`
- 最终交付说明：`benchmarks/reports/qwen35_omni_final_share_delivery_note_zh_20260621.md`
- 外部复现 handoff runbook：`benchmarks/reports/qwen35_omni_external_handoff_runbook_zh_20260621.md`
- 复现清单：`benchmarks/reports/qwen35_omni_reproduction_checklist_zh_20260621.md`
- 合作方复跑验收表：`benchmarks/reports/qwen35_omni_collaborator_rerun_validation_sheet_zh_20260621.md`
- 复跑差异定位 JSON：`results/qwen35_report_audit_20260619/rerun_delta_triage.json`
- 答辩 Q&A：`benchmarks/reports/qwen35_omni_defense_qna_zh_20260621.md`
- 分享 deck 提纲：`benchmarks/reports/qwen35_omni_share_deck_outline_zh_20260621.md`
- 快速收包校验脚本：`benchmarks/eval/validate_qwen35_omni_share_package.py`
- 原始目标审计：`results/qwen35_report_audit_20260619/objective_completion_audit.json`
- Deck 图表资产映射 JSON：`results/qwen35_report_audit_20260619/slide_asset_map.json`

当前 tarball digest 的权威来源是同目录 `.sha256` 文件和
`share_bundle_package_manifest.json` 的 `tarball_sha256` 字段；随包 Markdown
不内嵌 tarball digest 数值，避免报告文本变更导致 tarball hash 自引用漂移。
随包 `share_consistency_guard` 是 17/17 的包内一致性 guard；包内副本可能显示
`tarball_identity_fields_active=3`、`tarball_identity_deferred_fields=3`，这是预期状态，
因为 tarball-mode validation、receiver smoke validation 和 release seal 必须在打包后生成。
完整 post-validation hash 链由相邻 `.sha256`、package manifest、validation、receiver smoke 和
share release seal 共同证明。

`share_bundle_package_manifest.json`、`share_package_validation.json`、
`share_package_validation_extracted.json`、`share_package_receiver_smoke_validation.json` 和
`share_package_external_standalone_validation.json`，以及
`receiver_quickcheck_contract.json`、`final_completion_audit.json`、`share_release_seal.json` 是
和 tarball 同目录保存或运行生成的伴随验证证据，
不是 tarball 内成员；它们描述 tarball 自身，放入 tarball 会造成自引用 hash。
`qwen35_omni_final_completion_audit_zh_20260621.md` 是给人读的最终 completion gate，
也应和 tarball 相邻保存。
`qwen35_omni_share_release_seal_zh_20260621.md` 是给人读的同一封口证据，
也应和 tarball 相邻保存。
`PACKAGE_FILE_SHA256SUMS.txt` 是 tarball 内成员，记录每个随包源文件的相对仓库根路径和
逐文件 hash；tarball-mode validator 会先用它直接校验 tar member 内容，接收方解包后
运行 extracted-only validator 时会再次复核报告、证据 JSON、工具脚本和图表资产。
同一个 validator 还会直接扫描随包 `share_report` Markdown 中的裸 hash、坏表格、重复 heading 和坏展示 token，
并检查随包 `share_charts` CSV/SVG 可解析、非空且结构可渲染；通过时 evidence 应显示
`report_quality_offenders=[]` 和 `chart_quality_offenders=[]`。

如果接收方仓库路径不是 `/home/gangouyu/sglang-omni`，先改 `HOST_REPO`；
`SMOKE_DIR` 和 `EXTRACT_DIR` 只用于收包校验和手动解包校验。

接收方可先执行：

```bash
HOST_REPO="${HOST_REPO:-/home/gangouyu/sglang-omni}"
SMOKE_DIR="${SMOKE_DIR:-/tmp/qwen35_omni_receiver_smoke_final_status}"
EXTRACT_DIR="${EXTRACT_DIR:-/tmp/qwen35_omni_share_bundle_final_status}"
STANDALONE_DIR="${STANDALONE_DIR:-/tmp/qwen35_omni_external_standalone_bundle_validation_final_status}"
cd "$HOST_REPO"
bash benchmarks/eval/qwen35_omni_receiver_quickcheck.sh
jq '.summary' results/qwen35_report_audit_20260619/receiver_quickcheck_contract.json
python3 -m benchmarks.eval.build_qwen35_omni_final_completion_audit \
  --root "$HOST_REPO" \
  --output benchmarks/reports/qwen35_omni_final_completion_audit_zh_20260621.md \
  --json-output results/qwen35_report_audit_20260619/final_completion_audit.json
python3 -m benchmarks.eval.build_qwen35_omni_share_release_seal \
  --root "$HOST_REPO" \
  --strict \
  --output benchmarks/reports/qwen35_omni_share_release_seal_zh_20260621.md \
  --json-output results/qwen35_report_audit_20260619/share_release_seal.json
python3 -m benchmarks.eval.run_qwen35_omni_report_audit \
  --root "$HOST_REPO" \
  --summary-output results/qwen35_report_audit_20260619/audit_run_summary.json
```

## 4. 必须携带的 Caveat

- vLLM c=8 online parity protocol 当前为 `ready=True`，`online_parity_proven=False`；升级 headline 前必须先补 online ingress artifacts。
- ci-50/stress/synthetic 证据不能直接外推到完整线上流量；更大 Video-AMME 或真实线上流量需要同口径 stage/tail/quality gate。
- c=16 is pressure-boundary evidence, not a recommended serving point.
- Official SeedTTS full-set is not staged as headline evidence in this package.
- Strict vLLM c=8 online serving parity still needs an online-ingress rerun if that claim is required.

## 5. 当前发送判断

- send decision：`ready_to_share_with_documented_caveats`
- university review packet：`ready=True`，`checks=14/14`
- long-running goal complete：`False`
- final checkpoint watchlist：`ready=True`，`completion_allowed_now=True`
- stage latency budget：`ready=True`，`checks=12/12`
- stage boundary bottleneck ledger：`ready=True`，`checks=12/12`，`rows=37`
- stage causal graph JSON：`ready=True`，`checks=7/7`，`edges=7`，`raw_drilldown=5`
- slide asset map：`ready=True`，`rows=10`，`checks=5/5`
- rerun delta triage：`ready=True`，`rows=19`，`checks=8/8`
- rerun time budget：`ready=True`，`rows=9`，`timed_rows=6`，`8gpu_lower_bound_gpuh=3.5396535452476465`
- runtime fairness contract：`ready=True`，`checks=9/9`，`headline=warmed c=4 only`
- external standalone validation：`ready=True`，`checks=8/8`
- receiver quickcheck contract：`ready=True`，`checks=15/15`，`wer_asr_docs=3`，`evidence_smoke_docs=6`，`required_failures=0`
- collaborator return check：`ready=True`，`checks=16/16`，`return_files=34/34`，`command_rows=27/27`，`decision=eligible_for_current_scope_headline_replacement_review`
- share consistency guard：`ready=True`，`checks=17/17`，`identity_mismatches=0`
- share release seal：`ready=True`，`checks=14/14`，`send_decision=send_tarball_with_adjacent_seal_and_caveats`

这页摘要证明当前 share package 已可带 caveat 分享；更新后的目标不再等待 6.21 晚上，是否完成由 final completion audit 的证据门裁决。
