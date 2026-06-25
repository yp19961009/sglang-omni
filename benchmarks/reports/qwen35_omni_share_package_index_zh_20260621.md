# Qwen3.5-Omni 性能分析分享包索引

状态：2026-06-21 evidence-ready 交付稿；更新后的目标不再等待 6.21 晚上。
工作目录：`/home/gangouyu/sglang-omni`。
目标：给合作高校复核 SGLang-Omni Qwen3.5 性能、stage breakdown、vLLM 对比和复现路径。
需要一份可直接转发的中文正文时，先读
`/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_university_technical_report_zh_20260621.md`。
接收方如果先拿到 tarball 或解包目录，先读路径手册：
`/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_receiver_package_path_map_zh_20260621.md`。
需要一页式复制命令时，直接读接收方命令卡：
`/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_receiver_command_card_zh_20260621.md`。
需要机器确认一键收包入口没有漂移时，读接收方 quickcheck contract：
`/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_receiver_quickcheck_contract_zh_20260621.md`。
第一次打开分享包时，最短入口是：
`benchmarks/reports/qwen35_omni_start_here_zh_20260621.md`。
需要直接粘贴邮件/微信正文时，读：
`benchmarks/reports/qwen35_omni_university_share_cover_note_zh_20260621.md`。
需要 15 分钟同步会或合作高校首次审阅的一页半路线图时，读：
`benchmarks/reports/qwen35_omni_university_review_packet_zh_20260621.md`。
需要按压力条件快速查复跑命令和替换边界时，读：
`benchmarks/reports/qwen35_omni_pressure_repro_matrix_zh_20260621.md`。
需要先估算合作方复跑排期、计时段下界和 8-GPU 等效预算时，读：
`benchmarks/reports/qwen35_omni_rerun_time_budget_zh_20260621.md`。
需要现场用 `jq` 自证 headline 和 gate 时，读：
`benchmarks/reports/qwen35_omni_evidence_query_cards_zh_20260621.md`。
要把这些自证查询一次性跑完，执行：
`bash benchmarks/eval/qwen35_omni_evidence_query_cards_smoke.sh`。
host 仓库态可显式运行
`bash benchmarks/eval/qwen35_omni_evidence_query_cards_smoke.sh --root "$HOST_REPO" --mode host`；
解包 bundle 态可运行
`bash "$BUNDLE_ROOT/benchmarks/eval/qwen35_omni_evidence_query_cards_smoke.sh" --root "$BUNDLE_ROOT" --mode portable`。
这条 smoke 的 `PASS`/skip 摘要在 stdout；`--output` 只保存查询卡 bash block 的 JSON/文本正文。
合作方复跑完成并返回完整仓库证据后，先运行：
`python3 -m benchmarks.eval.qwen35_omni_collaborator_return_check --root "$HOST_REPO" --strict --json-output results/qwen35_report_audit_20260619/collaborator_return_check.json`。

## 0. 路径说明

本文中 `/home/gangouyu/sglang-omni` 是当前证据包的生成路径。合作方如果把仓库挂载到
其他位置，先设置：

```bash
export HOST_REPO="${HOST_REPO:-/home/gangouyu/sglang-omni}"
export SMOKE_DIR="${SMOKE_DIR:-/tmp/qwen35_omni_receiver_smoke_index}"
export EXTRACT_DIR="${EXTRACT_DIR:-/tmp/qwen35_omni_share_bundle_index}"
export STANDALONE_DIR="${STANDALONE_DIR:-/tmp/qwen35_omni_external_standalone_bundle_validation_index}"
```

随后 host 侧收包、full audit、图表和报告再生成命令都在 `$HOST_REPO` 下执行；
解包目录只用于阅读和 extracted-only validation；`STANDALONE_DIR` 只用于干净
`/tmp` standalone validation。
解包后的相对路径、`HOST_REPO` / `BUNDLE_ROOT` / container cwd 的区别，以及哪些操作不能在
解包目录直接做，见 `benchmarks/reports/qwen35_omni_receiver_package_path_map_zh_20260621.md`。

## 1. 五分钟阅读顺序

1. 先读最终状态摘要，一页确认当前 gate、tarball、send decision 和必须带上的 caveat：
   `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_final_status_summary_zh_20260621.md`
   如果是合作高校首次审阅或 15 分钟同步会，旁边同时打开：
   `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_university_review_packet_zh_20260621.md`
2. 接收方需要直接复制命令时，先读一页式命令卡：
   `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_receiver_command_card_zh_20260621.md`
3. 需要机器检查 quickcheck 入口是否仍覆盖 tarball、receiver-smoke、extracted-only 和 standalone validation 时读：
   `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_receiver_quickcheck_contract_zh_20260621.md`
   机器证据：`/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/receiver_quickcheck_contract.json`
4. 再读最终分享交付说明，确认要发送哪些文件、当前 gate 和必须带上的 caveat：
   `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_final_share_delivery_note_zh_20260621.md`
5. 需要裁决哪些 caveat 可以分享、哪些说法禁止、哪些条件触发补跑时读 Caveat 裁决矩阵：
   `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_caveat_adjudication_matrix_zh_20260621.md`
6. 需要一页扫完 headline 数字、压力条件、stage 结论时读 one-page scorecard：
   `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_one_page_scorecard_zh_20260621.md`
7. 需要确认 SGLang/vLLM 哪些 runtime/regime 可以公平比较时读对比合同：
   `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_runtime_comparison_contract_zh_20260621.md`
8. 需要确认两边 Docker image、digest、优化开关和可声明边界是否锁定时读 runtime image contract：
   `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_runtime_image_contract_zh_20260621.md`
9. 需要判断合作方复跑结果能否替换 headline 数字时读 rerun acceptance contract：
   `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_rerun_acceptance_contract_zh_20260621.md`
10. 需要确认 SGLang 当前 best recipe、推荐窗口和反例实验是否锁定时读 SGLang 优化锁定矩阵：
   `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_sglang_optimization_lock_zh_20260621.md`
11. 需要确认 vLLM baseline 镜像和优化开关是否锁定时读 vLLM 优化锁定矩阵：
   `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_vllm_optimization_lock_zh_20260621.md`
   需要跨 SGLang/vLLM 复核当前 measured-best recipe、anti-recipe 和 vLLM c=8 diagnostic 裁决时读优化候选 ledger：
   `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_optimization_candidate_ledger_zh_20260621.md`
   机器证据：`/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/optimization_candidate_ledger.json`
12. 需要判断 vLLM c=8 何时能从 offline diagnostic 升级为 strict online parity 时读升级协议：
   `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_vllm_online_parity_protocol_zh_20260621.md`
13. 需要确认 2026-06-21 晚间 final checkpoint 前哪些 caveat/补跑触发器仍需维护时读 final checkpoint watchlist：
   `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_final_checkpoint_watchlist_zh_20260621.md`
14. 需要用 latency 占比回答每个 stage 对端到端延迟的压力尺度时读 Stage latency budget：
   `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_stage_latency_budget_zh_20260621.md`
15. 需要逐 boundary 复核“是不是瓶颈、证据是什么、能否外推”时读 Stage boundary bottleneck ledger：
   `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_stage_boundary_bottleneck_ledger_zh_20260621.md`
16. 需要回答 stage 之间的因果关系、瓶颈转移、连接是否健康，以及要一跳追到
   manifest-backed 原始证据 Drilldown 时读 Stage 因果图：
   `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_stage_causal_graph_zh_20260621.md`
   需要一页横看单并发/高并发、短/长文本和 vLLM original/prebuild 的 stage 热点时读 Pressure × Stage Heatmap：
   `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_pressure_stage_heatmap_zh_20260621.md`
   需要把 Video-AMME ci-50 目标文本范围、synthetic short/long 固定输入形状、long c=8 RTF 和 handoff/decode guard 串在一页时读：
   `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_length_regime_coverage_zh_20260621.md`
   机器证据：`/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/length_regime_coverage.json`
17. 需要按 regime 判断推荐/边界/反例和下一步动作时读决策矩阵：
   `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_regime_decision_matrix_zh_20260621.md`
   需要把 c=1/2/4/8/16、synthetic short/long c=8 和 vLLM c=8 prebuild w4 映射成 serving/capacity 运行选择时读：
   `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_serving_capacity_matrix_zh_20260621.md`
   需要确认公开文档没有旧 gate 数字、旧命令数或旧 tarball identity snapshot 时读 share consistency guard：
   `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_share_consistency_guard_zh_20260621.md`
18. 再读中文简报，拿到结论和对外表述：
   `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_collaboration_brief_zh_20260621.md`
19. 需要确认原始要求逐项有证据时读 evidence map：
   `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_requirement_evidence_map_zh_20260621.md`
20. 需要总览每种压力条件的结论、瓶颈和证据路径时读 pressure matrix：
   `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_pressure_matrix_zh_20260621.md`
21. 需要追问某个数字来自哪个 JSON/artifact 时读 metric source map：
   `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_metric_source_map_zh_20260621.md`
22. 需要解释 lifecycle、compute、handoff、collect wait 等 stage 指标语义时读 stage metric dictionary：
   `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_stage_metric_dictionary_zh_20260621.md`
23. 需要准备现场问答时读 defense Q&A：
   `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_defense_qna_zh_20260621.md`
24. 需要继续调优或说明“当前最优 recipe”时读 optimization playbook：
   `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_optimization_playbook_zh_20260621.md`
25. 需要做 PPT、15 分钟讲稿节奏或现场追问证据跳转时读 deck 提纲：
   `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_share_deck_outline_zh_20260621.md`
   需要确认每页插哪张 SVG/CSV、哪些页只放 JSON/Markdown 证据时读 slide asset map：
   `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_slide_asset_map_zh_20260621.md`
   需要直接拿图表和表格时用 SVG/CSV 图表包：
   `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/share_charts/`
26. 需要完整证据、所有表格和 caveat 时读主报告：
   `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_stress_performance_plan_20260621.md`
27. 需要给外部 reviewer 最短可执行复现路径时读 handoff runbook：
   `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_external_handoff_runbook_zh_20260621.md`
28. 合作方复跑完成后需要逐项验收时读 rerun validation sheet：
   `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_collaborator_rerun_validation_sheet_zh_20260621.md`
   需要机器判断返回材料是否可进入 headline 替换评审时，运行
   `python3 -m benchmarks.eval.qwen35_omni_collaborator_return_check --root "$HOST_REPO" --strict --json-output results/qwen35_report_audit_20260619/collaborator_return_check.json`
29. 合作方复跑数字和当前 checkpoint 不一致，需要先判断是环境、admission、talker、
   handoff、code2wav、WER 还是 vLLM offline parity 边界时读 rerun delta triage：
   `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_rerun_delta_triage_zh_20260621.md`
30. 需要重跑或交叉检查每条命令时读复现清单：
   `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_reproduction_checklist_zh_20260621.md`
   复跑前需要估算计时段下界、8-GPU 等效预算和 WER/ASR 未计时边界时读：
   `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_rerun_time_budget_zh_20260621.md`
   机器证据：`/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/rerun_time_budget.json`
31. 需要机器证据时看审计摘要：
   `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/audit_run_summary.json`
   仓库根中的最终审计摘要必须是完成态 `ok=true`，并在顶层暴露
   `rerun_delta_triage`。解包目录里的包内副本可能因 tarball 自引用哈希顺序显示
   `in_progress=true`；只有 extracted-only/standalone validation 同时证明包内
   `direct_rerun_delta_triage` 为 `ready=true`、`rows_total>=19`、`checks_passed>=8`、
   `required_failures=0` 时，这个自举状态才可接受。
32. 如果要直接发送一个轻量包，可以使用便捷 tarball 和 checksum：
   `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/qwen35_omni_share_bundle_20260621.tar.gz`
   接收方先按 handoff runbook 的收包快检运行 `qwen35_omni_receiver_quickcheck.sh`，
   确认 tarball validation `17/17`、receiver smoke `17/17`、extracted-only
   validation `13/13`、standalone validation `8/8` 后进入报告阅读；同时确认 `report_quality_offenders=[]`、
   `chart_quality_offenders=[]`，再确认 final completion audit 和 release seal 均 `ready=true`，并确认 final readiness 中
   `public_doc_quality_guard=no bare hashes / malformed tables / malformed tokens / duplicate headings / semantic count drift`。
33. 需要确认本轮报告相关代码/工具链 smoke 回归时读单元测试 smoke 证据：
   `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_unit_test_smoke_zh_20260621.md`
   机器证据：`/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/unit_test_smoke.json`。
   该证据只说明 focused container unit smoke 为 `235 passed`，不替代性能 benchmark 或 full audit。

## 2. 当前可对外讲的结论

- 在 8x H20、Video-AMME ci-50 语音输出 workload 上，优化后的
  SGLang-Omni Qwen3.5 在 warmed c=4 严格对比中优于优化版 vLLM：
  latency mean、latency p95、RTF mean、RTF p95 均更低，accuracy/WER 不退化。
- SGLang-Omni 主压测覆盖 c=1/2/4/8/16，c=8 是当前 recipe 的吞吐峰值。
- 短/长文本输入 + 语音输出均有覆盖：短文本 74 chars / 12 words，长文本
  944 chars / 139 words；long synthetic c=8 仍快于实时。
- Stage 连接不是当前主瓶颈：`talker_ar -> code2wav` hop 健康，
  `code2wav_decode` 不是 compute bottleneck。
- 高并发瓶颈主要是 c=8/c=16 admission/queueing 和 talker AR tail；
  naive preprocessing 并发会退化或 OOM。
- vLLM c=8 offline runner 原始路径主要受 host prompt build/feed admission 限制；
  `--prebuild-prompts --prebuild-workers 4` 是最强 offline 诊断，但不是 online serving parity。

## 3. 不应过度声明的边界

- 不把官方 SeedTTS full-set 当作 headline；本地只提供 Video-AMME spoken-reference
  的 SeedTTS-compatible smoke path。
- ci-50/stress/synthetic 证据不能直接外推到完整线上流量；更大 Video-AMME 或真实线上流量
  需要同口径复跑、stage/tail/quality gate 和 caveat gate 全绿。
- 不把 vLLM c=8 prebuild w4 当作严格 online serving benchmark；它是 offline
  diagnostic，需要 online ingress 加 WER/ASR 后才能做严格 c=8 serving parity 声明。
- 不推荐 c=16 作为当前默认运行点；它是压力边界，用来证明 admission 饱和。
- 不声称 code2wav 是瓶颈；当前证据支持优先优化 talker AR 和 admission 策略。

## 4. 机器证据入口

| 证据 | 路径 | 用途 |
| --- | --- | --- |
| university technical report | `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_university_technical_report_zh_20260621.md` | 可直接发给合作高校的中文技术报告正文，覆盖 headline、单/高并发、短/长文本、stage breakdown、vLLM 边界和复现入口 |
| university review packet | `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_university_review_packet_zh_20260621.md` | 合作高校首次审阅或 15 分钟同步会入口，压缩结论、红线、stage 追问路线、复跑预算和替换边界 |
| university review packet JSON | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/university_review_packet.json` | 审阅会议包的结构化 gate，确认 14/14 checks、术语口径路线、15 分钟路线、headline scope、vLLM c=8 边界和复跑预算下界 |
| university technical report JSON | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/university_technical_report.json` | 中文技术报告的结构化 gate，确认 11 个章节和 16 个证据检查通过 |
| serving capacity matrix | `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_serving_capacity_matrix_zh_20260621.md` | 把压力点映射成 latency-first、balanced serving、c=8 throughput edge、c=16 saturation boundary、synthetic realtime guard 和 vLLM offline diagnostic 的运行决策 |
| serving capacity matrix JSON | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/serving_capacity_matrix.json` | Serving/capacity 矩阵的 7 行结构化决策和 10/10 机器 gate |
| share consistency guard | `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_share_consistency_guard_zh_20260621.md` | 发送前检查公开报告和关键 JSON 没有旧 gate 数字、旧命令数、旧 tarball identity snapshot 或 serving/capacity 路由漂移 |
| share consistency guard JSON | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/share_consistency_guard.json` | share consistency guard 的 17/17 机器 gate；要求 stale public/machine hits、university-review gate route、embedded identity leaks、manifest expected-gate unexpected fields、preflight alias mismatches、evidence-query host/portable route missing、current identity-field mismatches 均为 0；完整 tarball hash 链由相邻 release seal 校验 |
| final status summary | `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_final_status_summary_zh_20260621.md` | 一页汇总 full audit、objective、readiness、manifest、tarball 和 caveat |
| caveat adjudication matrix | `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_caveat_adjudication_matrix_zh_20260621.md` | caveat 可分享边界、禁止说法、补跑升级条件和替换数字触发器 |
| regime decision matrix | `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_regime_decision_matrix_zh_20260621.md` | 每个 pressure regime 的推荐状态、瓶颈、caveat、证据和下一步动作 |
| regime decision matrix JSON | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/regime_decision_matrix.json` | 17 个 regime 的结构化推荐状态、checks、SGLang/vLLM/负优化覆盖和 source evidence |
| runtime comparison contract | `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_runtime_comparison_contract_zh_20260621.md` | SGLang/vLLM 哪些 runtime/regime 可公平比较、哪些只能诊断、哪些不能说成 parity |
| runtime comparison contract JSON | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/runtime_comparison_contract.json` | 公平对比合同的 9/9 机器 gate；锁定 warmed c=4 headline、vLLM c=8 diagnostic 边界和优化 baseline 证据 |
| runtime image contract JSON | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/runtime_image_contract.json` | SGLang/vLLM 镜像、digest、GPU 合同和 strict/diagnostic scope 的机器 gate |
| rerun acceptance contract | `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_rerun_acceptance_contract_zh_20260621.md` | 合作方复跑阈值、替换 headline 数字的硬条件和默认裁决 |
| rerun delta triage | `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_rerun_delta_triage_zh_20260621.md` | 复跑数字偏移时，从症状映射到 stage/boundary、证据路径、裁决边界和下一步动作 |
| SGLang optimization lock | `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_sglang_optimization_lock_zh_20260621.md` | SGLang 镜像、compiled/graph recipe、c=8 峰值、stage handoff 和 anti-recipe 边界锁定 |
| vLLM optimization lock | `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_vllm_optimization_lock_zh_20260621.md` | vLLM 镜像、compile/CUDA graph/cache/prebuild 开关和 offline diagnostic 边界锁定 |
| optimization candidate ledger | `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_optimization_candidate_ledger_zh_20260621.md` | 当前 measured-best recipe、anti-recipe、vLLM baseline/diagnostic 和 code2wav-first 优先级裁决 |
| vLLM online parity protocol | `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_vllm_online_parity_protocol_zh_20260621.md` | vLLM c=8 从 offline diagnostic 升级为 strict online parity 前的必需 artifact 和替换 gate |
| stage boundary bottleneck ledger | `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_stage_boundary_bottleneck_ledger_zh_20260621.md` | 每个 stage boundary 的瓶颈判定、证据数字、优化动作和 claim scope |
| stage causal graph | `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_stage_causal_graph_zh_20260621.md` | manifest-backed 原始证据 Drilldown、manifest 证据清单、admission/queue、talker cadence、stream hop、code2wav collect/decode 和 vLLM offline admission 的因果关系 |
| stage causal graph JSON | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/stage_causal_graph.json` | Stage 因果图的 7/7 机器 gate、7 条 causal edge、5 行 raw drilldown 和 manifest-backed raw artifact 检查 |
| pressure stage heatmap | `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_pressure_stage_heatmap_zh_20260621.md` | 单并发/高并发、短/长文本语音和 vLLM original/prebuild 的一页 stage 热点、连接判断和 caveat 总览 |
| pressure stage heatmap JSON | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/pressure_stage_heatmap.json` | Heatmap 的 15 行结构化证据和 11/11 机器 gate；由 `build_stage_route_decision_matrix` 伴随生成 |
| stage reproduction drilldown | `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_stage_reproduction_drilldown_zh_20260621.md` | 先看第 2 节 5 条答辩 quick route；再按每个 stage row 下钻 jq 查询、metric provenance row、raw artifact 和 rerun command ID |
| stage route decision matrix | `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_stage_route_decision_matrix_zh_20260621.md` | 11 条 route 的瓶颈裁决、优化动作、安全说法和复核入口 |
| audit summary | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/audit_run_summary.json` | 一眼确认全套审计是否通过 |
| environment snapshot | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/environment_snapshot.json` | GPU、Docker image、git、模型/数据路径 |
| manifest | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/manifest.json` | 文件清单、大小、SHA-256、dirty worktree |
| claims verifier | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/claims_verification.json` | 17 条核心性能 claim |
| coverage matrix | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/coverage_matrix.json` | 用户目标覆盖情况 |
| headline scorecard | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/headline_scorecard.json` | PPT/分享用核心数字和 9/9 验收布尔值 |
| tail confidence appendix | `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_tail_confidence_appendix_zh_20260621.md` | strict c4、SGLang 压测、短/长 synthetic 和 vLLM diagnostic 的 Markdown p50/p90/p95/max/IQR、JSON p99 与 bootstrap tail 证据 |
| tail confidence appendix JSON | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/tail_confidence_appendix.json` | tail appendix 的 18 行分布统计、9 行 bootstrap 比较、13/13 gate、strict c4 tail 对比和 c8/c16 饱和判断 |
| slide asset map | `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_slide_asset_map_zh_20260621.md` | deck section 到 SVG/CSV/JSON 证据的映射，防止手工挑图或手改数字 |
| slide asset map JSON | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/slide_asset_map.json` | slide asset map 的机器 gate、图表存在性和 chart manifest 覆盖证据 |
| share chart pack manifest | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/share_charts/chart_pack_manifest.json` | SVG/CSV 图表包清单、来源和 ready gate |
| chart source consistency | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/chart_source_consistency.json` | 校验分享图表与审计 JSON 重生成结果 byte-exact 一致 |
| share_charts/ | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/share_charts/` | 可直接放 PPT 的 SVG 图和可复核的 CSV 表 |
| acceptance matrix | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/acceptance_matrix.json` | 单并发/高并发/短长语音/vLLM 诊断/反例逐项验收 |
| confidence ledger | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/confidence_ledger.json` | 高置信 claim、中置信边界、unsupported claim 防线 |
| caveat adjudication matrix JSON | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/caveat_adjudication_matrix.json` | 可说/不可说、升级条件、替换数字触发器和 active-goal 边界的机器 gate |
| objective completion audit | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/objective_completion_audit.json` | 原始目标逐项完成度、caveat 和 active-goal 状态 |
| objective requirement crosswalk | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/objective_requirement_crosswalk.json` | 原始用户要求到 objective rows、defense claims、metric row ids、raw artifacts、复跑命令和 optimization candidate ledger 的机器索引 |
| reproduction command manifest | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/repro_command_manifest.json` | full audit、SGLang、vLLM、表格、图表、preflight、coverage、manifest 的机器可读复跑命令 |
| rerun time budget | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/rerun_time_budget.json` | 合作方复跑排期预算、timed benchmark lower bound、8-GPU 等效 GPU 小时、command IDs 和 WER/ASR 未计时边界 |
| command reference hygiene | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/command_reference_hygiene.json` | 结构化 rerun command IDs 到 repro manifest 的全局解析检查，并确认关键 SGLang/vLLM 命令在公开报告中可见 |
| defense claim matrix | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/defense_claim_matrix.json` | 10 条答辩主张到证据文件、复跑命令和失败裁决的机器可读矩阵，并含 13 个 Q&A 问题到 claim 的 `qna_question_rows` 映射 |
| claim metric crosswalk | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/claim_metric_crosswalk.json` | 对外 defense claim 到 metric provenance rows、raw artifacts 和 rerun command IDs 的机器索引 |
| final readiness audit | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/final_readiness_audit.json` | 发送前总门禁、required failure 数和 send/no-send 决策 |
| share bundle manifest | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/share_bundle_manifest.json` | 推荐发送报告、机器证据和 SVG/CSV 图表资产的 hash 清单 |
| share path hygiene | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/share_path_hygiene.json` | `build_share_path_hygiene` 生成；分享报告中的 package/raw 路径引用、生成型 output dir 和 legacy gate token 的机器检查；要求 package/raw offenders 和 legacy hits 均为 0 |
| metric provenance index | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/metric_provenance_index.json` | headline、pressure、stage、vLLM diagnostic 和 acceptance 指标到 raw artifact / packaged evidence / command ref 的机器索引 |
| table summary | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/tables_summary.json` | 主报告表格来源 |
| stage interaction summary | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/stage_interaction_summary.json` | stage 连接、queue/admission、vLLM prompt-feed 机器证据 |
| pressure stage heatmap JSON | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/pressure_stage_heatmap.json` | 15 行 pressure × stage 机器总览，覆盖 SGLang Video-AMME、synthetic short/long 和 vLLM diagnostic |
| stage drilldown index | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/stage_drilldown_index.json` | 按 stage/边界聚合 budget、瓶颈判定、证据文件和复跑命令 |
| stage reproduction drilldown JSON | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/stage_reproduction_drilldown.json` | `quick_reproduction_map` 给 5 条答辩 quick route；完整 rows 给每个 stage row 的 jq 查询、metric provenance row、raw artifact 和 rerun command ID |
| stage route decision matrix JSON | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/stage_route_decision_matrix.json` | 11 条 route 的裁决、优化动作、安全说法、raw artifacts 和 rerun command IDs |
| stage boundary bottleneck ledger JSON | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/stage_boundary_bottleneck_ledger.json` | 37 条 stage boundary 的瓶颈判定、证据和 claim scope |
| vLLM admission diagnosis | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/vllm_admission_diagnosis.json` | vLLM offline c4/c8/prebuild 分类 |
| SGLang optimization lock JSON | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/sglang_optimization_lock.json` | SGLang best recipe 和反例证据 needle 的机器 gate |
| vLLM optimization lock JSON | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/vllm_optimization_lock.json` | vLLM baseline 优化开关和证据 needle 的机器 gate |
| optimization candidate ledger JSON | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/optimization_candidate_ledger.json` | 8 个优化候选、当前 best、anti-recipe 和 vLLM diagnostic 裁决的机器 gate |
| vLLM online parity protocol JSON | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/vllm_online_parity_protocol.json` | vLLM c=8 online parity 升级协议的机器 gate；当前必须保持 `online_parity_proven=false` |
| rerun acceptance contract JSON | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/rerun_acceptance_contract.json` | 18 条复跑阈值和 headline 替换规则的机器 gate |
| rerun delta triage JSON | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/rerun_delta_triage.json` | 复跑差异定位矩阵的机器可读证据 |
| collaborator return check JSON | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/collaborator_return_check.json` | 合作方返回材料是否满足 34/34 evidence、27/27 command matrix、同硬件/镜像、stage/vLLM caveat 和 share package validation 的机器裁决 |
| vLLM log stages | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/vllm_log_stage_summary.json` | vLLM engine-side stage 边界 |
| Video-AMME SeedTTS meta | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/videoamme_seedtts_meta.lst` | 本地 spoken-reference SeedTTS-compatible smoke path 的 50 条 audio_text meta |
| Video-AMME SeedTTS meta summary | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/videoamme_seedtts_meta_summary.json` | SeedTTS-compatible meta 的样本数、target mode、时长和文本长度摘要 |
| convenience tarball | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/qwen35_omni_share_bundle_20260621.tar.gz` | 便捷发送包，内容来自 share bundle manifest |
| tarball checksum | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/qwen35_omni_share_bundle_20260621.tar.gz.sha256` | 接收方用 `sha256sum -c` 校验 |
| internal file hash list | tarball 内 `PACKAGE_FILE_SHA256SUMS.txt` | tarball 内成员；记录每个随包源文件的相对仓库根路径和逐文件 hash，供 tarball-mode validator 直接校验 tar member，并供 extracted-only validator 解包后复核 |
| package manifest | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/share_bundle_package_manifest.json` | tarball 同目录伴随证据；记录文件数、source bytes、tarball hash 和源 manifest hash |
| share package validator | `/home/gangouyu/sglang-omni/benchmarks/eval/validate_qwen35_omni_share_package.py` | 接收方快速校验 tarball、checksum、关键 gate、stage budget 图表和 caveat |
| receiver quickcheck script | `/home/gangouyu/sglang-omni/benchmarks/eval/qwen35_omni_receiver_quickcheck.sh` | 一条 bash 入口串起 checksum、tarball-mode validation、receiver-smoke validation、extracted-only validation 和 external standalone validation |
| receiver quickcheck contract | `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_receiver_quickcheck_contract_zh_20260621.md` | 机器检查一键收包入口是否仍按六步覆盖四类 receiver evidence JSON，并路由到 final completion gate |
| receiver quickcheck contract JSON | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/receiver_quickcheck_contract.json` | contract 的机器可读证据；检查 quickcheck script、公开文档路由、final completion gate 路由、repro command manifest 和分享包成员 |
| share package validation JSON | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/share_package_validation.json` | tarball 同目录伴随证据；validator 的机器可读输出 |
| extracted validation JSON | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/share_package_validation_extracted.json` | 解包后运行 validator 生成的伴随证据 |
| receiver smoke validation JSON | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/share_package_receiver_smoke_validation.json` | tarball 同目录伴随证据；一条命令完成 tarball 校验、安全解包和 extracted-only 校验 |
| external standalone validation JSON | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/share_package_external_standalone_validation.json` | tarball 同目录伴随证据；证明可在干净 `/tmp` 解包根中直接运行随包 validator |
| final completion audit JSON | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/final_completion_audit.json` | tarball 同目录伴随证据；最终确认 full audit、watchlist、release seal 和 completion time gate |
| final completion audit report | `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_final_completion_audit_zh_20260621.md` | tarball 同目录伴随可读最终 completion gate；不是 tarball 内成员 |
| share release seal JSON | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/share_release_seal.json` | tarball 同目录伴随证据；封口确认 tarball、checksum、validation、final readiness 和 caveat |
| share release seal report | `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_share_release_seal_zh_20260621.md` | tarball 同目录伴随可读封口页；不是 tarball 内成员 |

`share_bundle_package_manifest.json`、`share_package_validation.json`、
`share_package_validation_extracted.json`、`share_package_receiver_smoke_validation.json` 和
`share_package_external_standalone_validation.json`，以及
`receiver_quickcheck_contract.json`、`final_completion_audit.json`、`share_release_seal.json` 是
和 tarball 同目录保存或运行生成的伴随验证证据，
不是 tarball 内成员；它们描述 tarball 自身，放入 tarball 会造成自引用 hash。
`qwen35_omni_final_completion_audit_zh_20260621.md` 是给人读的最终 completion gate，也应与 tarball 相邻保存。
`qwen35_omni_share_release_seal_zh_20260621.md` 是给人读的同一封口证据，也应与 tarball 相邻保存。
`PACKAGE_FILE_SHA256SUMS.txt` 是 tarball 内成员，记录每个随包源文件的相对仓库根路径和
逐文件 hash；tarball-mode validator 会先用它直接校验 tar member 内容，接收方解包后
运行 extracted-only validator 时会再次复核报告、证据 JSON、工具脚本和图表资产。
同一个 validator 还会直接扫描随包 `share_report` Markdown 中的裸 hash、坏表格、重复 heading
和坏展示 token，并检查随包 `share_charts` CSV/SVG 可解析、非空且结构可渲染；通过时 evidence 应显示
`report_quality_offenders=[]` 和 `chart_quality_offenders=[]`。

## 5. 几个最重要命令

先做收包快检：

```bash
HOST_REPO="${HOST_REPO:-/home/gangouyu/sglang-omni}"
SMOKE_DIR="${SMOKE_DIR:-/tmp/qwen35_omni_receiver_smoke_index}"
EXTRACT_DIR="${EXTRACT_DIR:-/tmp/qwen35_omni_share_bundle_index}"
STANDALONE_DIR="${STANDALONE_DIR:-/tmp/qwen35_omni_external_standalone_bundle_validation_index}"
cd "$HOST_REPO"
bash benchmarks/eval/qwen35_omni_receiver_quickcheck.sh
```

如果要手工拆开每一步：

```bash
HOST_REPO="${HOST_REPO:-/home/gangouyu/sglang-omni}"
SMOKE_DIR="${SMOKE_DIR:-/tmp/qwen35_omni_receiver_smoke_index}"
EXTRACT_DIR="${EXTRACT_DIR:-/tmp/qwen35_omni_share_bundle_index}"
STANDALONE_DIR="${STANDALONE_DIR:-/tmp/qwen35_omni_external_standalone_bundle_validation_index}"
cd "$HOST_REPO"

sha256sum -c results/qwen35_report_audit_20260619/qwen35_omni_share_bundle_20260621.tar.gz.sha256
python3 -m benchmarks.eval.validate_qwen35_omni_share_package \
  --root "$HOST_REPO" \
  --strict \
  --json-output results/qwen35_report_audit_20260619/share_package_validation.json

python3 -m benchmarks.eval.validate_qwen35_omni_share_package \
  --root "$HOST_REPO" \
  --strict \
  --receiver-smoke-dir "$SMOKE_DIR" \
  --json-output results/qwen35_report_audit_20260619/share_package_receiver_smoke_validation.json

rm -rf "$EXTRACT_DIR"
mkdir -p "$EXTRACT_DIR"
tar -xzf results/qwen35_report_audit_20260619/qwen35_omni_share_bundle_20260621.tar.gz \
  -C "$EXTRACT_DIR"
cd "$EXTRACT_DIR/qwen35_omni_share_bundle_20260621"
python3 benchmarks/eval/validate_qwen35_omni_share_package.py \
  --root "$PWD" \
  --extracted-only \
  --strict \
  --json-output "$HOST_REPO/results/qwen35_report_audit_20260619/share_package_validation_extracted.json"
cd "$HOST_REPO"

# command_id: validate_external_standalone_share_bundle
python3 -m benchmarks.eval.build_qwen35_omni_external_standalone_bundle_validation \
  --root "$HOST_REPO" \
  --strict \
  --work-dir "$STANDALONE_DIR" \
  --json-output results/qwen35_report_audit_20260619/share_package_external_standalone_validation.json

# command_id: build_share_release_seal
python3 -m benchmarks.eval.build_qwen35_omni_share_release_seal \
  --root "$HOST_REPO" \
  --strict \
  --output benchmarks/reports/qwen35_omni_share_release_seal_zh_20260621.md \
  --json-output results/qwen35_report_audit_20260619/share_release_seal.json
```

上述 package validation 和 standalone validation 通过时，资产 evidence 还应显示
`report_quality_offenders=[]`、`chart_quality_offenders=[]`。

再跑完整审计：

```bash
HOST_REPO="${HOST_REPO:-/home/gangouyu/sglang-omni}"
cd "$HOST_REPO"

python3 -m benchmarks.eval.run_qwen35_omni_report_audit \
  --root "$HOST_REPO" \
  --summary-output results/qwen35_report_audit_20260619/audit_run_summary.json
```

单独复核 stage interaction：

```bash
python3 -m benchmarks.eval.summarize_qwen35_stage_interactions \
  --root /home/gangouyu/sglang-omni \
  --json-output results/qwen35_report_audit_20260619/stage_interaction_summary.json
```

单独复核 headline scorecard：

```bash
python3 -m benchmarks.eval.build_qwen35_omni_headline_scorecard \
  --root /home/gangouyu/sglang-omni \
  --json-output results/qwen35_report_audit_20260619/headline_scorecard.json
```

单独重建分享用 SVG/CSV 图表包：

```bash
python3 -m benchmarks.eval.build_qwen35_omni_share_charts \
  --root /home/gangouyu/sglang-omni \
  --output-dir results/qwen35_report_audit_20260619/share_charts \
  --manifest-output results/qwen35_report_audit_20260619/share_charts/chart_pack_manifest.json

python3 -m benchmarks.eval.build_qwen35_omni_chart_source_consistency \
  --root /home/gangouyu/sglang-omni \
  --strict \
  --output benchmarks/reports/qwen35_omni_chart_source_consistency_zh_20260621.md \
  --json-output results/qwen35_report_audit_20260619/chart_source_consistency.json
```

单独复核 acceptance matrix：

```bash
python3 -m benchmarks.eval.build_qwen35_omni_acceptance_matrix \
  --root /home/gangouyu/sglang-omni \
  --json-output results/qwen35_report_audit_20260619/acceptance_matrix.json
```

单独复核 confidence ledger：

```bash
python3 -m benchmarks.eval.build_qwen35_omni_confidence_ledger \
  --root /home/gangouyu/sglang-omni \
  --json-output results/qwen35_report_audit_20260619/confidence_ledger.json
```

单独复核原始目标完成度：

```bash
python3 -m benchmarks.eval.build_qwen35_omni_objective_completion_audit \
  --root /home/gangouyu/sglang-omni \
  --strict \
  --json-output results/qwen35_report_audit_20260619/objective_completion_audit.json
```

单独复核原始需求到指标/证据/命令的 crosswalk：

```bash
python3 -m benchmarks.eval.build_qwen35_omni_objective_requirement_crosswalk \
  --root /home/gangouyu/sglang-omni \
  --strict \
  --json-output results/qwen35_report_audit_20260619/objective_requirement_crosswalk.json
```

单独复核机器可读复现命令清单：

```bash
python3 -m benchmarks.eval.build_qwen35_omni_repro_command_manifest \
  --root /home/gangouyu/sglang-omni \
  --strict \
  --json-output results/qwen35_report_audit_20260619/repro_command_manifest.json
```

单独复核最终分享 readiness：

```bash
python3 -m benchmarks.eval.build_qwen35_omni_final_readiness \
  --root /home/gangouyu/sglang-omni \
  --strict \
  --json-output results/qwen35_report_audit_20260619/final_readiness_audit.json
```

单独重建最终状态摘要：

```bash
python3 -m benchmarks.eval.build_qwen35_omni_final_status_summary \
  --root /home/gangouyu/sglang-omni \
  --output benchmarks/reports/qwen35_omni_final_status_summary_zh_20260621.md
```

单独重建 reviewer 决策矩阵：

```bash
python3 -m benchmarks.eval.build_qwen35_omni_regime_decision_matrix \
  --root /home/gangouyu/sglang-omni \
  --strict \
  --output benchmarks/reports/qwen35_omni_regime_decision_matrix_zh_20260621.md \
  --json-output results/qwen35_report_audit_20260619/regime_decision_matrix.json
```

单独重建 runtime 公平对比合同：

```bash
python3 -m benchmarks.eval.build_qwen35_omni_runtime_comparison_contract \
  --root /home/gangouyu/sglang-omni \
  --output benchmarks/reports/qwen35_omni_runtime_comparison_contract_zh_20260621.md \
  --json-output results/qwen35_report_audit_20260619/runtime_comparison_contract.json \
  --strict
```

单独重建 SGLang 优化锁定矩阵：

```bash
python3 -m benchmarks.eval.build_qwen35_omni_sglang_optimization_lock \
  --root /home/gangouyu/sglang-omni \
  --strict \
  --output benchmarks/reports/qwen35_omni_sglang_optimization_lock_zh_20260621.md \
  --json-output results/qwen35_report_audit_20260619/sglang_optimization_lock.json
```

单独重建 vLLM 优化锁定矩阵：

```bash
python3 -m benchmarks.eval.build_qwen35_omni_vllm_optimization_lock \
  --root /home/gangouyu/sglang-omni \
  --strict \
  --output benchmarks/reports/qwen35_omni_vllm_optimization_lock_zh_20260621.md \
  --json-output results/qwen35_report_audit_20260619/vllm_optimization_lock.json
```

单独重建优化候选裁决 ledger：

```bash
python3 -m benchmarks.eval.build_qwen35_omni_optimization_candidate_ledger \
  --root /home/gangouyu/sglang-omni \
  --strict \
  --output benchmarks/reports/qwen35_omni_optimization_candidate_ledger_zh_20260621.md \
  --json-output results/qwen35_report_audit_20260619/optimization_candidate_ledger.json
```

单独重建 vLLM online parity 升级协议：

```bash
python3 -m benchmarks.eval.build_qwen35_omni_vllm_online_parity_protocol \
  --root /home/gangouyu/sglang-omni \
  --strict \
  --output benchmarks/reports/qwen35_omni_vllm_online_parity_protocol_zh_20260621.md \
  --json-output results/qwen35_report_audit_20260619/vllm_online_parity_protocol.json
```

单独重建 final checkpoint watchlist：

```bash
python3 -m benchmarks.eval.build_qwen35_omni_final_checkpoint_watchlist \
  --root /home/gangouyu/sglang-omni \
  --strict \
  --output benchmarks/reports/qwen35_omni_final_checkpoint_watchlist_zh_20260621.md \
  --json-output results/qwen35_report_audit_20260619/final_checkpoint_watchlist.json
```

单独重建 Stage latency budget：

```bash
python3 -m benchmarks.eval.build_qwen35_omni_stage_latency_budget \
  --root /home/gangouyu/sglang-omni \
  --strict \
  --output benchmarks/reports/qwen35_omni_stage_latency_budget_zh_20260621.md \
  --json-output results/qwen35_report_audit_20260619/stage_latency_budget.json
```

单独重建 Stage boundary bottleneck ledger：

```bash
python3 -m benchmarks.eval.build_qwen35_omni_stage_boundary_bottleneck_ledger \
  --root /home/gangouyu/sglang-omni \
  --strict \
  --output benchmarks/reports/qwen35_omni_stage_boundary_bottleneck_ledger_zh_20260621.md \
  --json-output results/qwen35_report_audit_20260619/stage_boundary_bottleneck_ledger.json
```

单独重建 Stage 因果图：

```bash
python3 -m benchmarks.eval.build_qwen35_omni_stage_causal_graph \
  --root /home/gangouyu/sglang-omni \
  --output benchmarks/reports/qwen35_omni_stage_causal_graph_zh_20260621.md \
  --json-output results/qwen35_report_audit_20260619/stage_causal_graph.json \
  --strict
```

单独重建 Caveat 裁决矩阵：

```bash
python3 -m benchmarks.eval.build_qwen35_omni_caveat_adjudication_matrix \
  --root /home/gangouyu/sglang-omni \
  --strict \
  --output benchmarks/reports/qwen35_omni_caveat_adjudication_matrix_zh_20260621.md \
  --json-output results/qwen35_report_audit_20260619/caveat_adjudication_matrix.json
```

单独复核分享包文件 hash 清单：

```bash
python3 -m benchmarks.eval.build_qwen35_omni_share_bundle_manifest \
  --root /home/gangouyu/sglang-omni \
  --strict \
  --json-output results/qwen35_report_audit_20260619/share_bundle_manifest.json
```

单独重建便捷发送 tarball：

```bash
python3 -m benchmarks.eval.build_qwen35_omni_share_bundle_package \
  --root /home/gangouyu/sglang-omni \
  --strict \
  --source-manifest results/qwen35_report_audit_20260619/share_bundle_manifest.json \
  --output results/qwen35_report_audit_20260619/qwen35_omni_share_bundle_20260621.tar.gz \
  --json-output results/qwen35_report_audit_20260619/share_bundle_package_manifest.json
```

校验便捷发送包：

```bash
cd /home/gangouyu/sglang-omni
sha256sum -c results/qwen35_report_audit_20260619/qwen35_omni_share_bundle_20260621.tar.gz.sha256
tar -tzf results/qwen35_report_audit_20260619/qwen35_omni_share_bundle_20260621.tar.gz | head
```

复跑 vLLM c=8 prebuild w4 offline 诊断：

```bash
RUN_ROOT="/home/gangouyu/sglang-omni/results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuildw4_20260620_$(date +%H%M%S)" \
MAX_SAMPLES=50 MAX_CONCURRENCY=8 MAX_NUM_SEQS=8 \
RUN_TAG=ci50_offline_compile_c8_mns8_prebuildw4_20260620 \
EXTRA_ARGS="--prebuild-prompts --prebuild-workers 4" \
bash results/qwen35_vllm_videoamme_ci50_opt_20260618_162319/run_vllm_videoamme_ci5_offline_compile.sh
```

## 6. 答辩问题索引

| 问题 | 推荐查看 |
| --- | --- |
| 现场问答应该怎么回答？ | defense Q&A |
| 当前最优 recipe 和下一步调优怎么做？ | optimization playbook；完整报告 section 12/14 |
| 每种压力条件的推荐/不推荐状态是什么？ | pressure matrix；acceptance matrix |
| 某个 headline 或 stage 数字从哪里来？ | metric source map；headline scorecard；tables summary |
| stage lifecycle、actual compute、handoff 应该怎么读？ | stage metric dictionary；stage interaction summary |
| SGLang 是否至少和 vLLM 相当？ | 完整报告 section 1/5；claims verifier |
| 原始需求是否逐项覆盖？ | requirement evidence map；coverage matrix |
| vLLM baseline 是否足够优化？ | 完整报告 section 4.1/5；中文简报 section 4.2 |
| 为什么 c=8 是推荐高并发点？ | 完整报告 section 7/12；coverage matrix |
| 为什么 c=16 不推荐？ | 完整报告 section 7/12/13；stage interaction summary |
| 每个 pressure case 是否验收通过？ | acceptance matrix |
| 每个 pressure case 到底该推荐、边界还是回滚？ | regime decision matrix；pressure matrix；acceptance matrix |
| 哪些 runtime/regime 可以公平比较？ | runtime comparison contract；headline scorecard；confidence ledger |
| stage 之间的因果关系/瓶颈转移怎么看？ | stage causal graph 的 manifest-backed 原始证据 Drilldown；stage route decision matrix；stage reproduction drilldown；stage interaction summary；stage metric dictionary |
| caveat 到底能不能对外说、什么时候必须补跑？ | caveat adjudication matrix；confidence ledger；final readiness audit |
| 哪些话能高置信对外说？ | confidence ledger；中文简报 section 8 |
| stage 之间是不是卡住了？ | 完整报告 section 7.1/8.1/12.1；stage interaction summary |
| code2wav 是不是瓶颈？ | 完整报告 section 7.1/8.1/12.1；claims verifier |
| 长语音是否稳定？ | 完整报告 section 8；tables summary |
| WER/语音一致性有没有退化？ | 完整报告 section 7；claims verifier |
| vLLM c=8 为什么不能直接作为 online parity？ | 完整报告 section 5/12；vLLM admission diagnosis |
| 一页核心数字看哪里？ | one-page scorecard；headline scorecard；metric source map；metric provenance index |
| PPT 图表和可复核 CSV 在哪里？ | share_charts/；chart pack manifest；deck outline |
| PPT 图表有没有被手工改数字？ | chart source consistency；slide asset map；chart pack manifest |
| 分享/PPT 应该怎么排？ | deck outline |
| 最终发送给合作方时应该附什么说明？ | final share delivery note；share package index |
| 怎么把报告交给外部 reviewer 复现？ | external handoff runbook；复现清单；repro command manifest；full audit pipeline |
| reviewer 复跑后怎么判定是否能替换数字？ | rerun validation sheet；acceptance matrix；confidence ledger |

## 7. 当前接受门槛

- full audit `ok=true`
- claims `17/17`
- coverage `34/34`
- preflight `62` checks, `0` required failures
- manifest current `196` records, minimum `180`, `0` missing
- university technical report `ready=true`, checks `16/16`
- headline scorecard `ready=true`, checks `9/9`
- metric provenance index `ready=true`, headline/pressure/stage/vLLM metrics have raw artifact and command links
- claim metric crosswalk `ready=true`, 10 external defense claims map to concrete metric rows and rerun commands
- defense claim matrix `ready=true`, 13 Q&A question rows map live questions to the 10 external defense claims
- objective requirement crosswalk `ready=true`, 11 original requirement rows map to objective rows, claims, metric rows, raw artifacts, rerun commands, and 8 optimization candidate verdicts
- acceptance matrix `ready=true`, rows `17/17`
- confidence ledger `ready=true`, entries `12/12`，其中 high `9`、medium `3`、unsupported `0`
- SGLang optimization lock `ready=true`, checks `26/26`
- vLLM optimization lock `ready=true`, checks `22/22`
- vLLM online parity protocol `ready=true`, checks `18/18`, `online_parity_proven=false`
- runtime image contract `ready=true`, checks `12/12`
- rerun acceptance contract `ready=true`, checks `17/17`, rules `18`, return evidence files `34`, command return evidence rows `27`, matrix complete, silent-replacement/protocol-drift guard documented
- final checkpoint watchlist `ready=true`, checks `24/24`, `final_completion_evidence_ready=true`, `checkpoint_phase=completion_audit_ready`, `completion_allowed_now=true`
- stage latency budget `ready=true`, checks `12/12`
- stage boundary bottleneck ledger `ready=true`, checks `12/12`, rows `37`, pressure transition rows `11`
- length regime coverage `ready=true`, rows `7`, synthetic short/long rows `6`, long c=8 RTF p95 `<1`, handoff/decode guard healthy
- stage causal graph JSON `ready=true`, checks `7/7`, edges `7`, raw drilldown `5`
- repro command manifest `ready=true`, `63` commands / `7` phases
- final completion audit `ready=true`, required failures `0`, `completion_allowed_now=true`
- share release seal `ready=true`
- command reference hygiene `ready=true`, structured rerun command IDs resolve and critical SGLang/vLLM commands are documented
- objective completion audit `share_ready_with_documented_caveats=true`, rows `17`, required failures `0`
- final readiness audit `ready=true`, `49/49` checks, `0` required failures
- receiver quickcheck contract `ready=true`, checks `15/15`, `6` quickcheck steps, `4` receiver JSONs, completion-gate docs `6`, WER/ASR docs `3`, stage dictionary crosswalk needles `8`, `0` required failures
- external standalone validation `ready=true`, checks `8/8`
- share bundle manifest `ready=true`
- 便捷 tarball 的 `.sha256` 校验通过
- stage interaction summary 中四个关键布尔结论为 `true`

若上述任一门槛不满足，不应把该 evidence package 作为最终分享版本。
