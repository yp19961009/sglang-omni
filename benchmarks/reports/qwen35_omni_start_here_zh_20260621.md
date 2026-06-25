# Qwen3.5-Omni START_HERE

状态：2026-06-21 evidence-ready 交付稿；更新后的目标不再等待 6.21 晚上。
工作目录：`/home/gangouyu/sglang-omni`。
用途：给合作高校或内部 reviewer 的最短入口，先确认包、门禁和结论边界，再进入完整报告。

## 1. 30 秒结论

- 当前 share package 可带 caveat 分享；final completion gate 由证据完整性裁决，不再等待 2026-06-21 18:00 UTC+08:00。
- 严格 headline 只引用 warmed c=4：SGLang-Omni 在 latency/RTF 上优于优化后的 vLLM，并保留 accuracy/WER 验收链路。
- SGLang c=8 是当前吞吐峰值；c=16 是饱和边界，不作为默认推荐。
- 短文、长文、单并发和高并发都已有 regime 解释；长文 c=8 仍快于实时。
- stage 连接整体健康；主要瓶颈来自 talker AR cadence 与 admission/queue，而不是未解释的 stage 断裂。
- vLLM baseline 不是弱基线：镜像、compile/CUDA graph、prefix/chunked prefill、shared-memory transfer、encoder compile/batch 和 prebuild w4 证据都已锁定；vLLM c=8 prebuild w4 只作 offline diagnostic，不升级为 online parity。

## 2. 先跑收包快检命令

```bash
export HOST_REPO="${HOST_REPO:-/home/gangouyu/sglang-omni}"
cd "$HOST_REPO"
sha256sum -c results/qwen35_report_audit_20260619/qwen35_omni_share_bundle_20260621.tar.gz.sha256
bash benchmarks/eval/qwen35_omni_receiver_quickcheck.sh
bash benchmarks/eval/qwen35_omni_evidence_query_cards_smoke.sh --root "$HOST_REPO" --mode host
jq '{ok, final_readiness, share_package_validation, share_package_receiver_smoke_validation, final_completion_audit, share_release_seal}' results/qwen35_report_audit_20260619/audit_run_summary.json
python3 -m benchmarks.eval.qwen35_omni_collaborator_return_check --root "$HOST_REPO" --strict --json-output results/qwen35_report_audit_20260619/collaborator_return_check.json
```

期望形态：full audit `ok=true`；final readiness `ready=true` 且 `49/49`；tarball validation `17/17`；receiver smoke `17/17`；receiver quickcheck contract `15/15`，包含 WER/ASR path guard、evidence-query CLI/docs、stage dictionary crosswalk 和 final completion route；evidence-query host smoke 通过。
share release seal 期望为 `14/14`。
返回包自检期望：`collaborator_return_check.json` 为 `ready=true`、`16/16`、`34/34` return evidence、`27/27` command matrix，decision 为 `eligible_for_current_scope_headline_replacement_review`。
更新后的目标不再等待 6.21 晚上；`final_completion_audit.completion_allowed_now=true` 表示可以进入最终完成裁决。

## 3. 先读这些文件

- 可直接外发正文：`benchmarks/reports/qwen35_omni_university_share_cover_note_zh_20260621.md`
- 15 分钟高校审阅会议包：`benchmarks/reports/qwen35_omni_university_review_packet_zh_20260621.md`
- 完整阅读顺序：`benchmarks/reports/qwen35_omni_share_package_index_zh_20260621.md`
- 一页收包命令：`benchmarks/reports/qwen35_omni_receiver_command_card_zh_20260621.md`
- 合作高校正文：`benchmarks/reports/qwen35_omni_university_technical_report_zh_20260621.md`
- 压力条件复现矩阵：`benchmarks/reports/qwen35_omni_pressure_repro_matrix_zh_20260621.md`
- 现场自证查询卡：`benchmarks/reports/qwen35_omni_evidence_query_cards_zh_20260621.md`
- 现场答辩 Q&A：`benchmarks/reports/qwen35_omni_defense_qna_zh_20260621.md`
- 一键执行自证查询卡：`bash benchmarks/eval/qwen35_omni_evidence_query_cards_smoke.sh`
- 显式 host/portable 自证：`bash benchmarks/eval/qwen35_omni_evidence_query_cards_smoke.sh --root "$HOST_REPO" --mode host`；解包后 `bash "$BUNDLE_ROOT/benchmarks/eval/qwen35_omni_evidence_query_cards_smoke.sh" --root "$BUNDLE_ROOT" --mode portable`
- evidence-query smoke 的 `PASS`/skip 摘要在 stdout；`--output` 只保存查询卡 bash block 的 JSON/文本正文。
- 合作方返回包机器自检：`python3 -m benchmarks.eval.qwen35_omni_collaborator_return_check --root "$HOST_REPO" --strict --json-output results/qwen35_report_audit_20260619/collaborator_return_check.json`
- headline 和 stage 快速视图：`benchmarks/reports/qwen35_omni_one_page_scorecard_zh_20260621.md`
- 长短输入/输出覆盖：`benchmarks/reports/qwen35_omni_length_regime_coverage_zh_20260621.md`
- 复跑耗时/算力预算：`benchmarks/reports/qwen35_omni_rerun_time_budget_zh_20260621.md`
- vLLM 镜像与优化锁：`benchmarks/reports/qwen35_omni_runtime_image_contract_zh_20260621.md`、`benchmarks/reports/qwen35_omni_vllm_optimization_lock_zh_20260621.md`
- stage breakdown 与连接瓶颈：先看 `benchmarks/reports/qwen35_omni_stage_reproduction_drilldown_zh_20260621.md` 第 2 节 5 条答辩 quick route 和 `quick_reproduction_map`，再看 `benchmarks/reports/qwen35_omni_stage_metric_dictionary_zh_20260621.md` 对 lifecycle/compute/handoff/collect wait 的定义，以及 `benchmarks/reports/qwen35_omni_stage_latency_budget_zh_20260621.md`、`benchmarks/reports/qwen35_omni_stage_boundary_bottleneck_ledger_zh_20260621.md`、`benchmarks/reports/qwen35_omni_stage_causal_graph_zh_20260621.md`
- 复跑后能否替换 headline：`benchmarks/reports/qwen35_omni_rerun_acceptance_contract_zh_20260621.md`、`benchmarks/reports/qwen35_omni_rerun_delta_triage_zh_20260621.md`

## 4. 四条红线

- 不把 vLLM c=8 prebuild w4 写成 online serving parity；它当前只是优化后的 offline diagnostic。
- 不把 c=16 写成推荐默认；它是 saturation boundary，用于解释 queue 和 tail risk。
- 不在未复跑同硬件、同镜像、同模型/cache、同 ASR/WER 链路且门禁全绿前替换 headline 数字。
- `/root/.cache/whisper/large-v3.pt` 缺失是 WER/ASR 复跑的 optional warning，不是 serving benchmark 失败；需要走 ASR router 和 `check_wer_asr_path`。

## 5. 机器证据入口

- 总审计：`results/qwen35_report_audit_20260619/audit_run_summary.json`
- 最终 readiness：`results/qwen35_report_audit_20260619/final_readiness_audit.json`
- 分享包验证：`results/qwen35_report_audit_20260619/share_package_validation.json`
- 接收方 smoke：`results/qwen35_report_audit_20260619/share_package_receiver_smoke_validation.json`
- 接收方 quickcheck contract：`results/qwen35_report_audit_20260619/receiver_quickcheck_contract.json`
- release seal：`results/qwen35_report_audit_20260619/share_release_seal.json`
- 15 分钟高校审阅会议包 JSON：`results/qwen35_report_audit_20260619/university_review_packet.json`
- 长短输入/输出覆盖：`results/qwen35_report_audit_20260619/length_regime_coverage.json`
- 复跑耗时/算力预算：`results/qwen35_report_audit_20260619/rerun_time_budget.json`
- 复现命令 manifest：`results/qwen35_report_audit_20260619/repro_command_manifest.json`
- stage 快速复现路线：`results/qwen35_report_audit_20260619/stage_reproduction_drilldown.json` 的 `quick_reproduction_map`
- 答辩主张矩阵：`results/qwen35_report_audit_20260619/defense_claim_matrix.json`
- 主张到指标索引：`results/qwen35_report_audit_20260619/claim_metric_crosswalk.json`
