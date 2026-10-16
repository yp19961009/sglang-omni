# Qwen3.5-Omni 接收方 Quickcheck Contract

生成时间 UTC：`2026-06-21T02:00:33.448128+00:00`。
工作目录：`/home/gangouyu/sglang-omni`。

这份 contract 用来证明接收方第一条命令没有漂移：
`qwen35_omni_receiver_quickcheck.sh` 必须同时覆盖 checksum、tarball-mode validation、
receiver-smoke validation、extracted-only validation 和 external standalone validation。
命令卡还必须提供 `qwen35_omni_evidence_query_cards_smoke.sh`，用于只读验证
`qwen35_omni_evidence_query_cards_zh_20260621.md` 里的查询入口仍能命中关键证据；
也必须把 `qwen35_omni_stage_metric_dictionary_zh_20260621.md` 放进 stage
读法入口，避免 reviewer 混淆 lifecycle、compute、handoff 和 collect wait。
stage metric dictionary 还必须保留 evidence/rerun crosswalk，把指标口径连到
`metric_provenance_index.json`、`stage_reproduction_drilldown.json`、
`quick_reproduction_map` 和 `rerun_command_ids`。
命令卡还必须能从 `repro_command_manifest.json` 按 command id 直接抽出
`sglang_videoamme_stress` 和 `vllm_c8_prebuild_w4` 等复跑命令。

## 1. Gate

| Gate | Value |
| --- | ---: |
| Ready | True |
| Checks | 15/15 |
| Required failures | 0 |
| Warnings | 0 |
| Receiver JSONs | 4 |
| Public docs | 8 |
| Completion-gate docs | 6 |
| WER/ASR docs | 3 |
| Evidence smoke CLI options | 9 |
| Evidence smoke explicit docs | 6 |

## 2. Receiver Evidence

| Label | File | Ready | Checks | Required failures |
| --- | --- | ---: | ---: | ---: |
| tarball | `results/qwen35_report_audit_20260619/share_package_validation.json` | True | 17/17 | 0 |
| receiver_smoke | `results/qwen35_report_audit_20260619/share_package_receiver_smoke_validation.json` | True | 17/17 | 0 |
| extracted | `results/qwen35_report_audit_20260619/share_package_validation_extracted.json` | True | 13/13 | 0 |
| standalone | `results/qwen35_report_audit_20260619/share_package_external_standalone_validation.json` | True | 8/8 | 0 |

## 3. Contract Checks

| Status | Required | Check | Evidence |
| --- | --- | --- | --- |
| PASS | yes | quickcheck script has ordered six-step receiver flow | missing_steps=[] |
| PASS | yes | quickcheck script writes all receiver evidence JSONs | missing=[] |
| PASS | yes | quickcheck script keeps destructive temp cleanup bounded | missing=[] |
| PASS | yes | current receiver evidence JSONs are all green |  |
| PASS | yes | current receiver evidence is asset-quality clean | quality_failures=[] |
| PASS | yes | public receiver docs route to the quickcheck contract | missing={} |
| PASS | yes | receiver command card routes to evidence-query smoke, stage dictionary, and command-id lookup | route_missing=[]; cli_missing=[]; explicit_doc_missing={} |
| PASS | yes | stage metric dictionary carries evidence and rerun crosswalk | missing=[] |
| PASS | yes | public receiver docs route to the final completion gate | missing={} |
| PASS | yes | public receiver docs include quickcheck failure triage | missing={} |
| PASS | yes | public receiver docs preserve WER/ASR rerun path | checked_docs=['benchmarks/reports/qwen35_omni_receiver_command_card_zh_20260621.md', 'benchmarks/reports/qwen35_omni_reproduction_checklist_zh_20260621.md', 'benchmarks/reports/qwen35_omni_collaborator_rerun_validation_sheet_zh_20260621.md']; guard_needles=['/root/.cache/whisper/large-v3.pt', 'ASR router', 'ASR/WER 路径', 'WER/ASR 快查', 'WER/ASR 结果不可复现', 'asr_router_or_container_cache_required', 'check_wer_asr_path', 'claims_verification.json', 'compute_audio_consistency_from_results', 'headline_scorecard.json', 'optional', 'optional warning', 'throughput 峰值不是通过牺牲语音一致性得到的', 'whisper_large_v3_local_wer.json', '不替换吞吐结论', '同 ASR/WER 口径', '同一 ASR/WER 路径', '复现 SGLang WER', '替换 headline']; missing={} |
| PASS | yes | public receiver docs explain audit-summary self-reference recovery | missing={} |
| PASS | yes | repro command manifest binds quickcheck to all receiver evidence files | command_id=validate_share_bundle_receiver_smoke, evidence_after_run=['results/qwen35_report_audit_20260619/share_package_validation.json', 'results/qwen35_report_audit_20260619/share_package_receiver_smoke_validation.json', 'results/qwen35_report_audit_20260619/share_package_validation_extracted.json', 'results/qwen35_report_audit_20260619/share_package_external_standalone_validation.json'] |
| PASS | no | share bundle manifest includes quickcheck tool and contract evidence | quickcheck_present=True, evidence_smoke_present=True, contract_report_present=True, contract_json_present=True |
| PASS | no | tarball contains quickcheck tool and contract evidence | missing=[], error=None |

## 4. 复现入口

```bash
HOST_REPO="${HOST_REPO:-/home/gangouyu/sglang-omni}"
cd "$HOST_REPO"
bash benchmarks/eval/qwen35_omni_receiver_quickcheck.sh
bash benchmarks/eval/qwen35_omni_evidence_query_cards_smoke.sh --root "$HOST_REPO" --mode host
BUNDLE_ROOT="${BUNDLE_ROOT:-/tmp/qwen35_omni_share_bundle_final/qwen35_omni_share_bundle_20260621}"
bash "$BUNDLE_ROOT/benchmarks/eval/qwen35_omni_evidence_query_cards_smoke.sh" --root "$BUNDLE_ROOT" --mode portable
jq '.summary' results/qwen35_report_audit_20260619/receiver_quickcheck_contract.json
```

通过标准：quickcheck summary 显示 tarball `17/17`、receiver smoke `17/17`、
extracted-only `13/13`、standalone `8/8`；contract JSON 显示 `ready=true`、
`required_failures=0`；evidence-query smoke 显示 host 或 portable 查询卡全部通过。
最终 goal completion 另看相邻 `final_completion_audit.json` 和
`qwen35_omni_final_completion_audit_zh_20260621.md`；更新后的目标不再等待 6.21 晚上，
`completion_allowed_now=true` 表示可以进入最终完成裁决。

解包态如果因为 tarball 自引用哈希看到 `audit_run_summary.json` 为 `in_progress=true`，
只能在公开接收方文档解释 `recovered_from_in_progress_gates=True` 和
`direct_rerun_delta_triage` 约束时接受：`rows_total>=19`、`checks_passed>=8`、
`required_failures=0`。仓库根最终 summary 必须是完成态 `ok=true`，并在顶层暴露
`rerun_delta_triage`。

## 5. 快检失败分流

| 失败点 | 先看证据 | 裁决 |
| --- | --- | --- |
| checksum FAIL | `.tar.gz.sha256` 和 `sha256sum -c` 输出 | 不进入报告阅读；先重传 tarball/checksum。 |
| tarball-mode validation FAIL | `results/qwen35_report_audit_20260619/share_package_validation.json` | 不发送为最终包；修复缺失/mismatch 后重建 tarball。 |
| receiver smoke FAIL | `results/qwen35_report_audit_20260619/share_package_receiver_smoke_validation.json` | 不替换任何 benchmark 数字；先确认安全解包和 nested extracted-only gate。 |
| extracted-only validation FAIL | `results/qwen35_report_audit_20260619/share_package_validation_extracted.json` | 说明解包目录或随包文件不自洽；回到 `BUNDLE_ROOT`/`HOST_REPO` 路径手册排查。 |
| standalone validation FAIL | `results/qwen35_report_audit_20260619/share_package_external_standalone_validation.json` | 说明随包 validator 不能在干净 `/tmp` 独立运行；先修包，不进入外部复现。 |
| quality offenders 非空 | 对应 validation JSON 里的 `report_quality_offenders` / `chart_quality_offenders` | 先修 Markdown/CSV/SVG 质量，再重跑 full audit 和 release seal。 |
