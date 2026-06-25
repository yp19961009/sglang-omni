# Qwen3.5-Omni 最终 Checkpoint Watchlist

生成时间 UTC：`2026-06-21T02:00:33.672459+00:00`。
工作目录：`/home/gangouyu/sglang-omni`。

这份 watchlist 用来约束最终交付前的证据门：
哪些结论可以分享，哪些只能作为 caveat，哪些新证据会触发补跑或替换结论。

## 1. 当前状态

| Gate | Value |
| --- | ---: |
| Ready | True |
| Checks | 24/24 |
| Required failures | 0 |
| Watch items | 7 |
| Checkpoint | `evidence-ready completion audit` |
| Current time local | `2026-06-21T10:00:33.672198+08:00` |
| Checkpoint start local | `2026-06-21T18:00:00+08:00` |
| Checkpoint phase | `completion_audit_ready` |
| Seconds until checkpoint | 0 |
| Share ready with caveats | True |
| Final completion evidence ready | True |
| Goal complete | False |
| Completion allowed now | True |
| Completion blockers | `none` |
| Current decision | `run_final_completion_audit_now` |

## 2. Watch Items

| Item | Current State | Trigger | Action |
| --- | --- | --- | --- |
| long-running goal active | 更新后的目标已取消 2026-06-21 晚间等待；当前只按证据完整性、复现性和 caveat gate 裁决是否完成。 | 任何报告、JSON、chart、命令 manifest、tarball 或接收方验证证据发生变化。 | 重跑 full audit、preflight、manifest、final readiness、share bundle、tarball checksum、receiver smoke、extracted-only validation、standalone validation、release seal 和 stale scan；逐项检查更新后的目标后才可标记 complete。 |
| vLLM c=8 online parity | 当前只有 optimized offline diagnostic；online_parity_proven=false。 | 出现 online ingress、同口径 WER/ASR、stage boundary profile 和 c=8 完整结果。 | 先通过 vLLM online parity protocol，再更新 runtime contract、headline、scorecard 和 caveat matrix。 |
| SGLang high-concurrency boundary | c=4-c=8 是推荐窗口；c=16 是 saturation boundary。 | c=16 复跑延迟/RTF/queue 明显改善，且 quality 不退化。 | 只在 stage interaction、acceptance matrix 和 objective audit 都通过后提升推荐窗口。 |
| SeedTTS full-set boundary | 本包只使用 local SeedTTS-compatible smoke path，不把 official full-set 当 headline。 | official full-set 数据、命令、WER/ASR 和结果全部补齐。 | 新增独立章节和 machine evidence；否则继续按 caveat 分享。 |
| optional Whisper host cache | host 侧 large-v3 cache warning 是 optional，不阻塞当前 serving evidence。 | 合作方要求 host 侧直接重算 WER。 | 提供 cache、容器内 ASR 路径或明确 ASR router；不要把 optional warning 写成 required failure。 |
| stage metric interpretation | stage lifecycle、compute span、handoff、collect wait 已分开定义。 | 任何新报告把 lifecycle 当 compute 或把 collect wait 当 code2wav compute。 | 先修 stage metric dictionary、stage causal graph 和 pressure matrix，再发新包。 |
| artifact freshness | 当前 tarball 由 share_bundle_manifest 生成，带 SHA-256。 | 任何 report、JSON、chart、command manifest 或 audit summary 发生变化。 | 重跑 share bundle package、sha256sum -c、receiver smoke、extracted-only validation 和 release seal；仓库根目录固定为 /home/gangouyu/sglang-omni。 |

## 3. Machine Checks

| Status | Required | Check | Evidence |
| --- | --- | --- | --- |
| PASS | yes | updated objective removes the evening time gate | current_time_local=2026-06-21T10:00:33.672198+08:00, legacy_checkpoint_start_local=2026-06-21T18:00:00+08:00, checkpoint_phase=completion_audit_ready, seconds_until_checkpoint=0, updated_objective=no_6_21_evening_wait |
| PASS | yes | full audit remains green | audit_summary_signal=green_during_full_audit_refresh, ok_so_far=True, steps_seen=161; final audit_run_summary is rewritten at audit exit; recovered_from_current_gates=True |
| PASS | yes | share-ready objective, goal still active | objective={'share_ready_with_documented_caveats': True, 'goal_complete': False, 'rows_total': 17, 'required_failures': 0, 'boundary_items': 3, 'preflight_pending_in_full_audit': False, 'status_counts': {'PASS': 14, 'PASS_WITH_CAVEAT': 3}, 'send_decision': 'ready_to_share_with_documented_caveats'} |
| PASS | yes | final readiness contract remains current | ready=True, checks=49/49, required_failures=0, hard_gates=55, final_checkpoint_gate=24/24, audit_in_progress=True |
| PASS | yes | original objective final-completion evidence is assembled | objective={'share_ready_with_documented_caveats': True, 'goal_complete': False, 'rows_total': 17, 'required_failures': 0, 'boundary_items': 3, 'preflight_pending_in_full_audit': False, 'status_counts': {'PASS': 14, 'PASS_WITH_CAVEAT': 3}, 'send_decision': 'ready_to_share_with_documented_caveats'}; objective_requirement_crosswalk={'ready': True, 'requirement_rows_total': 11, 'required_rows_total': 10, 'pass_rows_total': 6, 'pass_with_caveat_rows_total': 5, 'claim_refs_total': 10, 'metric_row_refs_total': 232, 'unique_metric_rows_total': 114, 'raw_artifacts_total': 36, 'packaged_evidence_files_total': 59, 'command_refs_total': 40, 'categories_total': 11, 'optimization_candidate_ledger_ready': True, 'optimization_candidate_rows_total': 8, 'optimization_rejected_anti_recipes_total': 2, 'optimization_vllm_diagnostic_rows_total': 2, 'optimization_current_best_candidate_id': 'sglang_current_best_measured_recipe', 'optimization_not_global_optimum_boundary': True, 'checks_total': 20, 'checks_passed': 20, 'required_failures': 0, 'goal_complete': False, 'completion_allowed_now': True, 'share_scope': 'Maps each original user objective to objective-completion rows, defense claims, metric provenance row IDs, evidence files, and reproduction command IDs, plus an optimization candidate ledger.'}; final_readiness=ready=True, checks=49/49, receiver_quickcheck_contract={'ready': True, 'checks_total': 15, 'checks_passed': 15, 'required_failures': 0, 'warnings': 0, 'tarball_contract_checked': True, 'tarball_contains_contract': True, 'quickcheck_steps': 6, 'receiver_jsons_total': 4, 'public_docs_total': 8, 'completion_gate_docs_total': 6, 'wer_asr_docs_total': 3, 'evidence_smoke_cli_options': 9, 'evidence_smoke_explicit_docs_total': 6, 'stage_dictionary_crosswalk_needles_total': 8}; external_standalone_validation={'ready': True, 'checks_total': 8, 'checks_passed': 8, 'required_failures': 0, 'warnings': 0, 'extracted_validation_ready': True, 'extracted_validation_checks': 13, 'extracted_validation_required_failures': 0, 'repo_independent_invocation': True}; final_readiness_pending_in_full_audit=False; final_readiness_self_reference_pending=False; receiver_contract_pending_in_full_audit=False |
| PASS | yes | preflight remains reproducible | preflight={'total_checks': 62, 'required_failures': 0, 'warnings': 1, 'ready': True}; preflight_pending_in_full_audit=False; preflight_self_reference_pending=False |
| PASS | yes | manifest remains complete | manifest={'total_records': 196, 'missing_records': 0, 'file_records': 194, 'directory_records': 2} |
| PASS | yes | share bundle remains complete | share_bundle={'ready': True, 'records_total': 122, 'missing_required': 0, 'file_records': 121, 'directory_records': 1, 'category_counts': {'share_report': 50, 'machine_evidence': 53, 'share_tool': 4, 'share_chart_directory': 1, 'share_chart': 14}, 'checks': {'all_required_files_present': True, 'full_audit_ok': True, 'evidence_manifest_ready': True, 'chart_pack_ready': True, 'share_tool_ready': True, 'primary_reading_order_present': True}} |
| PASS | yes | share tarball checksum files exist | tarball=/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/qwen35_omni_share_bundle_20260621.tar.gz, checksum=/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/qwen35_omni_share_bundle_20260621.tar.gz.sha256 |
| PASS | yes | share package validation remains green | share_package_validation=ready=True, checks=17/17, required_failures=0, tar_members=124, missing_bundle_members=0, pending_in_full_audit=False |
| PASS | yes | receiver smoke validation remains green | receiver_smoke=ready=True, checks=17/17, required_failures=0, receiver_smoke_ready=True |
| PASS | yes | extracted-only package validation remains green | extracted_validation=ready=True, checks=13/13, required_failures=0, extracted_only=True |
| PASS | yes | share release seal remains green | share_release_seal=ready=True, checks=14/14, required_failures=0, tarball_identity_recorded_in_adjacent_release_seal=True, receiver_smoke_ready=True, goal_complete=False, pending_in_full_audit=False |
| PASS | yes | confidence ledger has no unsupported claims | confidence={'ready': True, 'entries_total': 12, 'entries_passed': 12, 'entries_failed': 0, 'high_confidence_claims': 9, 'medium_confidence_boundaries': 3, 'unsupported_claims': 0, 'confidence_counts': {'high': 9, 'medium': 3}, 'category_counts': {'bounded_medium_confidence_statement': 3, 'safe_high_confidence_claim': 9}} |
| PASS | yes | runtime image contract still pins image and optimization scope | runtime_image_contract={'ready': True, 'checks_total': 12, 'checks_passed': 12, 'required_failures': 0, 'sglang_image': 'frankleeeee/sglang-omni:dev', 'sglang_image_id': 'sha256:be7e72126f525c3767008a73de16f400f974a09db431ded3c52bd48370941a84', 'vllm_image': 'tongyi-duanwu-registry-vpc.cn-beijing.cr.aliyuncs.com/dashscope/dashllm:cuda129_cp312_test_vl_13589', 'vllm_image_id': 'sha256:e71dc281e0882896f81e3471206312b7c31ba4b6ab56ed9def0d4f1392a8c4ba', 'gpu_contract': '8x NVIDIA H20 / CUDA 12.8', 'sglang_scope': 'c4-c8 warmed serving; c8 peak throughput; c16 saturation boundary', 'vllm_strict_scope': 'optimized warmed c4 apples-to-apples headline only', 'vllm_c8_scope': 'prebuild w4 is optimized offline diagnostic, not online parity', 'environment_pending_in_full_audit': False} |
| PASS | yes | rerun acceptance contract still pins replacement rules | rerun_acceptance_contract={'ready': True, 'checks_total': 17, 'checks_passed': 17, 'required_failures': 0, 'rules_total': 18, 'sglang_stress_rules': 5, 'synthetic_rules': 6, 'vllm_rules': 4, 'return_evidence_files': 34, 'return_evidence_missing_current': 0, 'return_evidence_missing_sheet': 0, 'return_evidence_command_rows': 27, 'return_evidence_command_missing': 0, 'return_evidence_command_file_gaps': 0, 'replacement_scope': 'same hardware/image/model/data plus all gates green', 'default_decision': 'confirm_shape_unless_environment_and_all_gates_match'} |
| PASS | yes | SGLang optimization lock still pins the recipe | sglang_lock={'ready': True, 'checks_total': 26, 'checks_passed': 26, 'required_failures': 0, 'sglang_image': 'frankleeeee/sglang-omni:dev', 'sglang_image_id': 'sha256:be7e72126f525c3767008a73de16f400f974a09db431ded3c52bd48370941a84', 'recommended_window': 'c4-c8 warmed serving; c8 peak throughput; c16 saturation boundary', 'recipe_contract': 'compiled/graph SGLang recipe with serial preprocessing and 16GiB preprocessing cache'} |
| PASS | yes | vLLM optimization lock still pins the baseline | vllm_lock={'ready': True, 'checks_total': 22, 'checks_passed': 22, 'required_failures': 0, 'vllm_image': 'tongyi-duanwu-registry-vpc.cn-beijing.cr.aliyuncs.com/dashscope/dashllm:cuda129_cp312_test_vl_13589', 'vllm_image_id': 'sha256:e71dc281e0882896f81e3471206312b7c31ba4b6ab56ed9def0d4f1392a8c4ba', 'strict_c4_contract': 'optimized warmed c4 apples-to-apples headline only', 'c8_contract': 'prebuild w4 is optimized offline diagnostic, not online parity'} |
| PASS | yes | vLLM c8 parity is not over-promoted | vllm_online={'ready': True, 'checks_total': 18, 'checks_passed': 18, 'required_failures': 0, 'current_package_safe': True, 'online_parity_proven': False, 'upgrade_decision': 'do_not_promote_c8_parity_without_online_ingress_artifacts', 'required_artifacts_total': 6} |
| PASS | yes | stage handoff and code2wav bottleneck claims remain bounded | stage_interactions={'total_interactions': 37, 'status_counts': {'healthy': 28, 'queue_limited': 2, 'contention_regression': 1, 'prompt_feed_limited': 2, 'bottleneck': 1, 'diagnostic_only': 2, 'watch': 1}, 'sglang_talker_to_code2wav_healthy': True, 'sglang_code2wav_decode_not_bottleneck': True, 'vllm_original_c8_prompt_feed_limited': True, 'preprocessing_parallelism_regresses': True} |
| PASS | yes | final delivery note carries checkpoint caveats | missing= |
| PASS | yes | caveat matrix names rerun triggers | missing= |
| PASS | yes | stage metric dictionary prevents metric drift | missing= |
| PASS | yes | main report keeps final-checkpoint boundary visible | missing= |

## 4. Final Checkpoint Protocol

1. Run the full audit pipeline from the repository root.
2. Rebuild final status, final readiness, share bundle manifest, tarball, and checksum after any generated artifact changes.
3. Run qwen35_omni_receiver_quickcheck.sh and require checksum OK, tarball-mode 17/17, receiver smoke 17/17, extracted-only 13/13, and standalone 8/8.
4. Regenerate the adjacent release seal and require ready=true with 0 required failures.
5. Scan for stale gate counts, stale package hashes, and public bare 64-hex hashes after the final audit refresh.
6. Only mark the thread goal complete after every updated objective requirement is rechecked requirement by requirement and final_completion_audit reports completion_allowed_now=true.

最小收包命令：

```bash
cd /home/gangouyu/sglang-omni
export HOST_REPO="${HOST_REPO:-/home/gangouyu/sglang-omni}"
export SMOKE_DIR="${SMOKE_DIR:-/tmp/qwen35_omni_receiver_smoke_final}"
export EXTRACT_DIR="${EXTRACT_DIR:-/tmp/qwen35_omni_share_bundle_final}"
export STANDALONE_DIR="${STANDALONE_DIR:-/tmp/qwen35_omni_external_standalone_bundle_validation_final}"

python3 -m benchmarks.eval.run_qwen35_omni_report_audit \
  --root /home/gangouyu/sglang-omni \
  --summary-output results/qwen35_report_audit_20260619/audit_run_summary.json

bash benchmarks/eval/qwen35_omni_receiver_quickcheck.sh

python3 -m benchmarks.eval.build_qwen35_omni_share_release_seal \
  --root /home/gangouyu/sglang-omni \
  --strict \
  --output benchmarks/reports/qwen35_omni_share_release_seal_zh_20260621.md \
  --json-output results/qwen35_report_audit_20260619/share_release_seal.json
```

## 5. 不可提前升级的说法

- 不把当前 vLLM c=8 prebuild w4 写成 strict online serving parity。
- 不把 c=16 写成默认推荐服务点。
- 不把 official SeedTTS full-set 写成 headline benchmark。
- 不把 host-side Whisper optional warning 写成 required failure。
- 不在 `completion_allowed_now=true` 前把长线目标标记 complete。
