# Qwen3.5-Omni 外部复跑差异定位矩阵

状态：证据门已就绪；更新后的目标不再等待 6.21 晚间，后续变更必须重跑 full audit。
工作目录：`/home/gangouyu/sglang-omni`。
用途：合作方复跑后，如果 latency、RTF、WER、stage 占比或 vLLM 结果和当前 checkpoint 不一致，先用这张表定位差异，再决定是否能替换 headline 数字。

## 1. 使用原则

1. 先跑收包快检和 full audit；package 或 audit 不绿时，不讨论性能结论。
2. 再判断环境、镜像、模型、数据 cache、ASR/WER 路径是否与当前 checkpoint 一致。
3. 只有同硬件、同镜像、同模型、同数据、同 ASR/WER 路径，并且 rerun acceptance contract 通过，才进入 headline 替换评审。
4. vLLM c=8 prebuild w4 在当前包内仍是 optimized offline diagnostic；除非 online parity protocol 变为 proven，否则不能说成 online serving parity。
5. stage 结论要从 request profile、stage latency budget、boundary ledger 和 causal graph 一起读，不用单个 latency 数字直接归因。
6. 只重跑 raw benchmark 不足以替换数字；必须重跑 regeneration/full audit，并让报告、图表、JSON evidence 和 final readiness 同步全绿。
7. 不能手工只改公开 Markdown 或图表；任何 headline 或 stage 数字替换都必须重新生成 manifest、share bundle、tarball seal 和 receiver quickcheck。

## 2. 当前 gate 摘要

| Gate | 当前摘要 |
| --- | --- |
| triage ready | `True`，rows `19`，checks `8/8` |
| final readiness | `{'ready': True, 'checks_total': 49, 'checks_passed': 49, 'required_failures': 0}` |
| rerun acceptance | `{'ready': True, 'checks_total': 17, 'checks_passed': 17, 'required_failures': 0, 'rules_total': 18, 'sglang_stress_rules': 5, 'synthetic_rules': 6, 'vllm_rules': 4, 'return_evidence_files': 34, 'return_evidence_missing_current': 0, 'return_evidence_missing_sheet': 0, 'return_evidence_command_rows': 27, 'return_evidence_command_missing': 0, 'return_evidence_command_file_gaps': 0, 'replacement_scope': 'same hardware/image/model/data plus all gates green', 'default_decision': 'confirm_shape_unless_environment_and_all_gates_match'}` |
| stage interaction | `{'total_interactions': 37, 'status_counts': {'healthy': 28, 'queue_limited': 2, 'contention_regression': 1, 'prompt_feed_limited': 2, 'bottleneck': 1, 'diagnostic_only': 2, 'watch': 1}, 'sglang_talker_to_code2wav_healthy': True, 'sglang_code2wav_decode_not_bottleneck': True, 'vllm_original_c8_prompt_feed_limited': True, 'preprocessing_parallelism_regresses': True}` |
| vLLM online parity | `{'ready': True, 'checks_total': 18, 'checks_passed': 18, 'required_failures': 0, 'current_package_safe': True, 'online_parity_proven': False, 'upgrade_decision': 'do_not_promote_c8_parity_without_online_ingress_artifacts', 'required_artifacts_total': 6}` |

## 3. 差异定位矩阵

| 复跑症状 | 优先定位 stage / boundary | 第一证据 | 裁决边界 | 下一步动作 | 数字替换范围 |
| --- | --- | --- | --- | --- | --- |
| 收包或 full audit 失败 | package gate / evidence integrity | share_package_validation.json, share_package_receiver_smoke_validation.json, audit_run_summary.json | 红：不要读性能结论，也不要替换 headline 数字。 | 先修 checksum、tar member、extracted-only validation、public_doc_quality_guard；通过后再讨论性能差异。 | 不得替换 |
| 复跑机器、镜像、模型或数据 cache 与当前 checkpoint 不一致 | environment / runtime image boundary | environment_snapshot.json, runtime_image_contract.json, rerun_acceptance_contract.json | 黄：只能确认趋势，不能直接替换当前 8x H20 headline。 | 在 collaborator rerun validation sheet 标注差异；同硬件同镜像重跑后再进入替换评审。 | 附录趋势 |
| SGLang warmed c=4 latency/RTF 不再优于 vLLM warmed c=4 | strict c=4 comparison | headline_scorecard.json, claims_verification.json, runtime_comparison_contract.json | 红或黄：若环境不一致为黄；若同环境且 claims 失败为红，不能保留原 headline。 | 同时复核 SGLang c=4 profile、vLLM c=4 report、WER artifact；只在 rerun acceptance contract 通过后替换。 | 需替换评审 |
| c=8 不再是 SGLang 吞吐峰值 | admission / queueing / running-request cap | tables_summary.json, stage_latency_budget.json, stage_interaction_summary.json | 黄：先作为复跑差异，不直接改推荐点；同环境确认两次后再改 c=8 结论。 | 检查 --thinker-max-running-requests、--talker-max-running-requests、warmup、请求 profile run_id 是否一致。 | 附录趋势 |
| c=16 吞吐没有下降，或 queue/admission 不明显 | high-concurrency saturation boundary | stage_latency_budget.json, stage_boundary_bottleneck_ledger.json, request_profile_c16_profile_skipwer.json | 黄：说明本次 admission 形态不同；不要直接删除 c=16 saturation caveat。 | 确认并发驱动、样本数、warmup、GPU clock 和 profile 是否覆盖完整请求。 | 附录趋势 |
| prefill/queue 占比升高，mean/p95 latency 同时变差 | preprocessor / admission queue | stage_latency_budget.json, stage_boundary_bottleneck_ledger.json, stage_causal_graph report | 黄：先定位为 admission 压力，不归因到 code2wav。 | 检查 video preprocessing cache、输入帧数、max pixels、HTTP client 并发和 profile 中 queue/wait 字段。 | 附录趋势 |
| talker_ar p95 明显拉长，但 code2wav decode 稳定 | talker AR compute tail | stage_interaction_summary.json, stage_boundary_bottleneck_ledger.json, request_profile_*_profile*.json | 绿或黄：若 headline gate 仍过，可保留结论并把差异放在 tail analysis。 | 优先检查 talker torch compile、CUDA graph、max running requests、生成音频时长分布。 | 可进入评审 |
| talker_ar -> code2wav hop wait 升高 | stream handoff boundary | stage_interaction_summary.json, stage_boundary_bottleneck_ledger.json, qwen35_omni_stage_causal_graph_zh_20260621.md | 红：若 handoff health 失败，不要继续声明 stage 连接健康。 | 先复核 request profile 中 collect wait、handoff wait、decode cadence；再判断是否为 talker 输出节奏或 code2wav 消费侧问题。 | 需替换评审 |
| code2wav_decode 成为 latency 主项 | code2wav decode compute | stage_latency_budget.json, stage_boundary_bottleneck_ledger.json, sglang_optimization_lock.json | 红：当前报告的 code2wav-not-bottleneck 结论不再可直接引用。 | 检查 NO_CODE2WAV_TORCH_COMPILE、TORCHDYNAMO_DISABLE、模型权重、decode batch/cadence，并重新生成 stage ledger。 | 需替换评审 |
| long synthetic c=8 RTF 接近或超过 1 | long-form talker AR / audio duration pressure | synthetic_short_long_speech.csv, tables_summary.json, request_profile_long_c8_profile.json | 红：不能继续说 long c=8 快于实时。 | 先确认 long prompt 是 944 chars / 139 words；再复核音频时长、max_tokens、voice 和 serving warmup。 | 需替换评审 |
| 短/长文本输入形状不匹配 | synthetic workload contract | tables_summary.json, synthetic_short_long_speech.csv, qwen35_omni_reproduction_checklist_zh_20260621.md | 红：长短文覆盖证据无效，不能替换相关表格。 | 按 reproduction checklist 重新跑 short 74 chars / 12 words、long 944 chars / 139 words 的固定输入。 | 不得替换 |
| 复跑样本数、skip-first、warmup 或 fixed prompt 口径不一致 | measurement protocol / warmup boundary | repro_command_manifest.json, rerun_acceptance_contract.json, headline_scorecard.json, tables_summary.json | 红或黄：协议口径不一致时不能替换主报告数字；最多作为附录趋势。 | 按 command ID 重新跑，保留 sample_count、skip_first、warmup、fixed short/long prompt 和 WER/ASR 口径；再重跑 full audit。 | 不得替换 |
| 只重跑 benchmark 原始结果，没有重跑 regeneration/full audit | evidence regeneration boundary | audit_run_summary.json, final_readiness_audit.json, rerun_acceptance_contract.json | 红：raw 结果本身不足以替换 headline；必须证明生成表、图、证据索引和门禁同步。 | 运行 run_qwen35_omni_report_audit，确认 claims、acceptance、stage ledger、chart source consistency、final readiness 和 share package validation 全绿。 | 不得替换 |
| 只修改公开 Markdown 或图表，没有同步 JSON/manifest/share bundle | silent replacement / share consistency boundary | share_consistency_guard.json, chart_source_consistency.json, share_bundle_manifest.json, share_release_seal.json | 红：这是 silent-replacement 风险；公开数字不能脱离机器证据、manifest 和 tarball seal。 | 从源 JSON 重生成报告和 chart pack，再重跑 share bundle manifest、tarball package、receiver smoke、extracted-only、standalone validation 和 receiver quickcheck。 | 不得替换 |
| WER/accuracy 退化或 ASR 路径改变 | quality / ASR side channel | claims_verification.json, headline_scorecard.json, rerun_acceptance_contract.json | 红：accuracy/WER 不过 gate 时，不能只替换 latency/RTF headline。 | 离线单独复算 WER，确认 Whisper large-v3 权重或 ASR router 一致；避免与 serving 压测抢 GPU。 | 不得替换 |
| vLLM c=8 original offline 明显变好但 prebuild 证据缺失 | vLLM offline prompt build/feed admission | vllm_admission_diagnosis.json, vllm_log_stage_summary.json, vllm_optimization_lock.json | 黄：仍是 offline runner 形态，不能升级为 online serving parity。 | 补跑 --prebuild-prompts --prebuild-workers 4；再按 online parity protocol 补 online ingress 与 WER/ASR。 | 附录趋势 |
| 有人想用 vLLM c=8 prebuild w4 替换 online parity 结论 | vLLM parity scope | vllm_online_parity_protocol.json, runtime_comparison_contract.json, caveat_adjudication_matrix report | 红：当前 package 明确 online_parity_proven=false。 | 必须先补 online serving artifact、同口径 WER/ASR、ingress latency 和 rerun acceptance contract。 | 不得替换 |
| naive preprocessing 并发看起来提升吞吐 | preprocessing anti-recipe | sglang_optimization_lock.json, stage_interaction_summary.json, qwen35_omni_optimization_playbook_zh_20260621.md | 黄：先作为新候选 recipe，不覆盖当前 preproc=2 回退、preproc=4 OOM 边界。 | 至少补 c=4/c=8/c=16、WER、profile、OOM/稳定性记录，再进入 recipe 替换。 | 附录趋势 |
| 图表或公开 Markdown 被手工改动后 validator 失败 | share asset hygiene | share_package_validation.json, share_bundle_manifest.json, final_readiness_audit.json | 红：不要发送；先修报告资产，再重新打包。 | 重跑 share chart pack、share bundle manifest、tarball package、receiver smoke 和 extracted-only validation。 | 不得替换 |

## 4. 快速裁决模板

| 问题 | 需要填写 | 裁决提示 |
| --- | --- | --- |
| 复跑是否同 8x H20、同镜像、同模型、同数据 cache、同 ASR/WER 路径？ |  | 否则默认只能做附录趋势。 |
| full audit、share package validation、receiver smoke、extracted-only validation 是否全绿？ |  | 任一失败时不得替换数字。 |
| SGLang c=4 strict headline 是否仍优于 vLLM c=4？ |  | 失败时需要重新跑 strict pair 并更新 claims。 |
| c=8 是否仍是 SGLang 主 recipe 的吞吐峰值？ |  | 失败时先查 admission、queue 和 running request cap。 |
| long c=8 是否仍快于实时？ |  | 失败时不能继续引用 long c=8 RTF 结论。 |
| code2wav decode 是否仍不是主瓶颈？ |  | 失败时必须更新 stage latency budget 和 boundary ledger。 |
| vLLM c=8 是否已有 online serving parity artifact？ |  | 当前默认没有；prebuild w4 只能当 offline diagnostic。 |

## 5. 证据入口

- `results/qwen35_report_audit_20260619/rerun_delta_triage.json`：本矩阵的机器可读版本。
- `benchmarks/reports/qwen35_omni_collaborator_rerun_validation_sheet_zh_20260621.md`：collaborator rerun validation sheet，合作方复跑填写表。
- `benchmarks/reports/qwen35_omni_rerun_acceptance_contract_zh_20260621.md`：数字替换阈值合同。
- `benchmarks/reports/qwen35_omni_stage_latency_budget_zh_20260621.md`：stage latency 占比。
- `benchmarks/reports/qwen35_omni_stage_boundary_bottleneck_ledger_zh_20260621.md`：boundary-level bottleneck 账本。
- `benchmarks/reports/qwen35_omni_stage_causal_graph_zh_20260621.md`：stage 因果和 manifest-backed drilldown。
- `benchmarks/reports/qwen35_omni_vllm_online_parity_protocol_zh_20260621.md`：vLLM c=8 online parity 升级条件。
