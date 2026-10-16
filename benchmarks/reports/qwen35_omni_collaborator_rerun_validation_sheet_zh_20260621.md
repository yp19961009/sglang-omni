# Qwen3.5-Omni 合作方复跑验收表

状态：证据门已就绪；更新后的目标不再等待 6.21 晚间，后续变更必须重跑 full audit。
工作目录：`/home/gangouyu/sglang-omni`。
用途：合作高校或外部 reviewer 复跑后，用这张表把环境、命令、artifact、核心指标和
stage 结论逐项对齐，判断是否可以替换或确认当前报告数字。

相关文档：

- 主报告：
  `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_stress_performance_plan_20260621.md`
- 外部复现 handoff runbook：
  `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_external_handoff_runbook_zh_20260621.md`
- 复现清单：
  `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_reproduction_checklist_zh_20260621.md`
- 复跑验收阈值合同：
  `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_rerun_acceptance_contract_zh_20260621.md`
- 复跑耗时/算力预算：
  `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_rerun_time_budget_zh_20260621.md`
- 压力条件总表：
  `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_pressure_matrix_zh_20260621.md`
- Stage 指标字典：
  `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_stage_metric_dictionary_zh_20260621.md`
- 机器审计：
  `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/audit_run_summary.json`
- 返回包机器自检：
  `/home/gangouyu/sglang-omni/benchmarks/eval/qwen35_omni_collaborator_return_check.py`

机器 gate 摘要：

- coverage `34/34`
- preflight `62` checks, `0` required failures
- manifest current `196` records, minimum `180`, `0` missing
- SGLang optimization lock `26/26`
- vLLM optimization lock `22/22`
- vLLM online parity protocol `18/18`，`online_parity_proven=false`
- rerun acceptance contract `ready=true`, `17/17` checks, `18` rules, `34` return evidence files, `27` command matrix rows
- collaborator return check 当前自检 `16/16` checks，`34/34` return evidence files，`27/27` command rows，decision=`eligible_for_current_scope_headline_replacement_review`
- rerun time budget `ready=true`, `9` budget rows, `6` timed rows, `0` required failures
- tail confidence appendix `ready=true`, `18` rows / `13` checks
- length regime coverage `ready=true`, `7` rows / `10/10` checks, short prompt `74` chars, long prompt `944` chars
- repro command manifest `ready=true`, `63` commands / `7` phases
- final readiness audit `ready=true`, `49/49` checks
- share bundle manifest `ready=true`
- vLLM c=8 prebuild w4 仍是 offline diagnostic，不是 online serving parity
- evidence query card smoke 建议在 host/portable 模式通过，用于快速自证核心查询、stage gate 和长短文 coverage；它不改变 `34` 个 return evidence hard contract

填写说明：

- 表格中的空白单元格是给合作方复跑后填写的工作区，不代表当前报告缺失证据。
- “当前 checkpoint / 当前接受门槛 / 当前 checkpoint 形态”列是本报告已经审计通过的基准。
- “复跑环境 / 复跑结果（合作者填写）/ 结果（合作者填写）”列由复跑方填写实际数值、artifact 路径或 PASS/FAIL。
- “判定（PASS/FAIL/附录）”列用于决定该复跑能否确认当前形态、只能作为外部附录，或需要先补证据。

## 0. 三色替换判定

外部 reviewer 先用这一节给复跑结果定性，再填写后面的详细表格。这里的判定只决定是否
进入主报告数字替换流程；不影响把外部复跑作为附录证据保存。

| 判定 | 进入条件 | 处理 |
| --- | --- | --- |
| 绿：可进入替换评审 | 同 8x H20、同 SGLang/vLLM image digest、同模型、同 Video-AMME cache、同 ASR/WER 口径，full audit 和 rerun acceptance contract 全绿 | 可以提交数字替换评审，但仍需同步更新 evidence、chart、主报告和 share bundle |
| 黄：只确认趋势 | full audit 全绿，但硬件、image digest、模型路径、数据 cache 或 ASR 口径任一不同 | 作为外部复核附录；不覆盖 8x H20 headline 数字 |
| 红：不得替换 | claims、acceptance、confidence、stage ledger、runtime image contract、share package validation 任一 required gate 失败，或 vLLM c=8 只有 offline diagnostic | 不替换主报告；先定位差异并补齐机器证据 |

数字替换审批栈：

1. `environment_snapshot.json` 证明硬件、镜像、模型、数据和 ASR 口径一致。
2. `audit_run_summary.json`、`claims_verification.json`、`acceptance_matrix.json`、
   `confidence_ledger.json`、`stage_boundary_bottleneck_ledger.json` 和
   `runtime_comparison_contract.json`、`runtime_image_contract.json`、
   `rerun_acceptance_contract.json`、SGLang/vLLM optimization lock、
   `vllm_online_parity_protocol.json` 和 stage drilldown/route 证据全部 required gate 通过。
3. `tables_summary.json`、`headline_scorecard.json`、chart pack、主报告、压力矩阵、
   stage causal graph、Q&A、share package index 和 final delivery note 同步更新。
4. 重新生成 share bundle，重新校验 tarball checksum、receiver smoke、receiver quickcheck contract、extracted-only validation 和 standalone validation；同时重跑 final completion audit。

如果某次复跑只满足“黄”或“红”，可以把数据写进本表和外部附录，但不要把它写成
“SGLang/vLLM headline 已替换”或“vLLM c=8 online parity 已证明”。

返回包机器自检命令：

```bash
cd /home/gangouyu/sglang-omni

python3 -m benchmarks.eval.qwen35_omni_collaborator_return_check \
  --root /home/gangouyu/sglang-omni \
  --strict \
  --json-output results/qwen35_report_audit_20260619/collaborator_return_check.json
```

判定为 `eligible_for_current_scope_headline_replacement_review` 时，才进入主报告数字替换评审。
如果返回 `appendix_only_environment_or_image_mismatch`，说明 full audit 可以绿，但硬件或镜像边界
不同，只能作为附录趋势；如果返回 evidence/json/command gap，则先补齐返回材料。

silent-replacement 风险：

- 如果只重跑 raw benchmark、没有重跑 regeneration/full audit，不得替换 headline 或 stage 数字。
- 如果只修改公开 Markdown 或图表、没有同步 JSON evidence、manifest、share bundle、tarball seal 和 receiver quickcheck，不得发送。
- 如果 sample count、skip-first、warmup、fixed short/long prompt 或 WER/ASR 口径不同，只能作为附录趋势，不能覆盖当前 checkpoint。

## 1. 复跑结果登记

| 项 | 填写内容（合作者填写） | 是否必须 |
| --- | --- | --- |
| Reviewer / 单位 | 合作者填写 | 是 |
| 复跑日期 | 合作者填写 | 是 |
| 机器位置 / hostname | 合作者填写 | 是 |
| GPU 型号和数量 | 合作者填写 | 是 |
| SGLang image id | 合作者填写 | 是 |
| vLLM image id | 合作者填写 | 是 |
| Qwen3.5-Omni model path | 合作者填写 | 是 |
| Video-AMME cache path | 合作者填写 | 是 |
| 是否修改代码或参数 | 合作者填写 | 是 |
| 新结果 root | 合作者填写 | 是 |
| 是否运行 full audit | 合作者填写 | 是 |

## 2. 环境差异

先记录环境证据：

```bash
cd /home/gangouyu/sglang-omni

python3 -m benchmarks.eval.build_qwen35_omni_environment_snapshot \
  --root /home/gangouyu/sglang-omni \
  --json-output results/qwen35_report_audit_20260619/environment_snapshot.json
```

| 环境项 | 当前 checkpoint | 复跑环境（合作者填写） | 判定（PASS/FAIL/附录） |
| --- | --- | --- | --- |
| GPU | 8x NVIDIA H20 | 合作者填写 | 不同硬件不能直接替换 headline 吞吐 |
| SGLang image | `frankleeeee/sglang-omni:dev` | 合作者填写 | image id 不同需说明 |
| vLLM image | `dashllm:cuda129_cp312_test_vl_13589` | 合作者填写 | image id 不同需说明 |
| 模型路径 | `/myapp/models/qwen3_5_omni_23b_final_multilingual_all_voice_bf16_0315` | 合作者填写 | 必须是同模型 |
| 数据路径 | `/myapp/data/videoamme` | 合作者填写 | 必须是同 ci cache |
| Whisper / ASR | host cache 可选缺失 | 合作者填写 | WER 需可用 ASR |

如果 GPU、image、模型或数据 cache 有变化，复跑结果可以作为外部复核证据，但不要直接覆盖
当前 8x H20 checkpoint 的 headline 数字。

## 3. Full Audit 验收

```bash
cd /home/gangouyu/sglang-omni

python3 -m benchmarks.eval.run_qwen35_omni_report_audit \
  --root /home/gangouyu/sglang-omni \
  --summary-output results/qwen35_report_audit_20260619/audit_run_summary.json
```

| Gate | 当前接受门槛 | 复跑结果（合作者填写） | 判定（PASS/FAIL/附录） |
| --- | --- | --- | --- |
| full audit | `ok=true` | 合作者填写 | 必须 PASS |
| claims | `17/17` | 合作者填写 | 必须 PASS |
| coverage | `34/34` | 合作者填写 | 必须 PASS |
| preflight | `62` checks, `0` required failures | 合作者填写 | 必须 PASS |
| manifest | current `196` records, minimum `180`, `0` missing | 合作者填写 | 必须 PASS |
| headline scorecard | `9/9` | 合作者填写 | 必须 PASS |
| acceptance matrix | `17/17` | 合作者填写 | 必须 PASS |
| confidence ledger | `12/12`, unsupported `0` | 合作者填写 | 必须 PASS |
| SGLang optimization lock | `26/26` | 合作者填写 | 必须 PASS |
| vLLM optimization lock | `22/22` | 合作者填写 | 必须 PASS |
| vLLM online parity protocol | `18/18`, `online_parity_proven=false` | 合作者填写 | 必须 PASS |
| rerun acceptance contract | `17/17`, rules `18`, return evidence `34`, command rows `27` | 合作者填写 | 必须 PASS |
| rerun time budget | `ready=true`, `9` budget rows, `6` timed rows；不包含 server launch/warmup/WER/ASR | 合作者填写 | 必须 PASS |
| tail confidence appendix | `ready=true`, `18` rows / `13` checks | 合作者填写 | 必须 PASS |
| length regime coverage | `ready=true`, `7` rows / `10/10` checks；short `74` chars，long `944` chars，long c=8 RTF p95 < 1 | 合作者填写 | 必须 PASS |
| repro command manifest | `ready=true`, `63` commands / `7` phases | 合作者填写 | 必须 PASS |
| final readiness audit | `ready=true`, `49/49` checks, `0` required failures | 合作者填写 | 必须 PASS |
| share bundle manifest | `ready=true` | 合作者填写 | 必须 PASS |

允许的已知 warning：host 侧 `/root/.cache/whisper/large-v3.pt` 缺失。它是 optional；
只有需要在 host 直接重算 WER 时才必须补齐或改用 ASR router。

## 4. SGLang 复跑验收

主压测必须覆盖 Video-AMME ci-50 c=1/2/4/8/16。

| 项 | 当前 checkpoint 形态 | 复跑结果（合作者填写） | 判定（PASS/FAIL/附录） |
| --- | --- | --- | --- |
| c=1/c=2/c=4 | 低并发和 warmed baseline；tail 主要是 `talker_ar` | 合作者填写 | 形态应一致 |
| c=8 | 当前吞吐峰值，约 `2.540 req/s` | 合作者填写 | 应为峰值或需解释 |
| c=16 | 吞吐低于 c=8，queue/admission 变重 | 合作者填写 | 不应作为推荐默认点 |
| Accuracy | c=1/2/4/8/16 均约 `70.0%` | 合作者填写 | 不应异常下降 |
| WER | warmed stress rows 约 `2.88%-3.85%` | 合作者填写 | 不应靠牺牲语音一致性换吞吐 |
| preproc=2 | 比 baseline c=8 QPS 回退约 `35.4%` | 合作者填写 | 应作为反例 |
| preproc=4 | OOM/失败反例 | 合作者填写 | 应作为反例 |

短/长文本输入 + 语音输出必须覆盖 synthetic speech c=1/4/8；短文本应为
74 chars / 12 words，长文本应为 944 chars / 139 words。
复跑后同时检查 `results/qwen35_report_audit_20260619/length_regime_coverage.json`；
它把 Video-AMME ci-50 目标文本长度、synthetic short/long 固定 prompt、stage handoff/decode
guardrail 和“不能外推到完整线上流量”的红线放在同一张机器表里。

| 项 | 当前 checkpoint 形态 | 复跑结果（合作者填写） | 判定（PASS/FAIL/附录） |
| --- | --- | --- | --- |
| short c=1/4/8 | talker/code2wav isolated short-output path | 合作者填写 | 必须有结果 |
| long c=1/4/8 | long c=8 仍快于实时，RTF 约 `0.4932` | 合作者填写 | long c=8 RTF 应小于 1 |
| long tail | 主要是 talker AR，不是 code2wav decode | 合作者填写 | stage 形态应一致 |

## 5. vLLM 复跑验收

vLLM 严格 headline 仍以 warmed c=4 为 apples-to-apples 对比点。

| 项 | 当前 checkpoint 形态 | 复跑结果（合作者填写） | 判定（PASS/FAIL/附录） |
| --- | --- | --- | --- |
| vLLM c=4 warmed | latency mean/p95、RTF mean/p95 均高于 SGLang warmed c=4 | 合作者填写 | SGLang 应保持优势 |
| vLLM c=4 WER/accuracy | SGLang accuracy 不低于 vLLM，WER 不高于 vLLM | 合作者填写 | 不应退化 |
| vLLM original c=8 | `prompt_feed_limited` | 合作者填写 | 不能当 online parity |
| vLLM c=8 prebuild w4 | 50/50 success，runner wall/QPS 改善 | 合作者填写 | 是 offline diagnostic |
| vLLM c=8 online parity | 当前没有严格 online ingress + WER/ASR 结论 | 合作者填写 | 不要过度声明 |

如果复跑时 vLLM c=8 prebuild 结果显著改善，应先确认是否仍是 offline diagnostic；只有补齐
online serving ingress、同口径 WER/ASR 和 stage evidence 后，才能讨论 strict c=8 parity。

## 6. Stage 连接复核

复跑后重点检查 `stage_interaction_summary.json`：

| Stage / 边界 | 必须保持的结论 | 复跑结果（合作者填写） | 判定（PASS/FAIL/附录） |
| --- | --- | --- | --- |
| `sglang_talker_to_code2wav_healthy` | `true` | 合作者填写 | 必须 PASS |
| `sglang_code2wav_decode_not_bottleneck` | `true` | 合作者填写 | 必须 PASS |
| `vllm_original_c8_prompt_feed_limited` | `true` | 合作者填写 | 必须 PASS |
| `preprocessing_parallelism_regresses` | `true` | 合作者填写 | 必须 PASS |
| c=8/c=16 preprocessing lifecycle | 增长主要来自 queue/admission，不是 raw preprocess compute | 合作者填写 | 形态应一致 |
| `code2wav_window_collect` | 增长代表等待 talker chunk，不等价于 vocoder 慢 | 合作者填写 | 解释应一致 |

读 stage 表时不要把 lifecycle、actual compute、handoff 和 collect wait 混成同一个瓶颈；
具体口径见 Stage 指标字典。

## 7. 是否可以替换报告数字

| 条件 | 结果（合作者填写） | 处理 |
| --- | --- | --- |
| 同硬件、同 image、同模型、同数据、full audit 全绿且 rerun acceptance contract 通过 | 合作者填写 | 可以考虑替换主报告数字 |
| full audit 绿，但硬件或 image 不同 | 合作者填写 | 作为外部复核附录，不覆盖 headline |
| claims 或 acceptance 失败 | 合作者填写 | 不替换；先定位差异 |
| coverage 或 manifest 失败 | 合作者填写 | 不替换；先补证据链 |
| vLLM prebuild 变好但无 online ingress/WER | 合作者填写 | 只更新 offline diagnostic，不改 parity 结论 |
| WER/ASR 不可复现 | 合作者填写 | 不更新语音一致性结论 |

## 8. 差异定位矩阵

复跑出现差异时先填这一节，再决定是否进入数字替换流程。不要先改主报告文案；
先定位差异属于环境、artifact、stage 解释、质量口径还是 runtime regime 变化。

| 复跑现象 | 优先查看 | 处理动作 | 是否可替换 headline |
| --- | --- | --- | --- |
| GPU、Docker image、模型路径或数据 cache 与 checkpoint 不同 | `environment_snapshot.json`；`manifest.json` | 标注为外部复核附录；不要覆盖 8x H20 headline | 否 |
| full audit 失败但 benchmark 原始结果存在 | `audit_run_summary.json`；失败 step stdout/stderr | 先修 artifact 引用、表格生成或 gate，再讨论性能数字 | 否 |
| `claims_verification.json` 有失败 | `claims_verification.json`；主报告对应 claim | 判断是性能边界被推翻还是输入 artifact 不一致；不要绕过 claim gate | 否 |
| `coverage_matrix.json` 或 `manifest.json` 缺项 | `coverage_matrix.json`；`manifest.json` | 补机器证据或移除不再有证据的结论 | 否 |
| SGLang c=8 不再是吞吐峰值 | `acceptance_matrix.json`；`headline_scorecard.json`；SGLang run logs | 检查 warmup、max-running、CUDA graph、preprocessing cache 和 profile run id | 只有全部 gate 重新通过才可 |
| c=16 看起来优于 c=8 | `stage_interaction_summary.json`；stage latency budget | 确认是否 queue/admission、quality、WER 和 failure 同时健康；否则仍是压力边界 | 只有新增 acceptance 规则后才可 |
| `sglang_talker_to_code2wav_healthy=false` | request profile；stage boundary bottleneck ledger | 先定位 handoff、chunk cadence 或 profiler 口径变化；不要继续说连接健康 | 否 |
| `sglang_code2wav_decode_not_bottleneck=false` | `stage_interaction_summary.json`；code2wav decode spans | 重新解释 stage boundary，并同步更新 causal graph、confidence ledger 和 Q&A | 否 |
| sample count、skip-first、warmup 或 fixed prompt 口径不同 | `repro_command_manifest.json`；`rerun_acceptance_contract.json` | 按 command ID 重跑固定口径，再重跑 regeneration/full audit | 否 |
| 只重跑 raw benchmark，没有重跑 regeneration/full audit | `audit_run_summary.json`；`final_readiness_audit.json` | 不替换主报告数字；先重生成表、图、JSON evidence 和 final readiness | 否 |
| 只改公开 Markdown 或图表，没有同步 JSON/manifest/share bundle | `share_consistency_guard.json`；`share_release_seal.json` | 按 silent-replacement 风险处理；重新打包并跑 receiver quickcheck | 否 |
| WER/ASR 结果不可复现或明显变差 | WER JSON；ASR logs；Whisper cache/ASR router | 补齐 ASR 权重/路由并重算；语音一致性不过关时不替换吞吐结论 | 否 |
| `length_regime_coverage.json` 不 ready，或 long c=8 RTF p95 不小于 1 | `length_regime_coverage.json`；synthetic speech run log；stage latency budget | 先确认 short/long fixed prompt、c=1/4/8 覆盖和 talker/code2wav guardrail；不要继续宣称长短文覆盖已闭环 | 否 |
| vLLM c=4 warmed 变强或变弱 | vLLM run log；vLLM optimization lock；runtime image contract | 确认镜像、compile、CUDA graph、prefix/chunked prefill 和 shared-memory transfer 一致 | 只有同口径全绿才可 |
| vLLM c=8 prebuild w4 显著改善 | vLLM admission diagnosis；vLLM online parity protocol | 只更新 offline diagnostic；补 online ingress、同口径 WER/ASR 和 stage evidence 前不升级 parity | 否 |
| share package validation、receiver smoke、receiver quickcheck contract 或 final completion audit 失败 | `share_package_validation.json`；`share_package_receiver_smoke_validation.json`；`receiver_quickcheck_contract.json`；`final_completion_audit.json` | 先修 tarball、checksum、asset quality、completion gate route、extracted-only 或 standalone 校验 | 否 |

差异定位完成后，把对应 JSON、run log、request profile、WER JSON 和填写后的本表一起回传；
如果任何一项仍是“否”，主报告 headline 保持当前 checkpoint，只在附录中描述外部复跑差异。

## 9. 复跑后应回传的文件

最小回传包：

- `results/qwen35_report_audit_20260619/audit_run_summary.json`
- `results/qwen35_report_audit_20260619/environment_snapshot.json`
- `results/qwen35_report_audit_20260619/manifest.json`
- `results/qwen35_report_audit_20260619/coverage_matrix.json`
- `results/qwen35_report_audit_20260619/claims_verification.json`
- `results/qwen35_report_audit_20260619/headline_scorecard.json`
- `results/qwen35_report_audit_20260619/tail_confidence_appendix.json`
- `results/qwen35_report_audit_20260619/acceptance_matrix.json`
- `results/qwen35_report_audit_20260619/confidence_ledger.json`
- `results/qwen35_report_audit_20260619/runtime_comparison_contract.json`
- `results/qwen35_report_audit_20260619/runtime_image_contract.json`
- `results/qwen35_report_audit_20260619/rerun_acceptance_contract.json`
- `results/qwen35_report_audit_20260619/receiver_quickcheck_contract.json`
- `results/qwen35_report_audit_20260619/sglang_optimization_lock.json`
- `results/qwen35_report_audit_20260619/vllm_optimization_lock.json`
- `results/qwen35_report_audit_20260619/vllm_online_parity_protocol.json`
- `results/qwen35_report_audit_20260619/repro_command_manifest.json`
- `results/qwen35_report_audit_20260619/final_readiness_audit.json`
- `results/qwen35_report_audit_20260619/share_bundle_manifest.json`
- `results/qwen35_report_audit_20260619/share_bundle_package_manifest.json`
- `results/qwen35_report_audit_20260619/metric_provenance_index.json`
- `results/qwen35_report_audit_20260619/objective_requirement_crosswalk.json`
- `results/qwen35_report_audit_20260619/vllm_admission_diagnosis.json`
- `results/qwen35_report_audit_20260619/vllm_log_stage_summary.json`
- `results/qwen35_report_audit_20260619/stage_interaction_summary.json`
- `results/qwen35_report_audit_20260619/stage_latency_budget.json`
- `results/qwen35_report_audit_20260619/stage_boundary_bottleneck_ledger.json`
- `results/qwen35_report_audit_20260619/stage_causal_graph.json`
- `results/qwen35_report_audit_20260619/stage_reproduction_drilldown.json`
- `results/qwen35_report_audit_20260619/stage_route_decision_matrix.json`
- `results/qwen35_report_audit_20260619/rerun_delta_triage.json`
- `results/qwen35_report_audit_20260619/share_package_validation.json`
- `results/qwen35_report_audit_20260619/share_package_receiver_smoke_validation.json`
- `results/qwen35_report_audit_20260619/share_package_validation_extracted.json`
- `results/qwen35_report_audit_20260619/share_package_external_standalone_validation.json`
- 复跑产生的 SGLang/vLLM `videoamme_results.json`、`run.log`、request profile 和 WER JSON。

建议附加回传，但不计入上面 `34` 个最小机器证据合同：

- `results/qwen35_report_audit_20260619/length_regime_coverage.json`
- `benchmarks/reports/qwen35_omni_length_regime_coverage_zh_20260621.md`
- `results/qwen35_report_audit_20260619/rerun_time_budget.json`
- `benchmarks/reports/qwen35_omni_rerun_time_budget_zh_20260621.md`
- host 模式 evidence query card smoke 的 stdout/log：
  `bash benchmarks/eval/qwen35_omni_evidence_query_cards_smoke.sh --root "$HOST_REPO" --mode host`
- portable 模式 evidence query card smoke 的 stdout/log：
  `bash "$BUNDLE_ROOT/benchmarks/eval/qwen35_omni_evidence_query_cards_smoke.sh" --root "$BUNDLE_ROOT" --mode portable`

这些附加材料让 reviewer 能直接复查长短文 coverage、复跑排期预算和查询卡；如果缺失，不自动判定
rerun acceptance contract 失败，但会降低外部评审排查效率。

## 10. 对外结论填写模板

复跑通过时建议写：

> 在 reviewer 机器上，按 handoff runbook 复跑 full audit 后，claims、coverage、
> preflight、manifest、headline scorecard、acceptance matrix 和 confidence ledger 全部通过。
> 当前复跑支持 8x H20 checkpoint 的主要结论：SGLang warmed c=4 优于优化版 vLLM，
> c=8 是 SGLang 当前吞吐峰值，stage 连接健康，主要瓶颈仍在 admission/queueing 和
> talker AR tail。

复跑有差异时建议写：

> 本次复跑环境与 8x H20 checkpoint 存在差异，结果可作为外部复核证据，但暂不覆盖主报告
> headline。差异项为：GPU/image/model/data/ASR/参数中的哪几项；需要补充的证据为：
> claims、stage interaction、WER/ASR 或 vLLM online ingress。
