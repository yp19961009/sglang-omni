# Qwen3.5-Omni 性能报告答辩 Q&A

状态：证据门已就绪；更新后的目标不再等待 6.21 晚间，后续变更必须重跑 full audit。
工作目录：`/home/gangouyu/sglang-omni`。
用途：给合作高校分享时使用的口头答辩稿，重点固定“能说什么、不能说什么、证据在哪里”。

关联材料：

- 主报告：
  `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_stress_performance_plan_20260621.md`
- 原始需求-证据映射：
  `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_requirement_evidence_map_zh_20260621.md`
- 压力条件总表：
  `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_pressure_matrix_zh_20260621.md`
- 数字来源索引：
  `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_metric_source_map_zh_20260621.md`
- Stage 指标字典：
  `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_stage_metric_dictionary_zh_20260621.md`
- Stage 因果图：
  `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_stage_causal_graph_zh_20260621.md`
- 优化 playbook：
  `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_optimization_playbook_zh_20260621.md`
- 复现清单：
  `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_reproduction_checklist_zh_20260621.md`
- 外部复现 handoff runbook：
  `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_external_handoff_runbook_zh_20260621.md`
- 合作方复跑验收表：
  `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_collaborator_rerun_validation_sheet_zh_20260621.md`
- 复跑差异定位矩阵：
  `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_rerun_delta_triage_zh_20260621.md`
- 分享 Deck 图表资产映射：
  `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_slide_asset_map_zh_20260621.md`
- 机器审计：
  `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/audit_run_summary.json`

## 现场证据卡

现场优先跑 full audit；如果只是在问答中快速定位某个结论，下面这些是只读 `jq`
命令，不会改写当前 evidence package。
如果现场追问 stage breakdown、stage 连接或某个瓶颈是否可复现，优先查
`stage_reproduction_drilldown.json` 的 `quick_reproduction_map`；它给出 5 条答辩
quick route，再下钻到对应 stage row、metric row 和 rerun command。

| 追问 | JSON key / 机器来源 | 快查命令 |
| --- | --- | --- |
| warmed c=4 是否优于 vLLM | `headline_scorecard.json` -> `.strict_c4_comparison` | `jq '.strict_c4_comparison' results/qwen35_report_audit_20260619/headline_scorecard.json` |
| c=8 为什么是 SGLang 峰值 | `headline_scorecard.json` -> `.sglang_stress.throughput_peak` | `jq '.sglang_stress.throughput_peak' results/qwen35_report_audit_20260619/headline_scorecard.json` |
| 长文本/长语音是否快于实时 | `headline_scorecard.json` -> `.synthetic_long_c8` | `jq '.synthetic_long_c8' results/qwen35_report_audit_20260619/headline_scorecard.json` |
| stage 连接是否健康 | `stage_interaction_summary.json` -> `.summary` | `jq '.summary' results/qwen35_report_audit_20260619/stage_interaction_summary.json` |
| stage 瓶颈怎么快速复现 | `stage_reproduction_drilldown.json` -> `.quick_reproduction_map` | `jq '.quick_reproduction_map[] \| {question, stage_row_id, metric_row_id, first_rerun_command_id}' results/qwen35_report_audit_20260619/stage_reproduction_drilldown.json` |
| stage 因果链和原始 artifact 怎么追 | `qwen35_omni_stage_causal_graph_zh_20260621.md` -> manifest-backed 原始证据 Drilldown | `rg -n '原始证据 Drilldown' benchmarks/reports/qwen35_omni_stage_causal_graph_zh_20260621.md` |
| vLLM c=8 为什么是 offline diagnostic | `vllm_admission_diagnosis.json` -> `.rows[]` 按 label 选择 `vLLM-c8`，不依赖 rows 顺序 | `jq '.rows[] \| select(.label == "vLLM-c8")' results/qwen35_report_audit_20260619/vllm_admission_diagnosis.json` |
| vLLM prebuild w4 到底改善了什么 | `headline_scorecard.json` -> `.vllm_c8_diagnostics.prebuild_w4` | `jq '.vllm_c8_diagnostics.prebuild_w4' results/qwen35_report_audit_20260619/headline_scorecard.json` |
| 复跑数字能否替换主报告 | `rerun_acceptance_contract.json` -> `.summary` | `jq '.summary' results/qwen35_report_audit_20260619/rerun_acceptance_contract.json` |
| 复跑数字偏了怎么定位 | `rerun_delta_triage.json` -> `.rows[]` 按症状选择 stage/boundary | `jq '.summary, .rows[] \| {symptom, likely_stage, replacement_scope}' results/qwen35_report_audit_20260619/rerun_delta_triage.json` |
| PPT 该插哪张图 | `slide_asset_map.json` -> `.rows[]` 映射 deck section、SVG/CSV 和讲法 | `jq '.summary, .rows[] \| {deck_section, primary_asset, data_asset}' results/qwen35_report_audit_20260619/slide_asset_map.json` |
| 当前包是否可分享 | `audit_run_summary.json` -> full audit / readiness / package gates | `jq '{ok, final_readiness, share_package_validation, share_package_receiver_smoke_validation}' results/qwen35_report_audit_20260619/audit_run_summary.json` |

## 1. SGLang-Omni 是否至少和 vLLM 相当？

短答：

在当前 8x H20、Video-AMME ci-50、语音输出 workload、warmed c=4 严格横向对比中，
SGLang-Omni 比优化版 vLLM 更快，且 accuracy/WER 不退化。

关键数字：

| Runtime | Latency Mean | Latency P95 | RTF Mean | RTF P95 | Accuracy | WER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SGLang-Omni | 1.743s | 3.328s | 1.3536 | 2.4023 | 67.4% | 4.12% |
| vLLM | 2.093s | 3.525s | 1.4677 | 3.0717 | 63.0% | 7.44% |

可说：

- “在 warmed c=4 严格对比中，SGLang-Omni latency/RTF 均优于优化版 vLLM。”
- “accuracy 和 WER 没有因为加速而退化。”

不能说：

- “所有流量、所有并发下 SGLang 都严格压过 vLLM。”
- “vLLM c=8 online serving parity 已经完成。”

证据入口：

- `results/qwen35_report_audit_20260619/claims_verification.json`
- `results/qwen35_report_audit_20260619/headline_scorecard.json`
- 完整报告 section 1/5

## 2. vLLM baseline 是不是故意设弱了？

短答：

不是。vLLM 使用 Qwen3.5-capable 镜像，并锁定为非 eager 弱 baseline：服务侧开启
compile/CUDA graph、prefix/chunked prefill、shared-memory transfer，encoder 侧开启
encoder compile/batch；c=8 还补了 prebuilt prompt 的 w1/w4 offline diagnostic，其中
prebuild w4 用来确认原始 offline runner 的 prompt build/feed admission 限制。

可说：

- “vLLM baseline 不是 eager 弱 baseline。”
- “vLLM baseline 不是弱 baseline；公平性证据已经由 vLLM optimization lock 固化。”
- “c=4 对比使用优化版 vLLM；c=8 额外保留原始路径和 prebuild w4 诊断。”

不能说：

- “vLLM 已经做完所有可能优化。”
- “prebuild w4 的 runner QPS 就等于 online serving QPS。”

证据入口：

- 完整报告 section 4.1/5/11.7
- `benchmarks/reports/qwen35_omni_vllm_optimization_lock_zh_20260621.md`
- `results/qwen35_report_audit_20260619/vllm_optimization_lock.json`
- `results/qwen35_report_audit_20260619/vllm_admission_diagnosis.json`
- `results/qwen35_report_audit_20260619/vllm_log_stage_summary.json`

## 3. 为什么 headline 选 warmed c=4？

短答：

c=4 是两边都有清楚 warmed steady-state、WER/accuracy、latency/RTF 的严格横向对比点，
能避开 cold compile/CUDA graph capture 对 tail 的污染。

可说：

- “headline 用 warmed skip-first-4，是为了看稳态服务表现。”
- “cold start 单独报告，不混进 headline。”

不能说：

- “cold start 不重要。”
- “c=4 是唯一有价值的运行点。”

证据入口：

- 完整报告 section 1/10
- `results/qwen35_report_audit_20260619/headline_scorecard.json`

## 4. 单并发结果说明什么？

短答：

单并发主要用于隔离排队影响，确认基础请求 latency/RTF 和 talker tail。当前证据显示
SGLang 在 c=1 request latency/RTF 上明显优于 vLLM，且生成一致性稳定。

可说：

- “c=1 不是吞吐结论，而是基础路径和 tail 形态检查。”
- “低并发下已经能看到 SGLang 的 request path 优势。”

证据入口：

- 完整报告 section 5/7
- `results/qwen35_report_audit_20260619/acceptance_matrix.json`

## 5. 为什么 c=8 是推荐高并发点？

短答：

在当前 SGLang recipe 下，c=8 是吞吐峰值：2.540 req/s，5.372 generated-audio
seconds per wall second。c=16 没有 accuracy/WER 崩溃，但 QPS 回落、RTF 和 queue tail
变差，所以是压力边界，不是推荐默认点。

可说：

- “推荐 serving window 是 c=4-c=8。”
- “c=8 是当前 recipe 的吞吐峰值。”

不能说：

- “c=16 不可用。”
- “c=8 是任何配置下的全局最优。”

证据入口：

- 完整报告 section 7/12/13
- `results/qwen35_report_audit_20260619/acceptance_matrix.json`
- `results/qwen35_report_audit_20260619/stage_interaction_summary.json`

## 6. 长短文本/语音输出覆盖了吗？

短答：

覆盖了。报告有 short/long text-to-speech 的 c=1/4/8 guardrail；短文本是
74 chars / 12 words，长文本是 944 chars / 139 words。long c=8 平均生成约
52.3s 音频，平均 latency 约 25.8s，RTF 0.4932，仍快于实时。

可说：

- “短/长文本输入 + 语音输出都有压力覆盖。”
- “long c=8 仍快于实时，长语音没有把 code2wav decode 打成主瓶颈。”

不能说：

- “官方 SeedTTS full-set headline 已完成。”
- “synthetic long 完全等价于真实全量长音频流量。”

证据入口：

- 完整报告 section 5/6.3
- `results/qwen35_report_audit_20260619/length_regime_coverage.json`
- `results/qwen35_report_audit_20260619/tables_summary.json`
- `results/qwen35_report_audit_20260619/videoamme_seedtts_meta_summary.json`

## 7. 每个 stage 的瓶颈在哪里？

短答：

c=1/c=2/c=4 主要是 talker AR tail；c=8/c=16 出现 preprocessing admission/queueing
叠加 talker tail；long speech 主要仍是 talker AR；code2wav decode 不是当前 compute
bottleneck。

可说：

- “stage breakdown 不是只看 top latency，还看实际 compute、queue 和 stage handoff。”
- “当前优先优化 talker AR、admission/batching，而不是先动 vocoder decode。”

证据入口：

- 完整报告 section 7.1/8.1/12.1
- `results/qwen35_report_audit_20260619/stage_interaction_summary.json`
- `benchmarks/reports/qwen35_omni_stage_causal_graph_zh_20260621.md`
- `results/qwen35_report_audit_20260619/stage_boundary_bottleneck_ledger.json`
- `results/qwen35_report_audit_20260619/tables_summary.json`

## 8. Stage 之间有没有卡住？

短答：

当前 SGLang 证据不支持 stage handoff 是主瓶颈。`talker_ar -> code2wav` hop p95
在 c=1/2/4/8/16 下约 15-24ms；`code2wav_window_collect` 变长主要是在等 talker
codec chunk，不是 decode compute 卡住。

可说：

- “stage 连接健康，慢在上游生成节奏和 admission/queueing。”
- “code2wav collect 变长不能直接解读成 vocoder 慢。”

不能说：

- “所有 stage 边界永远不会成为瓶颈。”
- “vLLM prebuild 后的 engine/talker-side tail 已经完全解释完。”

证据入口：

- 完整报告 section 12.1
- `results/qwen35_report_audit_20260619/stage_interaction_summary.json`
- `benchmarks/reports/qwen35_omni_stage_causal_graph_zh_20260621.md`
- 需要追到原始 artifact 时，从 Stage 因果图里的 manifest-backed 原始证据 Drilldown 进入。

## 9. 为什么不直接提高 preprocessing 并发？

短答：

已经测过，朴素提高 preprocessing 并发是负优化。preproc=2 让 c=8 QPS 从 2.540
降到 1.642，preproc=4 出现 OOM/失败请求。问题不是单纯 worker 少，而是共享资源争用和
admission/placement 需要一起调。

可说：

- “preprocessing queue 不是靠简单加并发就能修。”
- “后续若继续调，要同时改 admission、memory fraction 或 preprocessing/encoder placement。”

证据入口：

- 完整报告 section 9/14
- `results/qwen35_report_audit_20260619/acceptance_matrix.json`
- `results/qwen35_report_audit_20260619/claims_verification.json`

## 10. vLLM c=8 为什么不能直接作为 online parity？

短答：

原始 vLLM c=8 offline runner 主要受 host prompt build/feed admission 限制；prebuild w4
把 runner wall 从 w1 的 352.1s 降到 235.1s，但这仍是 offline diagnostic。严格
online serving parity 需要在线 ingress、WER/ASR 和 engine/talker boundary 复核。

可说：

- “vLLM c=8 prebuild w4 是当前最强 offline diagnostic。”
- “它证明原始 c=8 慢点很大一部分在 prompt build/feed admission。”

不能说：

- “vLLM c=8 online serving QPS 已经严格对齐。”
- “prebuild 后的 engine/talker tail 已经没有问题。”

证据入口：

- 完整报告 section 5/11.7/12
- `results/qwen35_report_audit_20260619/vllm_admission_diagnosis.json`
- `results/qwen35_report_audit_20260619/vllm_log_stage_summary.json`

## 11. WER 和语音一致性有没有退化？

短答：

没有在当前证据中看到退化。SGLang stress sweep 的 offline Whisper large-v3 WER 在
c=1/2/4/8/16 下稳定，warmed c=4 的 WER 也优于 vLLM c=4。

可说：

- “性能提升没有以 WER 回退为代价。”
- “对外复现 WER 需要可用 Whisper large-v3 权重或 ASR router。”
- “host 侧 `/root/.cache/whisper/large-v3.pt` 缺失是 optional warning；它不代表
  serving benchmark 失败，也不能单独推翻 latency/RTF 结论。”

证据入口：

- 完整报告 section 1/7
- `results/qwen35_report_audit_20260619/claims_verification.json`
- `results/qwen35_report_audit_20260619/receiver_quickcheck_contract.json`
- `results/qwen35_report_audit_20260619/repro_command_manifest.json` 的
  `check_wer_asr_path`
- `benchmarks/reports/qwen35_omni_receiver_command_card_zh_20260621.md`

现场快查：

```bash
jq '.checks[] | select(.name=="public receiver docs preserve WER/ASR rerun path")' \
  results/qwen35_report_audit_20260619/receiver_quickcheck_contract.json

jq '.commands[] | select(.id=="check_wer_asr_path")' \
  results/qwen35_report_audit_20260619/repro_command_manifest.json
```

## 12. 现在的结论有多大外推范围？

短答：

高置信范围是本地 8x H20、Qwen3.5-Omni、Video-AMME ci-50、语音输出 workload 和当前
recipe。更大 Video-AMME、真实线上流量、官方 SeedTTS full-set、严格 vLLM c=8 online
parity 都需要额外复核。

可说：

- “当前证据包对本地目标 workload 很强。”
- “更大数据和真实流量是下一阶段外推验证。”

不能说：

- “ci-50 等价于所有线上流量。”
- “当前 stress/synthetic 证据已经覆盖完整线上流量分布。”
- “官方 SeedTTS full-set 已经支持 headline。”

证据入口：

- `results/qwen35_report_audit_20260619/confidence_ledger.json`
- 完整报告 section 14

## 13. 如何现场复现或验收？

短答：

先按外部 handoff runbook 跑一条 full audit，确认所有 gate 通过；需要重跑性能时再按复现清单分别启动
SGLang serving、SGLang stress/WER 和 vLLM offline/prebuild 诊断。

```bash
cd /home/gangouyu/sglang-omni

python3 -m benchmarks.eval.run_qwen35_omni_report_audit \
  --root /home/gangouyu/sglang-omni \
  --summary-output results/qwen35_report_audit_20260619/audit_run_summary.json
```

期望 gate：

- full audit `ok=true`
- claims `17/17`
- coverage `34/34`
- preflight `62` checks, `0` required failures
- manifest current `196` records, minimum `180`, `0` missing
- SGLang optimization lock `26/26`
- repro command manifest `63` commands / `7` phases
- defense claim matrix `10` claim rows / `13` Q&A question rows / checks `17/17` /
  `0` required failures；WER/ASR path guard 已覆盖 optional Whisper cache warning
- headline scorecard `9/9`
- acceptance matrix `17/17`
- confidence ledger `12/12`

证据入口：

- `results/qwen35_report_audit_20260619/audit_run_summary.json`
- `benchmarks/reports/qwen35_omni_reproduction_checklist_zh_20260621.md`
- `benchmarks/reports/qwen35_omni_requirement_evidence_map_zh_20260621.md`
- `benchmarks/reports/qwen35_omni_pressure_matrix_zh_20260621.md`
- `benchmarks/reports/qwen35_omni_metric_source_map_zh_20260621.md`
- `benchmarks/reports/qwen35_omni_stage_metric_dictionary_zh_20260621.md`
- `benchmarks/reports/qwen35_omni_optimization_playbook_zh_20260621.md`

## 14. 主张-证据-复跑-裁决矩阵

这张矩阵用于现场答辩时把每个可说主张压到四个动作：先说哪句话、打开哪个机器证据、要复跑哪条入口、如果证据失败怎么裁决。
不要用单个表格数字替代 full audit；任何 headline 替换都必须回到 rerun acceptance contract 和 rerun delta triage。
机器 JSON 里的 `qna_question_rows` 会把前 13 个现场问题映射到本节 10 条可声明主张，
因此问法可以多样，但对外口径、证据跳转和撤回条件仍收敛到同一套 claim gate。

| 主张 | 可说口径 | 机器证据 | 复跑命令/入口 | 失败时裁决 |
| --- | --- | --- | --- | --- |
| SGLang warmed c=4 优于优化版 vLLM | 当前 8x H20、Video-AMME ci-50、warmed c=4 严格对比中，SGLang latency/RTF 更好且质量不退化 | `claims_verification.json`; `headline_scorecard.json`; `runtime_comparison_contract.json` | `run_full_audit`; `vllm_c1_original`; `vllm_c4_original`; `vllm_c8_original`; strict c=4 artifacts | claims 或 headline 失败时不得沿用 headline，进入 rerun acceptance 替换评审 |
| vLLM baseline 不是弱 baseline | vLLM 使用 Qwen3.5-capable 镜像和 compile/CUDA graph、prefix/chunked prefill、shared-memory transfer、encoder compile/batch 等优化证据 | `vllm_optimization_lock.json`; `runtime_image_contract.json`; `vllm_log_stage_summary.json` | `build_vllm_optimization_lock`; `build_runtime_image_contract`; vLLM c=1/c=8 commands | image/optimization lock 失败时只能说现有 vLLM 证据不可复核，不能说 baseline 公平 |
| SGLang c=8 是当前高并发峰值 | c=8 是当前 recipe 的吞吐峰值，c=16 是压力边界，不是推荐默认点 | `acceptance_matrix.json`; `stage_latency_budget.json`; `stage_interaction_summary.json` | `sglang_videoamme_stress`; `build_acceptance_matrix`; `build_stage_latency_budget` | c=8 不再为峰值时先按 rerun delta triage 定位 admission/queue，不直接改主报告数字 |
| short/long text-to-speech 已覆盖 | short 74 chars / 12 words，long 944 chars / 139 words，c=1/4/8 均覆盖，long c=8 仍快于实时 | `length_regime_coverage.json`; `tables_summary.json`; `stage_latency_budget.json`; `headline_scorecard.json` | `sglang_synthetic_text_to_speech`; `build_report_tables`; `build_stage_latency_budget` | 输入形状或 long c=8 RTF 失败时，相关长短文结论不得替换或外推 |
| stage handoff 没有卡住 | talker 到 code2wav 的 stream hop p95 约 15-24ms，当前不是主瓶颈 | `stage_interaction_summary.json`; `stage_boundary_bottleneck_ledger.json`; stage causal graph Drilldown | `build_stage_interactions`; `build_stage_boundary_bottleneck_ledger`; `build_stage_causal_graph` | handoff health 失败时，不能继续说 stage 连接健康，先补 profile drilldown |
| code2wav decode 不是当前 compute bottleneck | decode 平均约 14-17ms/window，collect wait 更多是在等 talker chunk cadence | `claims_verification.json`; `stage_latency_budget.json`; `stage_boundary_bottleneck_ledger.json` | `sglang_videoamme_stress`; `sglang_synthetic_text_to_speech`; `build_stage_boundary_bottleneck_ledger` | decode 成为主项时，当前 code2wav-not-bottleneck 结论必须撤回或重写 |
| 朴素提高 preprocessing 并发是负优化 | preproc=2 回退，preproc=4 失败；当前应先管 admission、placement 和 shared-resource contention | `acceptance_matrix.json`; `sglang_optimization_lock.json`; `stage_interaction_summary.json` | `build_sglang_optimization_lock`; anti-recipe artifacts; full audit | 新候选 recipe 必须补齐 c=4/c=8/c=16、WER、profile 和稳定性证据后再评审 |
| vLLM c=8 prebuild w4 只是 offline diagnostic | prebuild w4 改善 runner prompt build/feed，但没有证明 online serving parity | `vllm_admission_diagnosis.json`; `vllm_online_parity_protocol.json`; `runtime_comparison_contract.json` | `vllm_c8_prebuild_w4`; `build_vllm_online_parity_protocol`; `build_runtime_comparison_contract` | `online_parity_proven=false` 时不得把 c=8 prebuild 写成 online parity |
| WER/quality 没有为性能让步 | SGLang stress WER 稳定，strict c=4 WER/accuracy 不劣于 vLLM | `claims_verification.json`; `headline_scorecard.json`; WER artifacts | `sglang_recompute_wer`; `verify_report_claims`; full audit | WER 或 ASR 路径不一致时，不得只替换 latency/RTF headline |
| 当前包可分享但仍有边界 | share-ready 是带 caveat 的阶段稿；更大数据/真实流量外推、SeedTTS full-set 和 vLLM c=8 online parity 不能越界 | `final_readiness_audit.json`; `confidence_ledger.json`; `caveat_adjudication_matrix` | `run_full_audit`; `validate_share_bundle_package`; receiver smoke | 任一 package/readiness gate 失败时先修包，不讨论性能结论 |
