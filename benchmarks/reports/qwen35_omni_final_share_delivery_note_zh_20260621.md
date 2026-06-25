# Qwen3.5-Omni 最终分享交付说明

状态：2026-06-21 evidence-ready 交付稿；更新后的目标不再等待 6.21 晚上。
工作目录：`/home/gangouyu/sglang-omni`。
用途：作为发给合作高校的交付说明，明确本次分享包包含什么、先读什么、如何复现、哪些结论
可以高置信引用、哪些边界必须一起说明。

## 0. 路径说明

本文命令默认仓库根目录为 `/home/gangouyu/sglang-omni`。合作方如果把仓库挂载到其他路径，
先设置：

```bash
export HOST_REPO="${HOST_REPO:-/home/gangouyu/sglang-omni}"
export SMOKE_DIR="${SMOKE_DIR:-/tmp/qwen35_omni_receiver_smoke_delivery}"
export EXTRACT_DIR="${EXTRACT_DIR:-/tmp/qwen35_omni_share_bundle_delivery}"
export STANDALONE_DIR="${STANDALONE_DIR:-/tmp/qwen35_omni_external_standalone_bundle_validation_delivery}"
```

host 侧收包快检、完整审计和报告再生成都在 `$HOST_REPO` 下执行；receiver smoke、
extracted-only 和 standalone 的临时目录只用于收包校验，不用于替换 benchmark artifact。

## 1. 建议发送文件

建议把下面文件作为同一批 share package 发送或挂在同一个目录下：

| 类别 | 文件 | 用途 |
| --- | --- | --- |
| 最短入口 | `qwen35_omni_start_here_zh_20260621.md` | 30 秒结论、收包快检命令、evidence-query smoke、先读文件和 caveat 红线 |
| 外发正文 | `qwen35_omni_university_share_cover_note_zh_20260621.md` | 可直接粘贴到邮件、微信或飞书的合作高校说明，包含附件顺序和禁止单独外发的说法 |
| 必读入口 | `qwen35_omni_share_package_index_zh_20260621.md` | 五分钟阅读顺序、证据入口、答辩问题路由 |
| 接收方路径手册 | `qwen35_omni_receiver_package_path_map_zh_20260621.md` | 解包后区分 `HOST_REPO`、`BUNDLE_ROOT` 和 container 路径，避免把生成端绝对路径当作接收方目录 |
| 接收方命令卡 | `qwen35_omni_receiver_command_card_zh_20260621.md` | 一页式收包快检、full audit、性能复跑入口、回传验收和 headline replacement 红线 |
| 接收方 quickcheck contract | `qwen35_omni_receiver_quickcheck_contract_zh_20260621.md` | 机器检查一键收包入口是否仍覆盖 tarball、receiver-smoke、extracted-only、standalone 验证和 final completion gate 路由 |
| 中文技术报告 | `qwen35_omni_university_technical_report_zh_20260621.md` | 可直接发给合作高校的中文正文，串联 headline、压力条件、stage breakdown、vLLM 边界和复现入口 |
| 压力条件复现矩阵 | `qwen35_omni_pressure_repro_matrix_zh_20260621.md` | 按 strict c=4、SGLang c=1/2/4/8/16、短/长文本、vLLM diagnostic 和 WER/ASR 查 command id、证据文件和替换边界 |
| Evidence query cards | `qwen35_omni_evidence_query_cards_zh_20260621.md` | 用可复制 `jq` 查询现场证明 headline、stage、vLLM baseline、复跑替换边界和分享包 gate |
| Evidence query card smoke | `qwen35_omni_evidence_query_cards_smoke.sh` | 一键执行自证查询卡并断言核心 SGLang/vLLM、stage、复跑和 receiver gate；host 用 `--root "$HOST_REPO" --mode host`，解包后用 `--root "$BUNDLE_ROOT" --mode portable`；PASS/skip 摘要在 stdout，`--output` 只保存查询正文 |
| 最终状态 | `qwen35_omni_final_status_summary_zh_20260621.md` | 一页确认 full audit、objective、readiness、manifest、tarball 和 caveat |
| Caveat 裁决矩阵 / caveat adjudication matrix | `qwen35_omni_caveat_adjudication_matrix_zh_20260621.md` | 哪些 caveat 可以分享、哪些说法禁止、哪些情况触发补跑或替换数字 |
| 一页数字 | `qwen35_omni_one_page_scorecard_zh_20260621.md` | headline、压力条件、stage 结论和 gate 的最快视图 |
| Deck 图表资产映射 / slide asset map | `qwen35_omni_slide_asset_map_zh_20260621.md` | 每页应该使用的 SVG/CSV/JSON 证据，避免手工挑图或手改数字 |
| 图表来源一致性 / chart source consistency | `qwen35_omni_chart_source_consistency_zh_20260621.md` | 校验 14 个 CSV/SVG 图表与审计 JSON 重生成结果 byte-exact 一致 |
| Final completion audit | `qwen35_omni_final_completion_audit_zh_20260621.md` | tarball 外侧伴随最终完成门；确认 full audit、watchlist、release seal 和时间 gate 的当前状态 |
| Share release seal | `qwen35_omni_share_release_seal_zh_20260621.md` | tarball 外侧伴随封口页，确认 checksum、validation、final readiness 和 caveat 一起发送 |
| 决策矩阵 | `qwen35_omni_regime_decision_matrix_zh_20260621.md` | 每个 pressure regime 是否推荐、瓶颈是什么、下一步怎么做 |
| 公平对比合同 / runtime comparison contract | `qwen35_omni_runtime_comparison_contract_zh_20260621.md` | strict c=4 headline、SGLang scaling、vLLM c=8 offline diagnostic 和无效 parity 比较的边界 |
| 复跑验收阈值合同 / rerun acceptance contract | `qwen35_omni_rerun_acceptance_contract_zh_20260621.md` | 合作方复跑后如何判定“确认形态 / 只能附录 / 可替换 headline 数字”的机器阈值 |
| SGLang 优化锁定 / SGLang optimization lock | `qwen35_omni_sglang_optimization_lock_zh_20260621.md` | SGLang 镜像、compiled/graph recipe、c=8 峰值、stage handoff 和 anti-recipe 边界锁定 |
| vLLM 优化锁定 / vLLM optimization lock | `qwen35_omni_vllm_optimization_lock_zh_20260621.md` | vLLM 镜像、compile/CUDA graph/cache/prebuild 开关和 offline diagnostic 边界锁定 |
| 优化候选裁决 / optimization candidate ledger | `qwen35_omni_optimization_candidate_ledger_zh_20260621.md` | 当前 measured-best recipe、anti-recipe、vLLM baseline/diagnostic 和 code2wav-first 优先级裁决 |
| vLLM online parity protocol | `qwen35_omni_vllm_online_parity_protocol_zh_20260621.md` | vLLM c=8 从 offline diagnostic 升级为 strict online parity 前的必需 artifact 和替换 gate |
| Final checkpoint watchlist | `qwen35_omni_final_checkpoint_watchlist_zh_20260621.md` | 最终交付前的 caveat、补跑触发器和 completion audit 红线 |
| Stage latency budget | `qwen35_omni_stage_latency_budget_zh_20260621.md` | SGLang/vLLM 每个关键 stage 相对 latency 的压力占比、queue/admission 和 handoff 解释 |
| Stage boundary bottleneck ledger | `qwen35_omni_stage_boundary_bottleneck_ledger_zh_20260621.md` | 每个 stage boundary 的瓶颈判定、证据数字、优化动作和 claim scope |
| Stage 因果图 / stage causal graph | `qwen35_omni_stage_causal_graph_zh_20260621.md` | stage 之间的因果关系、瓶颈转移、连接健康度、manifest-backed 原始证据 Drilldown 和 vLLM offline admission 读法 |
| Stage 因果图 JSON | `stage_causal_graph.json` | 7/7 机器 gate、7 条 causal edge、5 行 raw drilldown 和 manifest-backed raw artifact 检查 |
| Stage 复现实操 Drilldown | `qwen35_omni_stage_reproduction_drilldown_zh_20260621.md` | 先看第 2 节 5 条答辩 quick route；再按每个 stage row 下钻 jq 查询、metric provenance row、raw artifact 和 rerun command ID |
| Stage route 裁决矩阵 | `qwen35_omni_stage_route_decision_matrix_zh_20260621.md` | 11 条 route 的瓶颈裁决、优化动作、安全说法和复核入口 |
| 图表数据 | `results/qwen35_report_audit_20260619/share_charts/` | 可直接放 PPT 的 SVG 图和可复核 CSV，数字来自审计 JSON |
| 对外简报 | `qwen35_omni_collaboration_brief_zh_20260621.md` | 给合作方快速理解结论和边界 |
| 完整报告 | `qwen35_omni_stress_performance_plan_20260621.md` | 完整性能分析、stage breakdown、vLLM 对比和复现命令 |
| 原始需求映射 | `qwen35_omni_requirement_evidence_map_zh_20260621.md` | 用户原始要求逐项对应证据 |
| 压力条件总表 | `qwen35_omni_pressure_matrix_zh_20260621.md` | 单并发、高并发、短/长文本/语音、vLLM 诊断和反例的总览 |
| 数字来源 | `qwen35_omni_metric_source_map_zh_20260621.md` | headline、stage、vLLM、anti-recipe 数字来源 |
| Stage 口径 | `qwen35_omni_stage_metric_dictionary_zh_20260621.md` | lifecycle、compute、handoff、collect wait 的解释 |
| Handoff runbook | `qwen35_omni_external_handoff_runbook_zh_20260621.md` | 外部 reviewer 最短复现路径 |
| 复跑验收表 | `qwen35_omni_collaborator_rerun_validation_sheet_zh_20260621.md` | 合作方复跑后逐项验收和决定是否替换数字 |
| 复跑差异定位 / rerun delta triage | `qwen35_omni_rerun_delta_triage_zh_20260621.md` | 复跑数字偏移时定位到环境、admission、talker、handoff、code2wav、WER 或 vLLM parity 边界 |
| 复现清单 | `qwen35_omni_reproduction_checklist_zh_20260621.md` | 具体 SGLang/vLLM 重跑命令 |
| 答辩稿 | `qwen35_omni_defense_qna_zh_20260621.md` | 常见追问和安全话术 |
| 优化协议 | `qwen35_omni_optimization_playbook_zh_20260621.md` | 当前 recipe、下一步调优顺序和回滚规则 |
| 分享提纲 | `qwen35_omni_share_deck_outline_zh_20260621.md` | 15-25 分钟技术分享/PPT 顺序 |

机器证据目录：

- `results/qwen35_report_audit_20260619/audit_run_summary.json`
- `results/qwen35_report_audit_20260619/environment_snapshot.json`
- `results/qwen35_report_audit_20260619/manifest.json`
- `results/qwen35_report_audit_20260619/claims_verification.json`
- `results/qwen35_report_audit_20260619/coverage_matrix.json`
- `results/qwen35_report_audit_20260619/headline_scorecard.json`
- `results/qwen35_report_audit_20260619/university_technical_report.json`
- `results/qwen35_report_audit_20260619/runtime_comparison_contract.json`
- `results/qwen35_report_audit_20260619/slide_asset_map.json`
- `results/qwen35_report_audit_20260619/share_charts/chart_pack_manifest.json`
- `results/qwen35_report_audit_20260619/chart_source_consistency.json`
- `results/qwen35_report_audit_20260619/share_charts/`
- `results/qwen35_report_audit_20260619/acceptance_matrix.json`
- `results/qwen35_report_audit_20260619/confidence_ledger.json`
- `results/qwen35_report_audit_20260619/objective_completion_audit.json`
- `results/qwen35_report_audit_20260619/objective_requirement_crosswalk.json`
- `results/qwen35_report_audit_20260619/repro_command_manifest.json`
- `results/qwen35_report_audit_20260619/command_reference_hygiene.json`
- `results/qwen35_report_audit_20260619/defense_claim_matrix.json`
- `results/qwen35_report_audit_20260619/claim_metric_crosswalk.json`
- `results/qwen35_report_audit_20260619/final_readiness_audit.json`
- `results/qwen35_report_audit_20260619/share_bundle_manifest.json`
- `results/qwen35_report_audit_20260619/stage_interaction_summary.json`
- `results/qwen35_report_audit_20260619/stage_drilldown_index.json`
- `results/qwen35_report_audit_20260619/metric_provenance_index.json`
- `results/qwen35_report_audit_20260619/stage_reproduction_drilldown.json`
- `results/qwen35_report_audit_20260619/stage_route_decision_matrix.json`
- `results/qwen35_report_audit_20260619/vllm_admission_diagnosis.json`
- `results/qwen35_report_audit_20260619/sglang_optimization_lock.json`
- `results/qwen35_report_audit_20260619/vllm_optimization_lock.json`
- `results/qwen35_report_audit_20260619/optimization_candidate_ledger.json`
- `results/qwen35_report_audit_20260619/caveat_adjudication_matrix.json`
- `results/qwen35_report_audit_20260619/vllm_online_parity_protocol.json`
- `results/qwen35_report_audit_20260619/rerun_acceptance_contract.json`
- `results/qwen35_report_audit_20260619/rerun_delta_triage.json`
- `results/qwen35_report_audit_20260619/final_checkpoint_watchlist.json`
- `results/qwen35_report_audit_20260619/stage_latency_budget.json`
- `results/qwen35_report_audit_20260619/stage_boundary_bottleneck_ledger.json`
- `results/qwen35_report_audit_20260619/tables_summary.json`
- `results/qwen35_report_audit_20260619/share_package_validation.json`
- `results/qwen35_report_audit_20260619/share_package_validation_extracted.json`
- `results/qwen35_report_audit_20260619/share_package_receiver_smoke_validation.json`
- `results/qwen35_report_audit_20260619/share_package_external_standalone_validation.json`
- `results/qwen35_report_audit_20260619/receiver_quickcheck_contract.json`
- `results/qwen35_report_audit_20260619/final_completion_audit.json`
- `results/qwen35_report_audit_20260619/share_release_seal.json`

便捷发送包：

- `results/qwen35_report_audit_20260619/qwen35_omni_share_bundle_20260621.tar.gz`
- `results/qwen35_report_audit_20260619/qwen35_omni_share_bundle_20260621.tar.gz.sha256`
- `results/qwen35_report_audit_20260619/share_bundle_package_manifest.json`
- `benchmarks/reports/qwen35_omni_final_completion_audit_zh_20260621.md`
- `benchmarks/reports/qwen35_omni_share_release_seal_zh_20260621.md`
- tarball 内 `PACKAGE_FILE_SHA256SUMS.txt`

当前 tarball SHA-256 以同目录的
`results/qwen35_report_audit_20260619/qwen35_omni_share_bundle_20260621.tar.gz.sha256`
为准；发送前用 `sha256sum -c` 验证。

这个 tarball 只用于方便发送，内容来自 `share_bundle_manifest.json` 推荐的报告、机器证据和
SVG/CSV 图表；权威复核仍以 full audit、`manifest.json` 和 `share_bundle_manifest.json`
为准。

注意：`share_bundle_package_manifest.json`、`share_package_validation.json`、
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

## 2. 当前版本 Gate

当前 checkpoint 的机器 gate：

- full audit `ok=true`
- claims `17/17`
- coverage `34/34`
- preflight `62` checks, `0` required failures
- manifest current `196` records, minimum `180`, `0` missing
- university technical report `ready=true`，中文正文入口覆盖 11 个章节和 16 个证据检查
- headline scorecard `9/9`
- metric provenance index `ready=true`，指标、raw artifact、stage drilldown 和复跑命令链路完整
- claim metric crosswalk `ready=true`，10 个对外主张可直接追到 metric row、raw artifact 和复跑命令
- slide asset map `ready=true`，deck section 到 SVG/CSV/JSON 证据的映射完整
- chart source consistency `ready=true`，14 个 CSV/SVG 图表与审计 JSON 重生成结果 byte-exact 一致
- acceptance matrix `17/17`
- confidence ledger `12/12`，high `9`、medium `3`、unsupported `0`
- objective completion audit `share_ready_with_documented_caveats=true`，17 rows，0 required failures
- objective requirement crosswalk `ready=true`，11 个原始需求行可追到 objective rows、claim ids、metric rows、raw artifacts、复跑命令和 8 条 optimization candidate verdict
- SGLang optimization lock `ready=true`，26/26 checks
- vLLM optimization lock `ready=true`，22/22 checks
- vLLM online parity protocol `ready=true`，18/18 checks，`online_parity_proven=false`
- runtime image contract `ready=true`，12/12 checks
- rerun acceptance contract `ready=true`，17/17 checks，18 rules，34 return evidence files，并包含 27-row command return evidence matrix 和 silent-replacement/protocol-drift guard
- rerun delta triage `ready=true`，复跑偏移症状覆盖 stage/boundary、裁决边界和下一步动作
- final checkpoint watchlist `ready=true`，24/24 checks，`final_completion_evidence_ready=true`，`checkpoint_phase=completion_audit_ready` 且 `completion_allowed_now=true`
- stage latency budget `ready=true`，12/12 checks
- stage boundary bottleneck ledger `ready=true`，12/12 checks，37 boundary rows，11 pressure transition rows
- stage route decision matrix `ready=true`，11 route rows，52 covered stage rows
- repro command manifest `ready=true`，63 条命令、7 个阶段
- command reference hygiene `ready=true`，结构化 rerun command IDs 均可解析
- final readiness audit `ready=true`，49/49 checks，0 required failures
- share path hygiene `ready=true`，package/raw offenders 和 legacy hits 均为 0
- public doc quality guard `public_doc_quality_guard=no bare hashes / malformed tables / malformed tokens / duplicate headings / semantic count drift`，且 hash/table/token/duplicate-heading/semantic-count offenders 均为空
- share bundle manifest `ready=true`
- share package validation `ready=true`，17/17 checks，0 required failures
- extracted-only package validation `ready=true`，13/13 checks，0 required failures
- external standalone package validation `ready=true`，8/8 checks，0 required failures
- receiver quickcheck contract `ready=true`，15/15 checks，6-step wrapper，4 receiver JSONs，completion-gate docs 为 6，WER/ASR docs 为 3，stage dictionary crosswalk needles 为 8，0 required failures
- final completion audit `ready=true`，required failures 为 0；更新后的目标不再等待 6.21 晚上，`completion_allowed_now=true` 才允许标记完成
- package asset quality evidence `report_quality_offenders=[]`，`chart_quality_offenders=[]`
- receiver quickcheck 会同时覆盖 checksum、tarball-mode validation、receiver-smoke validation、extracted-only validation 和 standalone validation

唯一已知 warning：host 侧 `/root/.cache/whisper/large-v3.pt` 不存在。它是 optional，
只影响 host 侧直接重算 offline WER；如果容器内已有 Whisper 权重或改用 ASR router，
不影响当前报告其他 gate。

## 3. 对外邮件/消息模板

可以直接发送下面这段：

```text
各位老师/同学好，

这版是 Qwen3.5-Omni 在 SGLang-Omni 上的性能分析分享包，覆盖 8x H20 环境下
Video-AMME ci-50 语音输出 workload、SGLang c=1/2/4/8/16 压测、短/长文本输入 + 语音输出、
vLLM optimized baseline、c=8 prebuild offline diagnostic，以及可直接放 PPT 的 SVG/CSV
图表包。
vLLM baseline 使用 Qwen3.5-capable 镜像，并保留 compile/CUDA graph、
prefix/chunked prefill、shared-memory transfer、encoder compile/batch 和 prebuild w4
等优化证据；它不是弱 baseline。

建议先读 qwen35_omni_share_package_index_zh_20260621.md；如果从 tarball/解包目录阅读，
先打开 qwen35_omni_receiver_package_path_map_zh_20260621.md 区分 HOST_REPO、
BUNDLE_ROOT 和 container 路径；如果只想先复制命令，打开
qwen35_omni_receiver_command_card_zh_20260621.md，再读中文简报和主报告。
需要复现时，请先跑 external handoff runbook 中的 full audit；如果要重跑性能，请按
reproduction checklist 执行，并用 collaborator rerun validation sheet 填写复跑结果。

当前 high-confidence 结论是：在本地 8x H20、Video-AMME ci-50、warmed c=4 严格对比中，
优化后的 SGLang-Omni Qwen3.5 比优化版 vLLM 在 latency/RTF 上更优，accuracy/WER 不退化；
SGLang c=8 是当前吞吐峰值；主要瓶颈在 high-concurrency admission/queueing 和 talker AR
tail，不在 code2wav decode 或 stage handoff。

边界说明：官方 SeedTTS full-set 不是本报告 headline；vLLM c=8 prebuild w4 是 offline
diagnostic，不是 online serving parity；更大流量外推需要继续复核。
```

## 4. 对外可引用结论

可以高置信引用：

- SGLang warmed c=4 在 latency mean/p95、RTF mean/p95 上优于优化版 vLLM warmed c=4。
- SGLang warmed c=4 accuracy 不低于 vLLM，WER 不高于 vLLM。
- SGLang 主压测覆盖 c=1/2/4/8/16，c=8 是当前 recipe 的吞吐峰值。
- c=16 是压力边界，不建议作为当前默认运行点。
- short/long text-to-speech 覆盖 c=1/4/8；短文本 74 chars / 12 words，长文本
  944 chars / 139 words；long c=8 仍快于实时。
- `talker_ar -> code2wav` handoff 健康，`code2wav_decode` 不是当前主要瓶颈。
- naive preprocessing 并发不是优化方向：preproc=2 回退，preproc=4 OOM/失败。
- vLLM baseline 不是弱 baseline：使用 Qwen3.5-capable 镜像，并由 vLLM optimization
  lock 固化 compile/CUDA graph、prefix/chunked prefill、shared-memory transfer、
  encoder compile/batch 和 prebuild w4 证据。
- vLLM original c=8 offline runner 主要受 prompt build/feed admission 限制；prebuild w4 是合理 offline diagnostic。

必须同步说明：

- 本报告 headline 基于 8x H20、本地 Video-AMME ci-50 checkpoint。
- ci-50/stress/synthetic 证据不能直接外推到完整线上流量。
- official SeedTTS full-set 还不是 headline evidence。
- vLLM c=8 prebuild w4 不是 online serving parity。
- host Whisper cache warning 是可选项；重算 WER 时需要 Whisper 权重或 ASR router。

## 5. 接收方第一步

接收方拿到 tarball 后，先做收包快检：

```bash
HOST_REPO="${HOST_REPO:-/home/gangouyu/sglang-omni}"
SMOKE_DIR="${SMOKE_DIR:-/tmp/qwen35_omni_receiver_smoke_delivery}"
EXTRACT_DIR="${EXTRACT_DIR:-/tmp/qwen35_omni_share_bundle_delivery}"
STANDALONE_DIR="${STANDALONE_DIR:-/tmp/qwen35_omni_external_standalone_bundle_validation_delivery}"
cd "$HOST_REPO"

bash benchmarks/eval/qwen35_omni_receiver_quickcheck.sh

# command_id: build_share_release_seal
python3 -m benchmarks.eval.build_qwen35_omni_share_release_seal \
  --root "$HOST_REPO" \
  --strict \
  --output benchmarks/reports/qwen35_omni_share_release_seal_zh_20260621.md \
  --json-output results/qwen35_report_audit_20260619/share_release_seal.json
```

通过标准是：`sha256sum` 输出 `OK`，receiver smoke `ready=true`，
`receiver_smoke_ready=true`，
tarball validation `17/17`，nested extracted-only validation `13/13`，standalone validation
`8/8`，并证明随包 validator 可在干净解包目录中直接运行。
资产 evidence 还应显示 `report_quality_offenders=[]`、`chart_quality_offenders=[]`。

快检失败分流：

| 失败点 | 先看证据 | 裁决 |
| --- | --- | --- |
| checksum FAIL | `.tar.gz.sha256` 和 `sha256sum -c` 输出 | 不进入报告阅读；先重传 tarball/checksum。 |
| tarball validation FAIL | `results/qwen35_report_audit_20260619/share_package_validation.json` | 不发送为最终包；修复缺失/mismatch 后重建 tarball。 |
| receiver smoke FAIL | `results/qwen35_report_audit_20260619/share_package_receiver_smoke_validation.json` | 不替换任何 benchmark 数字；先确认安全解包和 nested extracted-only gate。 |
| extracted-only validation FAIL | `results/qwen35_report_audit_20260619/share_package_validation_extracted.json` | 说明解包目录或随包文件不自洽；回到路径手册确认 `BUNDLE_ROOT` / `HOST_REPO`。 |
| standalone validation FAIL | `results/qwen35_report_audit_20260619/share_package_external_standalone_validation.json` | 说明随包 validator 不能在干净 `/tmp` 独立运行；先修包，不进入外部复现。 |
| quality offenders 非空 | validation JSON 里的 `report_quality_offenders` / `chart_quality_offenders` | 先修 Markdown/CSV/SVG 质量，再重跑 full audit 和 release seal。 |

随后执行完整审计：

```bash
HOST_REPO="${HOST_REPO:-/home/gangouyu/sglang-omni}"
cd "$HOST_REPO"

python3 -m benchmarks.eval.run_qwen35_omni_report_audit \
  --root "$HOST_REPO" \
  --summary-output results/qwen35_report_audit_20260619/audit_run_summary.json
```

如果只是审阅当前 checkpoint，看到 full audit 通过即可开始看报告。
如果要复跑性能，按下面顺序：

1. 读 `qwen35_omni_external_handoff_runbook_zh_20260621.md`。
2. 按 `qwen35_omni_reproduction_checklist_zh_20260621.md` 执行 SGLang 和 vLLM 命令。
3. 用 `qwen35_omni_collaborator_rerun_validation_sheet_zh_20260621.md` 填写复跑结果。
4. 只有 full audit、claims、coverage、preflight、manifest、headline、chart source consistency、acceptance、confidence、tail confidence appendix、rerun acceptance contract、repro command manifest、final readiness audit、share bundle manifest、share package validation、receiver smoke validation、extracted-only package validation 和 external standalone package validation 全部通过时，才考虑替换主报告数字。

## 6. 不建议发送的说法

不要这样说：

- “已经证明所有线上场景 SGLang 都优于 vLLM。”
- “vLLM c=8 prebuild w4 已经证明 online serving parity。”
- “ci-50/stress/synthetic 证据已经覆盖完整线上流量。”
- “code2wav 是瓶颈。”
- “preprocessing 并发越大越好。”
- “官方 SeedTTS full-set 已经覆盖 headline。”

推荐说法：

- “当前 8x H20、Video-AMME ci-50、warmed c=4 严格对比中，SGLang-Omni 优于优化版 vLLM。”
- “c=8 是当前 SGLang recipe 的吞吐峰值，c=16 是压力边界。”
- “当前主要优化方向是 talker AR 和 admission/queueing，不是先优化 code2wav decode。”
- “vLLM c=8 prebuild w4 用于定位 offline runner prompt-feed/admission 问题，不能直接当 online parity。”
- “ci-50/stress/synthetic 证据不能直接外推到完整线上流量；更大 Video-AMME 或真实线上流量需要同口径复跑和 gate 全绿。”

## 7. 最终交付自检

发送前确认：

- 分享包索引、中文技术报告、中文简报、主报告、runbook、复现清单、复跑验收表都在同一个目录或同一条消息里。
- 接收方命令卡和路径手册都在同一个目录或同一条消息里。
- 机器证据 JSON 路径可以访问。
- `audit_run_summary.json` 中 `ok=true`。
- coverage `34/34`、preflight `62`、manifest current `196` records / minimum `180` 都是最新值。
- `university_technical_report.json` 为 `ready=true`，且 checks 为 `16/16`。
- `runtime_comparison_contract.json` 为 `ready=true`，且 checks 为 `9/9`。
- `sglang_optimization_lock.json` 为 `ready=true`，且 checks 为 `26/26`。
- `vllm_optimization_lock.json` 为 `ready=true`，且 checks 为 `22/22`。
- `vllm_online_parity_protocol.json` 为 `ready=true`，且 `online_parity_proven=false`。
- `runtime_image_contract.json` 为 `ready=true`，并锁定 SGLang/vLLM 镜像、digest 和优化开关。
- `rerun_acceptance_contract.json` 为 `ready=true`，且 checks 为 `17/17`、rules 为 `18`、return evidence files 为 `34`、command return evidence rows 为 `27`，matrix 无缺口，并覆盖 silent-replacement/protocol-drift guard。
- `final_checkpoint_watchlist.json` 为 `ready=true`，checks 为 `24/24`，`final_completion_evidence_ready=true`，`checkpoint_phase=completion_audit_ready`，且 `completion_allowed_now=true`。
- `stage_latency_budget.json` 为 `ready=true`，且 checks 为 `12/12`。
- `repro_command_manifest.json` 为 `ready=true`。
- `chart_source_consistency.json` 为 `ready=true`，且 14 个图表文件 byte-exact。
- `objective_completion_audit.json` 为 `share_ready_with_documented_caveats=true`，且 required failures 为 0。
- `objective_requirement_crosswalk.json` 为 `ready=true`，且 required failures 为 0，并包含 8 条 optimization candidate verdict。
- `final_readiness_audit.json` 为 `ready=true`，且 checks 为 `49/49`。
- `final_completion_audit.json` 为 `ready=true`，且 required failures 为 0；`completion_allowed_now=true` 时才可标记目标完成。
- `share_bundle_manifest.json` 为 `ready=true`，推荐发送文件和图表资产都可校验。
- `share_package_validation.json` 为 `ready=true`，且 checks 为 `17/17`。
- `receiver_quickcheck_contract.json` 为 `ready=true`，checks 为 `15/15`，completion-gate docs 为 `6`，WER/ASR docs 为 `3`，stage dictionary crosswalk needles 为 `8`，且 required failures 为 `0`。
- share release seal `ready=true`，`share_release_seal.json` required failures 为 0。
- 如需一命令收包 smoke，可运行 `qwen35_omni_receiver_quickcheck.sh`；它会输出
  `share_package_validation.json`、`share_package_receiver_smoke_validation.json`、
  `share_package_validation_extracted.json` 和 `share_package_external_standalone_validation.json`。
- 如果用便捷 tarball 发送，`.tar.gz.sha256` 能通过 `sha256sum -c`。
- 解包后用随包 validator 运行 extracted-only 校验应为 `13/13`。
- 只用干净 `/tmp` 解包目录和随包 validator 的 standalone 校验应为 `8/8`。
- package asset quality evidence 显示 `report_quality_offenders=[]`、`chart_quality_offenders=[]`。
- 没有把 optional Whisper cache warning 写成 required failure。
- 没有把 vLLM offline diagnostic 写成 online serving parity。
- Do not mark the whole thread goal complete before `final_completion_audit.json` reports `completion_allowed_now=true`.
