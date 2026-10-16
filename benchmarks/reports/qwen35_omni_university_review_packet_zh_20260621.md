# Qwen3.5-Omni 高校合作方审阅会议包

生成时间 UTC：`2026-06-21T02:00:27.210613+00:00`。
工作目录：`/home/gangouyu/sglang-omni`。

用途：给合作高校第一次审阅或 15 分钟同步会使用。它不替代完整技术报告，
而是把结论、边界、复现入口、stage 追问路线和复跑替换条件压缩到一页半。

## 1. 会前先跑

```bash
export HOST_REPO="${HOST_REPO:-/home/gangouyu/sglang-omni}"
cd "$HOST_REPO"
sha256sum -c results/qwen35_report_audit_20260619/qwen35_omni_share_bundle_20260621.tar.gz.sha256
bash benchmarks/eval/qwen35_omni_receiver_quickcheck.sh
bash benchmarks/eval/qwen35_omni_evidence_query_cards_smoke.sh --root "$HOST_REPO" --mode host
```

期望：release seal、tarball validation、receiver smoke、extracted-only、standalone
和 evidence query smoke 都通过；更新后的目标不再等待 6.21 晚上，
`completion_allowed_now=true` 表示可以进入最终完成裁决。

## 2. 15 分钟同步节奏

| 时间 | 讲什么 | 直接打开 |
| --- | --- | --- |
| 0-2 min | 包是否可读、checksum/quickcheck 是否绿 | `qwen35_omni_start_here_zh_20260621.md` |
| 2-4 min | 先统一术语：c、warmed、RTF、queue estimate、offline diagnostic | `qwen35_omni_university_technical_report_zh_20260621.md` section 2.1 |
| 4-6 min | strict warmed c=4 headline 与 vLLM baseline 公平性 | `qwen35_omni_university_technical_report_zh_20260621.md` |
| 6-8 min | c=1/4/8/16、短/长文本、RTF 与 serving 窗口 | `qwen35_omni_regime_decision_matrix_zh_20260621.md` |
| 8-11 min | stage breakdown、handoff、queue/admission 与瓶颈迁移 | `qwen35_omni_stage_boundary_bottleneck_ledger_zh_20260621.md` |
| 11-13 min | vLLM c=8 prebuild w4 为什么仍是 offline diagnostic | `qwen35_omni_vllm_online_parity_protocol_zh_20260621.md` |
| 13-15 min | 复跑预算、替换 headline 条件和差异 triage | `qwen35_omni_rerun_acceptance_contract_zh_20260621.md` |

## 3. 可直接相信的结论

- Final readiness：`49/49`，required failures `0`。
- Strict headline scope：`warmed c=4 only`；vLLM c=8 scope：`offline_diagnostic_not_online_parity`。
- SGLang 推荐运行窗口：`c4-c8`；饱和边界：`c16`。
- 复跑计时段下界：`1592.84s`；8-GPU 等效下界：`3.54` GPU-hours。

| 结论 | 当前数字 | 证据 |
| --- | --- | --- |
| Strict warmed c=4 | SGLang latency mean `1.743s`, vLLM `2.093s`; SGLang lower by `16.7%` | headline_scorecard.json / strict_c4_comparison |
| Strict c=4 tail/RTF | SGLang RTF p95 `2.402`, vLLM `3.072`; SGLang lower by `21.8%` | headline_scorecard.json / relative_sglang_lower_pct |
| SGLang high-concurrency edge | c=8 throughput `2.540 qps`, latency p95 `5.853s`; c=16 remains the saturation boundary | headline_scorecard.json / sglang_stress.throughput_peak |
| Long-form speech guard | long c=8 target `944` chars / `139` words, RTF p95 `0.5001` | headline_scorecard.json / synthetic_long_c8 |

## 4. 不能越界的说法

- 不把 vLLM c=8 prebuild w4 写成 online serving parity；当前只是优化后的 offline diagnostic。
- 不把 c=16 写成默认推荐；它是 saturation boundary，用来解释 queue 和 tail risk。
- 不在未复跑同硬件、同镜像、同模型/cache、同 ASR/WER 链路且门禁全绿前替换 headline 数字。
- ci-50/stress/synthetic 证据不能直接外推到完整线上流量；更大 Video-AMME 或真实线上流量需要同口径复跑。
- 不说已经搜索完所有 future kernel 或全局最优；当前是 measured-best recipe。

## 5. 追问时的证据路线

| 追问 | 先看 | 再查机器证据 |
| --- | --- | --- |
| c、warmed、RTF、queue estimate 怎么定义 | `qwen35_omni_university_technical_report_zh_20260621.md` section 2.1 | `university_technical_report.json` |
| vLLM baseline 是否公平 | `qwen35_omni_runtime_image_contract_zh_20260621.md` | `runtime_image_contract.json`, `runtime_comparison_contract.json` |
| 为什么 headline 选 c=4 | `qwen35_omni_runtime_comparison_contract_zh_20260621.md` | `headline_scorecard.json` |
| 单并发/高并发/长短文本是否覆盖 | `qwen35_omni_length_regime_coverage_zh_20260621.md` | `length_regime_coverage.json` |
| stage 之间是否卡住 | `qwen35_omni_stage_causal_graph_zh_20260621.md` | `stage_causal_graph.json`, `stage_interaction_summary.json` |
| 哪个 stage 是瓶颈 | `qwen35_omni_stage_boundary_bottleneck_ledger_zh_20260621.md` | `stage_boundary_bottleneck_ledger.json` |
| 合作方复跑不同怎么办 | `qwen35_omni_rerun_delta_triage_zh_20260621.md` | `rerun_delta_triage.json` |
| 现场如何自证 | `qwen35_omni_evidence_query_cards_zh_20260621.md` | `qwen35_omni_evidence_query_cards_smoke.sh` |

## 6. 复跑替换边界

- 可以复跑并替换 headline 的前提：同 8x H20、同 SGLang/vLLM 镜像、同模型/cache、同数据口径、同 warmup/skip-first、同 ASR/WER 验收。
- 必须返回 `rerun_acceptance_contract` 要求的 34 份 evidence 和 27 行命令矩阵。
- 只重跑 raw benchmark、没有重建表格/图表/JSON/full audit 时，不替换主报告数字。
- vLLM c=8 要升级为 strict online parity，必须补 online ingress、同口径 WER/ASR 和 vLLM online parity protocol 的 artifact。

## 7. 机器 Gate 摘要

| Gate | Status | Evidence |
| --- | --- | --- |
| full audit and readiness are green | `PASS` | Final readiness `49/49`, required failures `0`. |
| share package receiver path is validated | `PASS` | release=14/14, tarball=17/17, receiver=17/17, extracted=13/13, standalone=8/8, tarball_pending_in_full_audit=False |
| headline claims are evidence-backed | `PASS` | headline scorecard `9/9`; strict c4 win, c8 peak, long c8 RTF and vLLM diagnostic checks are all green. |
| runtime fairness and vLLM baseline strength are locked | `PASS` | runtime image `12/12`; comparison `9/9`; headline scope `warmed c=4 only`; vLLM c8 `offline_diagnostic_not_online_parity`. |
| review glossary and metric-scope route is present | `PASS` | technical_report=benchmarks/reports/qwen35_omni_university_technical_report_zh_20260621.md; missing_glossary_terms=[] |
| stage and boundary diagnosis are green | `PASS` | stage latency `12/12`, boundary `12/12`, causal graph `7/7`; ledger rows `37`, recommended window `c4-c8`. |
| short/long and high-concurrency regimes are covered | `PASS` | 7 regime rows; short `3`, long `3`; long c8 RTF p95 `0.5001`. |
| rerun budget and replacement contract are explicit | `PASS` | budget rows `9`, timed wall lower bound `1592.84s`; acceptance `17/17`, return evidence `34`. |
| rerun command references resolve | `PASS` | commands `63`, structured refs `2229`, unresolved `0`. |
| receiver quickcheck and defense path are ready | `PASS` | receiver contract `15/15` with `6` steps; defense matrix `17/17`, question rows `13`. |
| caveat and vLLM c8 parity boundaries are explicit | `PASS` | caveats `12`; current-best scope `measured_best_not_global_optimum`; vLLM online parity proven `False`. |
| final completion is evidence-gated | `PASS` | completion_allowed_now `True`; blockers `[]`. |
| review reading route files are present | `PASS` | missing_docs=[] |
| share and evidence manifests include the reviewable package | `PASS` | share bundle ready `True`, records `122`; manifest records `196`, missing `0`. |

## 8. 当前完成门

- completion_allowed_now：`True`
- completion_blockers：`[]`
- 不再按 2026-06-21 18:00 UTC+08:00 等待；按 final completion audit 的证据门决定是否标记 goal complete。

机器证据：`results/qwen35_report_audit_20260619/university_review_packet.json`。
