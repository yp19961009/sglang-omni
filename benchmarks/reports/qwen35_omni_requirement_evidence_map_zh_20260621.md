# Qwen3.5-Omni 原始需求-证据映射表

状态：证据门已就绪；更新后的目标不再等待 6.21 晚间，后续变更必须重跑 full audit。
工作目录：`/home/gangouyu/sglang-omni`。
用途：把“要给合作高校分享的 SGLang-Omni Qwen3.5 性能分析”原始要求逐项映射到
本地证据、复现入口和置信边界。

建议阅读顺序：

1. 先读本文件，确认每个原始要求都能找到证据。
2. 再读中文简报：
   `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_collaboration_brief_zh_20260621.md`
3. 查每种压力条件的推荐状态和证据路径时读 pressure matrix：
   `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_pressure_matrix_zh_20260621.md`
4. 查 headline/stage/vLLM 诊断数字来源时读 metric source map：
   `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_metric_source_map_zh_20260621.md`
5. 查 stage lifecycle、compute、handoff 语义时读 stage metric dictionary：
   `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_stage_metric_dictionary_zh_20260621.md`
6. 准备现场问答时读 defense Q&A：
   `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_defense_qna_zh_20260621.md`
7. 继续调优或说明当前最优 recipe 时读 optimization playbook：
   `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_optimization_playbook_zh_20260621.md`
8. 做 PPT 时读 deck 提纲：
   `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_share_deck_outline_zh_20260621.md`
9. 给外部 reviewer 最短验收路径时读 handoff runbook：
   `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_external_handoff_runbook_zh_20260621.md`
10. 合作方复跑完成后读 rerun validation sheet：
   `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_collaborator_rerun_validation_sheet_zh_20260621.md`
11. 复现每条命令时读 checklist：
   `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_reproduction_checklist_zh_20260621.md`
12. 需要完整细节时读主报告：
   `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_stress_performance_plan_20260621.md`

## 1. 总体结论和边界

当前证据支持的对外主结论：

- 在本地 8x NVIDIA H20、Video-AMME ci-50 语音输出 workload、warmed c=4
  严格横向对比中，优化后的 SGLang-Omni Qwen3.5 在 latency mean、latency
  p95、RTF mean、RTF p95 上优于优化版 vLLM，accuracy/WER 不退化。
- SGLang-Omni 的推荐 serving window 是 c=4-c=8；c=8 是当前 recipe 的吞吐峰值，
  c=16 是 admission/queueing 饱和边界，不建议作为默认运行点。
- Stage breakdown 显示当前优先瓶颈是 talker AR tail、c=8/c=16 admission/queueing，
  而不是 `code2wav_decode` compute。
- Stage 连接证据显示 `talker_ar -> code2wav` hop 健康；当前没有证据表明
  stage handoff 是主瓶颈。
- vLLM 边界要讲清楚：c=4 是严格 warmed 横向对比；c=8 prebuild w4 是 offline
  diagnostic，不是 online serving parity。
- vLLM c=8 prebuild w4 是强 offline diagnostic，用来证明 vLLM 原始 c=8
  主要受 prompt build/feed admission 限制；它仍不是严格 online serving parity。

当前必须保留的置信边界：

- 不把官方 SeedTTS full-set 当 headline；本地是 Video-AMME spoken-reference
  SeedTTS-compatible smoke path。
- 不把 vLLM c=8 prebuild w4 说成 online serving QPS 对比。
- 不声称 c=16 是推荐高并发点。
- 不把 naive preprocessing 并发放大说成优化；preproc=2 回退、preproc=4 OOM。

## 2. 原始要求到证据

| 原始要求 | 当前回答 | 最强证据 | 复现/审计入口 | 置信度 |
| --- | --- | --- | --- | --- |
| “性能优化得比较好，至少和 vLLM 相当” | warmed c=4 严格对比中，SGLang-Omni latency/RTF 全部优于优化版 vLLM，accuracy/WER 不退化。 | 完整报告 section 1/5；`claims_verification.json`；`headline_scorecard.json` | `run_qwen35_omni_report_audit`；复现清单 section 1/5 | High |
| “vLLM baseline 也要性能比较好” | vLLM 使用 Qwen3.5-capable 镜像、compile mode、`FULL_AND_PIECEWISE` CUDA graph、talker code predictor graph；c=8 还补了 prebuild w4 offline diagnostic。 | 完整报告 section 4.1/5/11.7；`vllm_admission_diagnosis.json` | vLLM wrapper + `EXTRA_ARGS="--prebuild-prompts --prebuild-workers 4"` | High for optimized baseline；Medium for strict c=8 online parity |
| “单并发和高并发都要有” | SGLang 覆盖 c=1/2/4/8/16；vLLM 覆盖 c=1/c=4/c=8 诊断。c=8 是 SGLang 吞吐峰值，c=16 是压力边界。 | 完整报告 section 7/12；`tables_summary.json`；`acceptance_matrix.json` | 复现清单 section 2/7 | High |
| “长短文都有” | 本报告覆盖 short/long text-to-speech 的 c=1/4/8：短文本为 74 chars / 12 words，长文本为 944 chars / 139 words；long c=8 仍快于实时。Video-AMME spoken-reference smoke path 用于自然语音复核。 | 完整报告 section 8/8.3；`tables_summary.json`；`videoamme_seedtts_meta_summary.json` | 复现清单 section 3；`build_videoamme_seedtts_meta` | High for synthetic short/long；Medium for broader real traffic |
| “breakdown 很具体” | 主报告拆出 preprocessing、talker、code2wav stage、decode、window collect、talker-to-code2wav hop、vLLM log stage/admission。 | 完整报告 section 5/7.1/8.1/12.1；`stage_interaction_summary.json` | `summarize_qwen35_stage_interactions`；`summarize_qwen35_omni_report_artifacts` | High |
| “每个 stage 的性能都要分析” | SGLang Video-AMME、synthetic speech、vLLM log-derived stages 都有表格和机器摘要；c=1/c=4 talker tail、c=8/c=16 queue/admission、code2wav 非瓶颈均有解释。 | 完整报告 section 7.1/8.1/12；`tables_summary.json` | coverage matrix；stage interaction summary | High |
| “stage 之间有没有瓶颈、相互影响” | `talker_ar -> code2wav` hop p95 在 SGLang 压测中保持约 15-24ms；`code2wav_window_collect` 增长主要是等待 talker codec chunk；vLLM 原始 c=8 是 prompt feed/admission 限制。 | 完整报告 section 12.1；`stage_interaction_summary.json` | `summarize_qwen35_stage_interactions` | High |
| “最后达到性能最优” | 当前最优推荐 recipe 是 SGLang c=4-c=8 window、保留 CUDA graph/compile/code2wav compile，避免 naive preproc 并发；c=16 和 preproc=2/4 是反例。 | 完整报告 section 4.1/7/9/12/14；`acceptance_matrix.json` | 复现清单 section 7 | High for measured recipe；not a global optimum proof |
| “能根据文档复现 SGLang 和 vLLM 性能” | 外部 handoff runbook 给出最短验收路径；中文 checklist 给出 full audit、SGLang serving/stress、WER、vLLM c=4/c=8/prebuild w4、表格再生成和最终 gate；rerun validation sheet 给出复跑后是否可替换数字的判断表。 | handoff runbook；复现清单；复跑验收表；完整报告 section 11；`preflight_repro.json`；`manifest.json` | `python3 -m benchmarks.eval.run_qwen35_omni_report_audit ...` | High |
| “要专业并且置信” | 机器证据包含 claims verifier、coverage matrix、headline scorecard、acceptance matrix、confidence ledger；对外话术分 High/Medium/Unsupported。 | `confidence_ledger.json`；分享包索引；中文简报 section 8 | `build_qwen35_omni_confidence_ledger` | High for stated claims |

## 3. 单并发和高并发

单并发重点：

- SGLang c=1：用于确认无并发排队时的基础 latency/RTF 和 talker tail。
- vLLM c=1：用于确认 cross-runtime 低并发下不是只靠 SGLang 高并发调度获胜。
- 当前可说：SGLang 在单并发 request latency/RTF 上明显优于 vLLM，且生成一致性稳定。

高并发重点：

- SGLang c=4：最干净的 warmed cross-runtime headline comparison。
- SGLang c=8：当前 recipe 的吞吐峰值。
- SGLang c=16：证明 admission/queueing 进入饱和，不能作为默认推荐。
- vLLM c=8：原始 offline runner 主要受 prompt build/feed admission 限制；
  prebuild w4 改善 runner wall，但还需要 online ingress + WER/ASR 才能做严格
  serving-throughput parity。

## 4. 短/长文本输入 + 语音输出

短输出用于看低生成长度下的 fixed cost、talker tail 和 handoff；长输出用于看
持续 codec chunk 生成、window collect、code2wav decode 和端到端 RTF。

当前结论：

- short/long text-to-speech 均覆盖 c=1/4/8；短文本为 74 chars / 12 words，长文本为
  944 chars / 139 words。
- long c=8 仍快于实时，说明长语音不是 code2wav decode compute 被打爆。
- 长语音的主要压力仍回到 talker AR cadence 和并发 admission，而不是 vocoder。

## 5. Stage breakdown 和 stage 连接

Stage breakdown 读法：

- `preprocessing`：高并发下容易被 admission/queueing 放大，naive 并发放大不是解法。
- `talker_ar`：c=1/c=4 与 long synthetic 的主要 tail 来源。
- `code2wav_window_collect`：随并发增长通常是在等 talker codec chunk，不等于 decode 慢。
- `code2wav_decode`：当前约十几毫秒量级，不是 compute bottleneck。
- `talker_ar -> code2wav` hop：当前 SGLang 连接健康，不是主瓶颈。

Stage 之间的相互影响：

- talker cadence 变慢会让 `code2wav_window_collect` 看起来变长。
- admission/queueing 变重会抬高 preprocessing lifecycle，但不是 HF processor 单点 compute。
- vLLM prompt-feed/admission 过慢会掩盖 engine 内部 stage；prebuild 之后才看得到
  engine/workload/talker-side tail。

## 6. 复现路径

完整审计：

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
- stage reproduction drilldown `ready=true`
- stage route decision matrix `ready=true`
- objective requirement crosswalk `ready=true`，含 8 条 optimization candidate verdict
- headline scorecard `9/9`
- acceptance matrix `17/17`
- confidence ledger `12/12`

核心证据 JSON：

- `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/audit_run_summary.json`
- `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/coverage_matrix.json`
- `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/claims_verification.json`
- `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/headline_scorecard.json`
- `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/objective_requirement_crosswalk.json`
- `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/acceptance_matrix.json`
- `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/confidence_ledger.json`
- `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/stage_interaction_summary.json`
- `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/vllm_admission_diagnosis.json`

## 7. 答辩时的一句话版本

可以说：

> 在本地 8x H20、Video-AMME ci-50 语音输出 workload 上，我们不是只给了
> headline 数字，而是把 SGLang-Omni 和优化版 vLLM 的 c=4 严格对比、SGLang
> c=1/2/4/8/16 压测、短/长文本输入 + 语音输出、stage breakdown、stage 连接、反例和复现
> gate 都串成了一个可审计证据包；当前高置信结论是 SGLang-Omni 至少与优化版
> vLLM 相当，并在 warmed c=4 严格对比中更快。

不要说：

> vLLM c=8 online serving parity 已经被证明。

更准确的说法：

> vLLM c=8 offline prompt-feed 瓶颈已经被 prebuild w4 诊断和缓解；严格 c=8
> online serving parity 还需要 online ingress 加 WER/ASR 复核。
