# Qwen3.5-Omni 一页式核心数字 Scorecard

状态：证据门已就绪；更新后的目标不再等待 6.21 晚间，后续变更必须重跑 full audit。
工作目录：`/home/gangouyu/sglang-omni`。
用途：给合作高校快速扫一眼当前 headline、压力条件、stage 结论和复现 gate；完整解释仍以主报告为准。

关联材料：

- 主报告：
  `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_stress_performance_plan_20260621.md`
- 最终分享交付说明：
  `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_final_share_delivery_note_zh_20260621.md`
- 数字来源索引：
  `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_metric_source_map_zh_20260621.md`
- 压力条件总表：
  `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_pressure_matrix_zh_20260621.md`
- Stage 指标字典：
  `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_stage_metric_dictionary_zh_20260621.md`
- Stage latency budget：
  `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_stage_latency_budget_zh_20260621.md`
- Stage boundary bottleneck ledger：
  `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_stage_boundary_bottleneck_ledger_zh_20260621.md`
- Final checkpoint watchlist：
  `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_final_checkpoint_watchlist_zh_20260621.md`

## 1. Headline

在 8x NVIDIA H20、本地 Video-AMME ci-50、语音输出 workload、warmed c=4
严格横向对比中，优化后的 SGLang-Omni Qwen3.5 在 latency/RTF 上优于优化版 vLLM，
accuracy/WER 不退化。

| Runtime | Scope | n | Accuracy | Latency Mean | Latency P95 | RTF Mean | RTF P95 | WER |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vLLM optimized | Video-AMME ci-50, c=4, skip first 4 | 46 | 63.0% | 2.093s | 3.525s | 1.4677 | 3.0717 | 7.44% |
| SGLang-Omni optimized | Video-AMME ci-50, c=4, skip first 4 | 46 | 67.4% | 1.743s | 3.328s | 1.3536 | 2.4023 | 4.12% |

相对 vLLM，SGLang-Omni warmed c=4 mean latency 低 16.7%，p95 latency 低 5.6%，
mean RTF 低 7.8%，p95 RTF 低 21.8%。

## 2. SGLang 压力条件

| 条件 | 结论 | QPS | Latency Mean/P95 | RTF Mean/P95 | WER | 主要解释 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| c=1 | 低并发基线 | 0.760 | 1.316s / 2.406s | 1.0490 / 2.0198 | 3.85% | talker AR tail |
| c=2 | 低并发扩展 | 1.315 | 1.508s / 3.124s | 1.0816 / 1.9309 | 3.85% | 仍主要是 talker AR |
| c=4 | 推荐窗口 | 2.036 | 1.929s / 3.633s | 1.4015 / 2.4983 | 3.85% | 稳态高性价比点 |
| c=8 | 吞吐峰值 | 2.540 | 3.064s / 5.853s | 2.2141 / 4.3925 | 3.23% | preprocessing admission + talker tail |
| c=16 | 压力边界 | 2.407 | 6.066s / 7.846s | 4.8489 / 10.4087 | 2.88% | queue/admission saturation，不推荐默认 |

推荐窗口：c=4 到 c=8。c=8 是当前 recipe 的吞吐峰值；c=16 用于暴露饱和，不作为默认服务点。

## 3. 短/长文本输入 + 语音输出

| Workload | Text Input | c | Audio Mean | Latency Mean/P95 | RTF Mean/P95 | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| short text-to-speech | 74 chars / 12 words | 1 | 4.23s | 0.866s / 0.924s | 0.2052 / 0.2112 | 低 fixed cost 路径健康 |
| short text-to-speech | 74 chars / 12 words | 4 | 4.31s | 1.768s / 2.056s | 0.4105 / 0.4388 | c=4 仍快于实时 |
| short text-to-speech | 74 chars / 12 words | 8 | 4.27s | 2.638s / 2.828s | 0.6257 / 0.7440 | c=8 仍快于实时 |
| long text-to-speech | 944 chars / 139 words | 8 | 52.33s | 25.799s / 26.318s | 0.4932 / 0.5001 | 长文本/长语音 c=8 仍快于实时 |

长文本输入会自然拉长语音输出；当前瓶颈主要是 talker AR，不是 code2wav decode。

## 4. Stage 结论

| 结论 | 核心数字 | 证据 |
| --- | --- | --- |
| `talker_ar -> code2wav` handoff 健康 | SGLang hop p95：c=1 15.5ms，c=2 16.1ms，c=4 17.8ms，c=8 20.4ms，c=16 19.7ms | `stage_interaction_summary.json` |
| `code2wav_decode` 不是主要 compute bottleneck | decode avg 约 14-17ms/window | `claims_verification.json`；stage breakdown |
| 高并发变慢主要是 admission/queueing + talker tail | c=8/c=16 preprocessing lifecycle 增长，但 actual preprocess compute 约 0.29-0.30s | `stage_interaction_summary.json` |
| naive preprocessing 并发不是优化方向 | preproc=2 QPS 1.642 vs baseline c=8 QPS 2.540；preproc=4 OOM/失败 | `acceptance_matrix.json` |

读表原则：`stage_input_received->stage_complete` 是 lifecycle，可能包含 queue/admission；
`preprocess_start->preprocess_end` 和 `code2wav_decode_start->code2wav_decode_end`
才更接近 actual compute。

## 5. vLLM c=8 诊断

| 条件 | Runner QPS | Engine QPS | Admission Span Avg/P95 | 结论 |
| --- | ---: | ---: | ---: | --- |
| vLLM original c=8 | 0.1622 | 0.1622 | 33.3s / 44.0s | offline prompt build/feed admission limited |
| vLLM c=8 prebuild w1 | 0.1420 | 0.5391 | 4.44s / 5.43s | prompt feed 被移除，tail 转向 engine/workload |
| vLLM c=8 prebuild w4 | 0.2127 | 0.5360 | 4.09s / 4.89s | 当前最强 offline diagnostic |

vLLM c=8 `--prebuild-prompts --prebuild-workers 4` 是合理的 offline diagnostic，
不是 online serving parity；严格 c=8 serving parity 仍需 online ingress + WER/ASR。

## 6. 当前机器 Gate

- full audit `ok=true`
- claims `17/17`
- coverage `34/34`
- preflight `62` checks, `0` required failures
- manifest current `196` records, minimum `180`, `0` missing
- headline scorecard `9/9`
- acceptance matrix `17/17`
- confidence ledger `12/12`，high `9`、medium `3`、unsupported `0`
- SGLang optimization lock `26/26`
- vLLM optimization lock `22/22`
- vLLM online parity protocol `18/18`，`online_parity_proven=false`
- runtime image contract `12/12`
- rerun acceptance contract `17/17`，18 rules，34 return evidence files，27 command matrix rows，matrix complete，silent-replacement/protocol-drift guard documented
- final checkpoint watchlist `24/24`，`final_completion_evidence_ready=true`，`checkpoint_phase=completion_audit_ready` 且 `completion_allowed_now=true`
- stage latency budget `12/12`
- stage boundary bottleneck ledger `12/12`，11 条 pressure transition rows
- repro command manifest `63` commands / `7` phases
- final readiness audit `ready=true`，49/49 checks，0 required failures
- share bundle manifest `ready=true`

唯一已知 warning：host 侧 `/root/.cache/whisper/large-v3.pt` 不存在。它是 optional，
只影响 host 侧直接重算 offline WER；容器内权重或 ASR router 可替代。

## 7. 一句话

当前可以高置信对外说：在 8x H20、本地 Video-AMME ci-50 speech-output
workload、warmed c=4 严格对比上，优化后的 SGLang-Omni Qwen3.5 比优化版 vLLM
更快且 WER/accuracy 不退化；SGLang c=8 是当前吞吐峰值；主要瓶颈在
admission/queueing 和 talker AR tail，不在 code2wav decode 或 stage handoff。
