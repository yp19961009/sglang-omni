# Qwen3.5-Omni 压力条件总表

状态：证据门已就绪；更新后的目标不再等待 6.21 晚间，后续变更必须重跑 full audit。
工作目录：`/home/gangouyu/sglang-omni`。
用途：把 SGLang-Omni 和 vLLM 的单并发、高并发、短/长文本/语音、反例和诊断条件汇总成
一张人读矩阵，方便合作高校快速复核“每种压力条件下到底结论是什么”。

关联材料：

- 主报告：
  `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_stress_performance_plan_20260621.md`
- 机器 acceptance matrix：
  `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/acceptance_matrix.json`
- stage interaction summary：
  `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/stage_interaction_summary.json`
- 数字来源索引：
  `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_metric_source_map_zh_20260621.md`
- Stage 指标字典：
  `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_stage_metric_dictionary_zh_20260621.md`
- 外部复现 handoff runbook：
  `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_external_handoff_runbook_zh_20260621.md`
- 合作方复跑验收表：
  `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_collaborator_rerun_validation_sheet_zh_20260621.md`
- 优化 playbook：
  `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_optimization_playbook_zh_20260621.md`

## 1. 读表规则

| 状态 | 含义 |
| --- | --- |
| 低并发基线 | 用于确认无明显排队时的基础路径和 tail 形态 |
| 推荐窗口 | 当前 recipe 可作为默认服务窗口的一部分 |
| 吞吐峰值 | 当前 recipe 的最高吞吐点 |
| 压力边界 | 可运行但不建议默认使用，用来暴露饱和瓶颈 |
| 回归保护 | 用于证明短/长输出、WER 或 stage 连接没有退化 |
| 诊断证据 | 用于定位 vLLM 或 runner 的限制，不直接当 online serving 结论 |
| 反例 | 明确不推荐的配置或调参方向 |
| n/a | 不适用或未计入该行结论：低并发 queue estimate 尚未形成、失败反例无稳定性能数字、或 diagnostic 行未计算 WER/ASR；不能把 n/a 当作 0 或当作通过证据 |

## 2. SGLang Video-AMME c=1/2/4/8/16

| Pressure | 状态 | Accuracy | Latency Mean/P95 | RTF Mean/P95 | QPS | WER | 主要瓶颈 | 结论 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| c=1 | 低并发基线 | 70.0% | 1.316s / 2.406s | 1.0490 / 2.0198 | 0.760 | 3.85% | `talker_ar` tail | 低并发基础路径健康 |
| c=2 | 低并发基线 | 70.0% | 1.508s / 3.124s | 1.0816 / 1.9309 | 1.315 | 3.85% | `talker_ar` tail | 仍无明显 admission 压力 |
| c=4 | 推荐窗口 | 70.0% | 1.929s / 3.633s | 1.4015 / 2.4983 | 2.036 | 3.85% | `talker_ar` tail | 稳态高性价比点，横向对比也在 c=4 |
| c=8 | 吞吐峰值 | 70.0% | 3.064s / 5.853s | 2.2141 / 4.3925 | 2.540 | 3.23% | preprocessing admission + talker tail | 当前最高吞吐点 |
| c=16 | 压力边界 | 70.0% | 6.066s / 7.846s | 4.8489 / 10.4087 | 2.407 | 2.88% | queue/admission saturation | 可运行但不推荐默认使用 |

核心解释：

- c=1/2/4 主要看基础 latency 和 talker tail。
- c=8 是当前吞吐峰值。
- c=16 没有质量崩溃，但 QPS 低于 c=8，RTF/tail 明显变差。
- WER 在 c=1/2/4/8/16 下稳定，性能拐点不是质量退化造成的。

## 3. SGLang Stage 压力转移

| Pressure | Top Stage | Preproc Lifecycle Avg/P95 | Actual Preprocess Avg/P95 | Talker Avg/P95 | Decode Avg/P95 | Hop P95 | 判断 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| c=1 | talker AR | 295 / 306ms | n/a | 444 / 1516ms | 14 / 18ms | 15.5ms | talker tail 主导 |
| c=2 | talker AR | 334 / 542ms | n/a | 526 / 2024ms | 16 / 20ms | 16.1ms | talker tail 主导 |
| c=4 | talker AR | 487 / 956ms | 296 / 353ms | 663 / 2628ms | 16 / 22ms | 17.8ms | talker tail + 少量 queue |
| c=8 | preprocessing | 1227 / 2164ms | 289 / 336ms | 983 / 4418ms | 17 / 26ms | 20.4ms | lifecycle 主要是 admission/queue |
| c=16 | preprocessing | 4395 / 5884ms | 305 / 341ms | 816 / 3636ms | 17 / 24ms | 19.7ms | queue/admission 已饱和 |

核心解释：

- preprocessing lifecycle 在 c=8/c=16 变大，但 actual preprocess compute 基本稳定。
- `talker_ar -> code2wav` hop p95 始终约 15-24ms，不是主瓶颈。
- `code2wav_decode` 约 14-17ms/window，不是当前 compute bottleneck。

## 4. 短/长文本输入 + 语音输出

| Scenario | Text Input | c | 状态 | Audio Mean | Latency Mean/P95 | RTF Mean/P95 | QPS | Audio Throughput | 主要瓶颈 | 结论 |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| short | 74 chars / 12 words | 1 | 回归保护 | 4.2s | 0.866s / 0.924s | 0.2052 / 0.2112 | 1.154 | 4.877x | talker short tail | 短文本/短输出基础路径健康 |
| short | 74 chars / 12 words | 4 | 回归保护 | 4.3s | 1.768s / 2.056s | 0.4105 / 0.4388 | 2.218 | 9.561x | talker + batching | 短文本/短输出并发健康 |
| short | 74 chars / 12 words | 8 | 回归保护 | 4.3s | 2.638s / 2.828s | 0.6257 / 0.7440 | 2.983 | 12.738x | talker cadence / collect wait | 仍快于实时 |
| long | 944 chars / 139 words | 1 | 回归保护 | 51.9s | 9.168s / 9.465s | 0.1766 / 0.1776 | 0.109 | 5.663x | talker AR | 长文本/长输出基础路径健康 |
| long | 944 chars / 139 words | 4 | 回归保护 | 52.6s | 17.551s / 18.025s | 0.3338 / 0.3373 | 0.227 | 11.923x | talker AR | 长文本/长输出并发健康 |
| long | 944 chars / 139 words | 8 | 回归保护 | 52.3s | 25.799s / 26.318s | 0.4932 / 0.5001 | 0.303 | 15.870x | talker AR | long c=8 仍快于实时 |

核心解释：

- short/long text-to-speech 覆盖 c=1/4/8；短输入是 74 chars / 12 words，长输入是
  944 chars / 139 words。
- long c=8 平均生成约 52.3s 音频，latency 约 25.8s，RTF 约 0.4932，仍快于实时。
- 长语音主要压力是 talker AR 和 chunk cadence，不是 code2wav decode。

## 5. vLLM 对比和 c=8 诊断

| vLLM Pressure | 状态 | Completed | Runner QPS | Engine QPS | Admission Span Avg/P95 | 主要解释 | 可对外说法 |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| c=4 optimized | 严格横向 baseline | 50/50 | 0.1536 | 0.1536 | 15.11s / 19.14s | offline runner prompt-feed/admission 很重 | warmed c=4 用作公平横向对比 |
| c=8 original | 诊断证据 | 50/50 | 0.1622 | 0.1622 | 33.31s / 43.97s | host prompt build/feed admission limited | 不能直接当 online serving parity |
| c=8 prebuild w1 | 诊断证据 | 50/50 | 0.1420 | 0.5391 | 4.44s / 5.43s | admission span 降低，但 runner 仍慢 | prebuild 能隔离 prompt-feed 问题 |
| c=8 prebuild w4 | 优化 offline diagnostic | 50/50 | 0.2127 | 0.5360 | 4.09s / 4.89s | runner wall 改善，tail 转向 engine/talker | 最强 offline 诊断，不是 online parity |

核心解释：

- vLLM baseline 不是弱 baseline：使用 Qwen3.5-capable 镜像和 compile/CUDA graph 路径。
- vLLM c=8 原始 offline 路径主要受 prompt-feed/admission 限制。
- prebuild w4 是正确的 offline 优化方向，但严格 c=8 online parity 仍需要 online ingress + WER/ASR。

## 6. 反例和不要做的优化

| Case | 状态 | Completed/Failed | QPS | Latency Mean/P95 | 结论 |
| --- | --- | ---: | ---: | ---: | --- |
| preproc=1 c=8 baseline | 推荐基线 | 50/0 | 2.540 | 3.064s / 5.853s | 当前最高吞吐点 |
| preproc=2 c=8 | 反例 | 50/0 | 1.642 | 4.579s / 6.313s | QPS 回退 35.4%，不是优化 |
| preproc=4 c=8 | 反例 | 43/7 | n/a | n/a | OOM/失败请求，不可作为当前 recipe |

核心解释：

- c=8/c=16 的 preprocessing lifecycle 问题不能靠简单加 preprocessing worker 解决。
- 若要重试 preprocessing，需要同时调整 thinker admission、memory fraction 或 preprocessing/encoder placement。

## 7. 最终人读结论

可以对外说：

> 当前压力矩阵覆盖 SGLang Video-AMME c=1/2/4/8/16、短/长 synthetic speech c=1/4/8、
> vLLM c=4/c=8/prebuild 诊断和 preprocessing 反例。证据支持 c=4-c=8 作为推荐
> serving window，c=8 是当前 recipe 的吞吐峰值，c=16 是饱和边界；stage 连接和
> code2wav decode 不是当前主瓶颈。

不要对外说：

> vLLM c=8 online serving parity 已完成，或 c=16 是推荐默认高并发点。

## 8. 复现 gate

完整审计命令：

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
- headline scorecard `9/9`
- acceptance matrix `17/17`
- confidence ledger `12/12`
