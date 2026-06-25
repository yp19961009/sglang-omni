# Qwen3.5-Omni 性能优化 Playbook

状态：证据门已就绪；更新后的目标不再等待 6.21 晚间，后续变更必须重跑 full audit。
工作目录：`/home/gangouyu/sglang-omni`。
用途：把当前性能证据转成可执行 tuning 协议，说明“当前最优 recipe 是什么、下一步先动什么、
哪些 knob 暂时不要动、每次调参如何验收和回滚”。

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
- 答辩 Q&A：
  `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_defense_qna_zh_20260621.md`
- 复现清单：
  `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_reproduction_checklist_zh_20260621.md`
- 外部复现 handoff runbook：
  `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_external_handoff_runbook_zh_20260621.md`
- 合作方复跑验收表：
  `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_collaborator_rerun_validation_sheet_zh_20260621.md`

## 1. 当前推荐 recipe

当前对外推荐的 SGLang-Omni Qwen3.5 recipe 是保守但强证据的 c=4-c=8 serving
window：

| 组件 | 当前推荐 | 原因 | 验收证据 |
| --- | --- | --- | --- |
| Thinker/Talker graph | 保持 `--thinker-cuda-graph on`、`--talker-cuda-graph on`、`--talker-torch-compile on` | warmed c=4 下 latency/RTF 优于优化版 vLLM | `headline_scorecard.json` |
| Code2wav | 保持 `NO_CODE2WAV_TORCH_COMPILE=0`、`TORCHDYNAMO_DISABLE=0` | decode 约 14-17ms/window，不是瓶颈，但 compile 路径稳定 | `stage_interaction_summary.json` |
| 并发窗口 | 推荐 c=4-c=8，c=8 是吞吐峰值 | c=8 QPS 2.540，c=16 回落到 2.407 且 tail 变差 | `acceptance_matrix.json` |
| Preprocessing | 保持 `PREPROCESSING_MAX_CONCURRENCY=1` | preproc=2 QPS 回退 35.4%，preproc=4 OOM/失败 | 完整报告 section 9 |
| WER/质量 | serving 压测后独立跑 WER/ASR | 避免 ASR 与 serving 抢 GPU，且保留 audio consistency evidence | `claims_verification.json` |

一句话版本：

> 当前最优不是“把所有并发和 worker 拉满”，而是保留 graph/compile/code2wav 稳定路径，
> 把 serving window 控制在 c=4-c=8，并优先优化 talker AR 与 admission 策略。

## 2. 不要先动什么

这些 knob 当前有反例或不该作为第一优化方向：

| 不建议动作 | 为什么 | 何时才能重试 |
| --- | --- | --- |
| 不要先优化 code2wav decode | 当前 decode 约 14-17ms/window，`talker_ar -> code2wav` hop p95 约 15-24ms；code2wav collect 变长多数是在等 codec chunk | 只有当 decode p95 独立升高、且 collect-minus-decode 不再解释 tail 时 |
| 不要把 c=16 作为默认 serving 点 | c=16 QPS 低于 c=8，RTF 和 preprocessing queue tail 变差 | 有新的 admission/batching 策略后重新压测 |
| 不要单独放大 `PREPROCESSING_MAX_CONCURRENCY` | preproc=2 回退，preproc=4 OOM/失败，说明资源争用不是 worker 数不足 | 同时改变 thinker admission、memory fraction 或 preprocessing/encoder placement 后 |
| 不要把 vLLM c=8 prebuild w4 当 online parity | 它是 offline diagnostic，未覆盖 online ingress 和 WER/ASR | 加在线 ingress + WER/ASR 后再声明 strict parity |
| 不要把 synthetic long 当官方 SeedTTS full-set | 它是长语音 guardrail，不是自然语音 full-set headline | 预置官方 SeedTTS 或同等自然语音 full-set 后 |

## 3. Stage 到 knob 的决策表

| 观察到的现象 | 先看哪个证据 | 优先 knob | 禁止误判 |
| --- | --- | --- | --- |
| c=1/c=4 latency tail 高 | request profile 的 `talker_ar` avg/p95 | talker AR、batching、chunk cadence | 不要先改 preprocessing |
| c=8 QPS 到峰值但 tail 抬升 | preprocessing lifecycle vs actual preprocess compute | admission 策略、max running request 边界、batch smoothness | 不要直接把 preproc worker 加大 |
| c=16 QPS 回落 | `acceptance_matrix.json` 与 tail appendix | 降低 admission、分片或更细批控 | 不要把 c=16 作为默认 |
| `code2wav stage` 看起来变大 | compare `code2wav_window_collect`、`code2wav_decode`、hop | talker codec cadence；保持 decode compile | 不要把 collect wait 当 decode compute |
| vLLM c=8 wall QPS 慢 | `vllm_admission_diagnosis.json` | prompt prebuild、online ingress 分离、engine/talker tail 分析 | 不要直接比较 offline wall QPS |
| WER 变差 | `whisper_large_v3_wer.json`、audio consistency | 回滚最近 generation/talker 相关改动 | 不要只看 QPS |

## 4. 下一轮优化实验顺序

### P0：锁住当前可分享 recipe

目标：确保任何改动前都有稳定 baseline。

动作：

1. 固定当前 SGLang c=4-c=8 recipe。
2. 跑 full audit。
3. 保存 manifest 和 audit summary。

验收：

- full audit `ok=true`
- claims `17/17`
- coverage `34/34`
- preflight `62` checks, `0` required failures
- manifest current `196` records, minimum `180`, `0` missing
- SGLang optimization lock `26/26`
- repro command manifest `63` commands / `7` phases

### P1：c=8 admission 微调

目标：降低 c=8 queue tail，同时不牺牲 c=8 吞吐峰值。

建议实验：

| 实验 | 成功条件 | 回滚条件 |
| --- | --- | --- |
| c=8 附近更平滑 admission | QPS 不低于 2.540 的 97%，latency p95 或 preprocessing lifecycle p95 下降 | QPS 下降超过 3%，WER 变差，或 c=4 latency 回退 |
| thinker admission 降低或更细批控 | c=8 tail 下降且 c=4 不退化 | c=8 QPS 明显下降或 queue 转移到 talker |
| c=6/c=8 对照 | 找到比 c=8 更稳的服务窗口 | c=6 没有 tail 收益或吞吐明显不足 |

必须复核：

- `stage_interaction_summary.json`
- `acceptance_matrix.json`
- WER/audio consistency

### P1：talker AR 和 chunk cadence

目标：改善单并发、c=4、long speech 的共同主因。

建议实验：

| 实验 | 成功条件 | 回滚条件 |
| --- | --- | --- |
| Talker AR kernel/compile path 复核 | c=1/c=4 talker p95 降低，WER 不变差 | WER 变差或 long speech RTF 回退 |
| chunk cadence / window collect 调整 | `code2wav_window_collect` 下降，decode p95 不上升 | 音频质量下降或 hop p95 上升 |
| batching 策略微调 | c=4/c=8 latency p95 下降，QPS 不回退 | c=1 latency 明显变差 |

### P2：preprocessing placement/admission 联合调参

目标：解决 c=8/c=16 preprocessing lifecycle 里 queue/admission 放大的部分。

安全前提：

- 不单独把 `PREPROCESSING_MAX_CONCURRENCY` 从 1 改到 2/4。
- 必须同时约束 thinker admission 或调整 preprocessing/encoder placement。
- 必须监控 media load、image encoder lifecycle、thinker lifecycle。

成功条件：

- c=8 QPS 接近或高于 2.540。
- preprocessing lifecycle p95 下降。
- actual preprocessing compute 不显著上升。
- WER 不变差。

### P2：vLLM online ingress + WER/ASR

目标：把 vLLM c=8 从 offline diagnostic 推进到更严格 serving comparison。

动作：

1. 保留现有 vLLM optimized image 和 prebuild w4 artifact。
2. 新增 online ingress 路径，避免把 prompt build/feed 当 engine serving。
3. 补 WER/ASR。
4. 复查 engine/talker boundary tail。

可说边界：

- 完成前只能说“vLLM c=8 prebuild w4 是强 offline diagnostic”。
- 完成后才讨论 strict c=8 online serving parity。

## 5. 每次调参后的验收命令

先跑 full audit：

```bash
cd /home/gangouyu/sglang-omni

python3 -m benchmarks.eval.run_qwen35_omni_report_audit \
  --root /home/gangouyu/sglang-omni \
  --summary-output results/qwen35_report_audit_20260619/audit_run_summary.json
```

单独复核关键 JSON：

```bash
python3 -m benchmarks.eval.summarize_qwen35_stage_interactions \
  --root /home/gangouyu/sglang-omni \
  --json-output results/qwen35_report_audit_20260619/stage_interaction_summary.json

python3 -m benchmarks.eval.build_qwen35_omni_acceptance_matrix \
  --root /home/gangouyu/sglang-omni \
  --json-output results/qwen35_report_audit_20260619/acceptance_matrix.json

python3 -m benchmarks.eval.build_qwen35_omni_confidence_ledger \
  --root /home/gangouyu/sglang-omni \
  --json-output results/qwen35_report_audit_20260619/confidence_ledger.json
```

## 6. 回滚规则

任何新优化触发以下条件之一，都不进入对外报告 headline：

- warmed c=4 SGLang latency mean/p95 或 RTF mean/p95 不再优于 vLLM。
- c=8 不再是当前 recipe 的吞吐峰值，且没有新的更好 serving window 证据。
- WER/audio consistency 变差。
- `code2wav_decode` 或 stage handoff 真的变成主瓶颈，但没有解释和修复。
- preflight、coverage、manifest、claims 任一 required gate 失败。
- 新结论无法进入 `confidence_ledger.json` 的 high 或 bounded medium 类别。

## 7. 对外话术

可以说：

> 当前最优 recipe 已经由 stress sweep、stage breakdown、anti-recipe 和机器 audit 共同约束：
> c=4-c=8 是推荐窗口，c=8 是吞吐峰值，c=16 是饱和边界；下一步优化应该优先看
> talker AR、chunk cadence 和 admission 策略，而不是先改 code2wav 或盲目加 preprocessing
> 并发。

不要说：

> 我们已经证明了所有可能配置下的全局最优。

更准确的说法：

> 当前 recipe 是本地 8x H20、Video-AMME ci-50、Qwen3.5-Omni 语音输出 workload 下的
> 证据最强推荐点；进一步优化必须按 playbook 逐项复核并重新进入 audit。
