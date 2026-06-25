# Qwen3.5-Omni Stage 指标字典

状态：证据门已就绪；更新后的目标不再等待 6.21 晚间，后续变更必须重跑 full audit。
工作目录：`/home/gangouyu/sglang-omni`。
用途：解释 SGLang-Omni / vLLM stage breakdown 中每个时间指标的语义、能说明什么、
不能说明什么，避免把 queue、compute、handoff、collect wait 混成同一种瓶颈。

关联材料：

- 主报告：
  `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_stress_performance_plan_20260621.md`
- 压力条件总表：
  `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_pressure_matrix_zh_20260621.md`
- 数字来源索引：
  `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_metric_source_map_zh_20260621.md`
- 外部复现 handoff runbook：
  `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_external_handoff_runbook_zh_20260621.md`
- 合作方复跑验收表：
  `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_collaborator_rerun_validation_sheet_zh_20260621.md`
- stage interaction summary：
  `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/stage_interaction_summary.json`
- tables summary：
  `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/tables_summary.json`

## 1. 先分清五层时间

| 层级 | 代表指标 | 主要回答 | 常见误读 |
| --- | --- | --- | --- |
| Client request time | latency mean/p95 | 用户看到一个请求花多久 | 不能直接定位哪个 stage 慢 |
| Workload throughput | QPS、audio throughput | 单位 wall time 完成多少请求/音频 | 不能单独代表尾延迟健康 |
| Stage lifecycle | `stage_input_received->stage_complete` | 某个 stage 从收到输入到完成输出的生命周期 | 不是纯 compute，里面可能有 queue/admission/wait |
| Internal compute span | `preprocess_start->preprocess_end`、`code2wav_decode_start->code2wav_decode_end` | 某段核心算子实际工作多久 | 不能代表该 stage 全部等待时间 |
| Boundary / stream hop | `talker_ar -> code2wav` hop、vLLM feed/drain | stage 之间交接有没有卡住 | 健康的 handoff 不等于上游/下游没有 compute tail |

## 2. SGLang Stage 指标

| 指标 | 语义 | 当前报告中的判断方式 | 结论边界 |
| --- | --- | --- | --- |
| `latency_mean_s` / `latency_p95_s` | benchmark client 观测的端到端请求时间 | 用于 headline 和 pressure 曲线 | 只说明请求体验，不直接等价于某个 stage compute |
| `rtf_mean` / `rtf_p95` | latency / generated audio duration | 用于判断语音输出是否快于实时 | 长短文本/语音输出要分开看，不能混成一个平均 |
| `throughput_qps` | wall time 每秒完成请求数 | c=8 是当前 SGLang recipe 吞吐峰值 | QPS 高不代表 tail 一定低 |
| `audio_throughput_s_per_s` | 每秒 wall time 生成多少秒音频 | synthetic short/long 用来衡量语音生成能力 | 与 Video-AMME accuracy/WER 是不同维度 |
| `stage_input_received->stage_complete` | stage lifecycle，包含 queue/admission、stage 内等待和实际工作 | c=8/c=16 preprocessing lifecycle 放大时，要和 actual preprocess 对照 | 不可直接说成“这个 stage 算子耗时” |
| `preprocess_start->preprocess_end` | preprocessing 内部实际预处理 compute | c=4/c=8/c=16 都约 0.29-0.30s，说明高并发慢主要是 queue/admission | 该 span 不覆盖 stage 外部排队 |
| `hf_processor` | HF processor 相关预处理 compute | 与 actual preprocess 一起确认 video/audio processor 没突然变慢 | 不代表 encoder/thinker 的排队 |
| `talker_ar` lifecycle | Talker 自回归 codec/text 生成路径生命周期 | c=1/c=2/c=4 和 long speech 的主要 tail 来源 | lifecycle 包含调度和等待，不是单 kernel 时间 |
| `code2wav` lifecycle | code2wav stage 的生命周期视角 | 需要和 collect/decode 分开看 | 不能只看 stage lifecycle 就说 vocoder 是瓶颈 |
| `code2wav_window_collect_start->code2wav_window_collect_end` | code2wav 等到足够 codec chunks 组成 decode window 的时间 | collect 大于 decode 时，优先怀疑 talker chunk cadence / upstream wait | collect wait 不是 vocoder compute |
| `code2wav_decode_start->code2wav_decode_end` | vocoder / code2wav decode compute span | 当前约十几到二十几 ms/window，非主瓶颈 | 只覆盖 decode window，不代表端到端音频生成 |
| `talker_to_code2wav_hop` | talker stream chunk 到 code2wav 消费之间的 hop 时间 | SGLang p95 约 15-24ms，判定 handoff 健康 | hop 健康不代表 talker AR 本身没有 tail |

## 3. vLLM Stage 指标

| 指标 | 语义 | 当前报告中的判断方式 | 结论边界 |
| --- | --- | --- | --- |
| runner wall QPS | offline runner 从整体 wall time 看完成多少请求 | 反映 host prompt build/feed + engine 的总成本 | 不能直接当 online serving QPS |
| engine QPS | 从 engine-admitted request 时间看吞吐 | prebuild 后用于隔离 prompt-feed 问题 | 仍不是完整 online ingress benchmark |
| `prompt_build_wall_s` | offline runner 本地构造 multimodal prompt 的耗时 | prebuild w4 将 prompt wall 从 w1 的 249.3s 降到 129.2s | 原始 c=8 没有 prebuild 时容易掩盖 engine 内部瓶颈 |
| `batch_admission_span_ms` | 同一 warmed batch 从第一个到最后一个请求进入 engine 的跨度 | 原始 c=8 33.31/43.97s，prebuild w4 降到 4.09/4.89s | 这是 offline admission 诊断，不是 online serving parity |
| `thinker_to_talker_feed_ms` | vLLM engine 内 thinker 输出 feed 到 talker 的边界时间 | 原始 c1/c4/c8 p95 约 1ms，说明该 handoff 不是原始 c8 限制 | prebuild 后仍需看 talker/codec tail |
| `talker_feed_to_first_codec_ms` | talker 接到输入后到首个 codec 的时间 | prebuild w1/w4 暴露后续 engine/talker-side tail | 不能用它替代端到端 latency |
| `talker_to_code2wav_drain_ms` | talker codec 输出到 code2wav drain 完成的边界 | 原始 c1/c4/c8 p95 约 16-17.5ms；prebuild 后变成 watch/bottleneck 信号 | prebuild 是诊断路径，不是最终 online serving 结论 |
| runner overhead pct | runner wall 中不属于 engine active request 的比例 | 原始 c8 overhead 81.8%，说明 prompt-feed/admission 主导 | 只适用于这个 offline runner |

## 4. 报告中的核心判定规则

| 判定 | 使用的证据组合 | 当前结论 |
| --- | --- | --- |
| preprocessing 是否真算慢 | `stage_input_received->stage_complete` 对比 `preprocess_start->preprocess_end` | c=8/c=16 lifecycle 变长，但 actual compute 稳定，所以主要是 admission/queue |
| stage handoff 是否卡住 | `talker_to_code2wav_hop`、vLLM `thinker_to_talker_feed_ms`、`talker_to_code2wav_drain_ms` | SGLang handoff 健康；vLLM 原始路径 handoff 也不是主限制 |
| code2wav 是否是 compute bottleneck | `code2wav_decode_start->code2wav_decode_end` 与 request latency、talker tail、collect wait 对比 | 当前不是主瓶颈 |
| 长语音是否拖垮 | synthetic long c=1/4/8 的 latency、RTF、talker stage、decode stage | long c=8 仍快于实时，主要压力是 talker AR |
| vLLM c=8 为什么不能直接当 parity | runner wall、engine wall、batch admission span、prebuild w1/w4 对比 | 原始 offline c=8 是 prompt-feed/admission limited；prebuild w4 是强诊断，不是 online serving parity |
| naive preproc 并发是否有效 | preproc=1 baseline vs preproc=2/preproc=4 的 QPS、latency、failure | preproc=2 回退，preproc=4 OOM/失败 |

## 5. 现场解释模板

当别人问“为什么 c=8/c=16 top stage 是 preprocessing，却说 preprocessing compute 不是瓶颈”时：

> 这里的 top stage 用的是 lifecycle span，包含 admission 和 queue。我们额外看了
> `preprocess_start->preprocess_end`：c=8/c=16 仍在约 0.29-0.30s。真正变长的是
> lifecycle minus actual compute，所以是 admission/queue 压力，不是预处理算子突然慢。

当别人问“为什么不先优化 code2wav”时：

> 当前 `code2wav_decode` 是十几到二十几毫秒/window，`talker_ar -> code2wav` hop
> p95 也在约 15-24ms。更大的 tail 来自 talker AR 和高并发 admission，所以先优化
> code2wav 的收益不如 talker/admission 明确。

当别人问“vLLM c=8 为什么不直接拿来对比 online serving”时：

> 原始 offline c=8 的 runner wall 主要被 host prompt build/feed 和 admission span
> 主导；prebuild w4 能把 admission span 降下来，但它仍是 offline diagnostic。严格
> c=8 online serving parity 需要在线 ingress 加 WER/ASR 复核。

## 6. 指标到证据/复跑入口

| 要复核的口径 | 先读的机器证据 | 继续下钻 | 复跑命令入口 |
| --- | --- | --- | --- |
| headline latency / RTF / WER | `results/qwen35_report_audit_20260619/headline_scorecard.json`、`claims_verification.json` | `results/qwen35_report_audit_20260619/metric_provenance_index.json` 里的 `headline_strict_c4` rows | `run_full_audit`、`build_headline_scorecard`、`verify_report_claims` |
| SGLang c=1/2/4/8/16 pressure | `results/qwen35_report_audit_20260619/tables_summary.json`、`headline_scorecard.json` | `metric_provenance_index.json` 里的 `sglang_videoamme_stress` rows 和 raw `videoamme_results.json` / `request_profile_*.json` | `launch_sglang_optimized`、`sglang_videoamme_stress`、`build_stage_interactions` |
| short/long synthetic speech | `tables_summary.json`、`stage_interaction_summary.json` | `metric_provenance_index.json` 里的 `synthetic_speech` rows 和 synthetic raw artifacts | `sglang_synthetic_text_to_speech`、`build_stage_interactions` |
| lifecycle vs actual compute | `stage_interaction_summary.json`、`stage_drilldown_index.json` | `stage_reproduction_drilldown.json` 的 `quick_reproduction_map`、对应 `stage_row` / `metric_row` jq query | `build_stage_interactions`、`build_stage_drilldown_index`、`build_stage_reproduction_drilldown` |
| stage handoff / collect wait / decode compute | `stage_boundary_bottleneck_ledger.json`、`stage_causal_graph.json` | `stage_reproduction_drilldown.json` 中 `talker -> code2wav_stream`、`code2wav_collect -> code2wav_decode` rows | `build_stage_boundary_bottleneck_ledger`、`build_stage_causal_graph`、`build_stage_route_decision_matrix` |
| vLLM original/prebuild c=8 admission | `vllm_log_stage_summary.json`、`vllm_admission_diagnosis.json` | `stage_reproduction_drilldown.json` 的 vLLM quick routes 和 `metric_provenance_index.json` 的 `vllm_offline_diagnostic` rows | `vllm_c8_original`、`vllm_c8_prebuild_w4`、`summarize_vllm_log_stages`、`diagnose_vllm_admission` |
| 对外能不能替换 headline | `rerun_acceptance_contract.json`、`collaborator_return_check.json`、`rerun_delta_triage.json` | `objective_requirement_crosswalk.json`、`claim_metric_crosswalk.json` 和合作方回填表 | `build_rerun_acceptance_contract`、`build_collaborator_return_check`、`build_rerun_delta_triage` |

现场查单条 stage 时优先用 `stage_reproduction_drilldown.json` 的
`quick_reproduction_map`。每条 route 都给出 `stage_query`、`metric_query` 和
`first_rerun_command_id`；如果需要完整命令，再用 `rerun_command_ids` 去
`results/qwen35_report_audit_20260619/repro_command_manifest.json` 查对应命令。

## 7. 复现 gate

期望 gate：

- full audit `ok=true`
- claims `17/17`
- coverage `34/34`
- preflight `62` checks, `0` required failures
- manifest current `196` records, minimum `180`, `0` missing
- SGLang optimization lock `26/26`
- repro command manifest `63` commands / `7` phases
- metric provenance index `248` rows / `11/11`
- stage reproduction drilldown `52` rows / `17/17`
- stage route decision matrix `ready=true`
- headline scorecard `9/9`
- acceptance matrix `17/17`
- confidence ledger `12/12`
