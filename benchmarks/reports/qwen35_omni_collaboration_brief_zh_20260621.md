# Qwen3.5-Omni SGLang-Omni 性能分析对外简报

状态：2026-06-21 checkpoint 阶段稿；完整工作报告与审计证据在
更新后的目标不再等待 6.21 晚间；后续变更必须重跑 full audit。
主报告：
`/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_stress_performance_plan_20260621.md`
最终分享交付说明：
`/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_final_share_delivery_note_zh_20260621.md`
一页式核心数字 scorecard：
`/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_one_page_scorecard_zh_20260621.md`
分享包索引：
`/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_share_package_index_zh_20260621.md`
原始需求-证据映射：
`/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_requirement_evidence_map_zh_20260621.md`
压力条件总表：
`/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_pressure_matrix_zh_20260621.md`
数字来源索引：
`/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_metric_source_map_zh_20260621.md`
Stage 指标字典：
`/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_stage_metric_dictionary_zh_20260621.md`
外部复现 handoff runbook：
`/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_external_handoff_runbook_zh_20260621.md`
合作方复跑验收表：
`/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_collaborator_rerun_validation_sheet_zh_20260621.md`
答辩 Q&A：
`/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_defense_qna_zh_20260621.md`
优化 playbook：
`/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_optimization_playbook_zh_20260621.md`
分享 deck 提纲：
`/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_share_deck_outline_zh_20260621.md`

## 1. 结论先行

在本地 8x NVIDIA H20 环境、Qwen3.5-Omni
`qwen3_5_omni_23b_final_multilingual_all_voice_bf16_0315` 模型、Video-AMME
ci-50 语音输出 workload 上，当前 SGLang-Omni 优化方案已经具备对外分享的性能证据：

- 与优化版 vLLM 的严格 warmed c=4 对比中，SGLang-Omni 在 latency mean、
  latency p95、RTF mean、RTF p95 均优于 vLLM，同时 accuracy 和 WER 不退化。
- SGLang-Omni 在 c=1/2/4/8/16 压测中，c=8 是当前 recipe 的吞吐峰值：
  2.540 req/s，5.372 generated-audio seconds / wall second。
- c=16 不建议作为当前 recipe 的默认运行点：吞吐从 c=8 的 2.540 req/s
  下降到 2.407 req/s，RTF mean 上升到 4.8489，主要是排队和 admission 压力。
- 长语音合成路径仍然快于实时：synthetic long c=8 平均生成约 52.3s 音频，
  平均 latency 约 25.8s，RTF 0.4932。
- `code2wav_decode` 不是当前瓶颈；stage 间连接，尤其 `talker_ar -> code2wav`
  的流式 hop，在现有证据下没有形成主要阻塞。
- naive 提高 preprocessing 并发不是有效优化：preproc=2 使 c=8 QPS 下降
  35.4%，preproc=4 出现 OOM 和失败请求。

## 2. 最关键的横向对比

严格横向对比使用 warmed skip-first-4 的 c=4 slice，避免首次 compile /
CUDA graph capture 污染稳态性能。

| Runtime | Scope | n | Accuracy | Latency Mean | Latency P95 | RTF Mean | RTF P95 | WER Corpus |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vLLM optimized | Video-AMME ci-50, c=4, skip first 4 | 46 | 63.0% | 2.093s | 3.525s | 1.4677 | 3.0717 | 7.44% |
| SGLang-Omni optimized | Video-AMME ci-50, c=4, skip first 4 | 46 | 67.4% | 1.743s | 3.328s | 1.3536 | 2.4023 | 4.12% |

相对 vLLM，SGLang-Omni 在该 warmed c=4 对比中：

- mean latency 低 16.7%
- p95 latency 低 5.6%
- mean RTF 低 7.8%
- p95 RTF 低 21.8%

这是当前最强的 apples-to-apples cross-runtime 对比点。

## 3. SGLang-Omni 压测形态

| Concurrency | Accuracy | Latency Mean | Latency P95 | RTF Mean | Throughput | WER Corpus | 结论 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| c=1 | 70.0% | 1.315s | 2.629s | 0.9207 | 0.760 req/s | 3.85% | 单并发 talker tail 主导 |
| c=2 | 70.0% | 1.521s | 2.968s | 1.0416 | 1.315 req/s | 3.85% | 仍主要是 talker tail |
| c=4 | 70.0% | 1.965s | 3.532s | 1.5236 | 2.036 req/s | 3.85% | 稳态高性价比运行点 |
| c=8 | 70.0% | 3.064s | 5.733s | 2.2881 | 2.540 req/s | 3.23% | 当前吞吐峰值 |
| c=16 | 70.0% | 6.008s | 9.288s | 4.8489 | 2.407 req/s | 2.88% | 超过当前 admission sweet spot |

性能拐点很清楚：c=1/c=2/c=4 主要受 `talker_ar` 尾部生成影响；c=8 开始出现
preprocessing admission queue 叠加 talker tail；c=16 排队压力明显，不建议用于当前
默认配置。

## 4. Stage Breakdown 和连接关系

当前 profiler 证据支持以下归因：

- `talker_ar`：短问答和长语音输出的主要 compute tail，尤其在 c=1/c=4 以及 long
  synthetic speech 中最明显。
- `preprocessing`：实际 compute 均值约 0.27-0.32s，但 lifecycle 在 c=8/c=16
  增大，说明主要是 admission / queueing，而不是视频预处理本身突然变慢。
- `code2wav_decode`：平均约 13-17ms/window，远小于整体请求耗时，不是主要 compute
  bottleneck。
- `talker_ar -> code2wav`：流式 hop p95 约 15-24ms，stage 连接健康。
- `code2wav_window_collect` 大于 decode 本身，说明 code2wav 表面 tail 主要是在等
  codec chunks，而不是 vocoder 算不动。

因此当前最值得优化的是 talker AR 效率、c=8 附近的 admission 策略，以及避免 c=16
把 preprocessing/thinker 入口过度塞满。

### 4.1 Stage 连接和相互影响矩阵

给合作方讲解时，可以用下面这张表把“哪里慢”和“是不是 stage 之间卡住”分开：

| Stage 边界 | 证据 | 判断 |
| --- | --- | --- |
| request admission -> preprocessing | c=8 preprocessing lifecycle 约 1.23s，但实际 preprocess compute 约 0.29s；c=16 lifecycle 到 4.40s，实际 compute 仍约 0.30s | 高并发慢在 admission/queue，不是视频预处理本身算不动 |
| preprocessing -> encoder/thinker | preproc=2 后 media load、image/audio encoder、thinker lifecycle 全部变慢，preproc=4 OOM | 盲目放开 preprocessing 会把排队问题变成 GPU0/thinker 争用 |
| thinker -> talker | vLLM 日志里 thinker-to-talker feed p95 约 1ms；SGLang 尾部主要出现在 talker AR 工作本身 | 边界 handoff 不是主瓶颈，后续更该优化 talker AR 和 batching |
| talker -> code2wav | SGLang stream-hop p95 约 15-24ms；原始 vLLM c1/c4/c8 talker-to-code2wav drain p95 约 16-17.5ms | 当前 SGLang stage 连接健康；code2wav 前的流式连接不是 c=8/c=16 主因 |
| code2wav collect -> decode | decode 平均约 14-17ms/window，collect 比 decode 大 | 表面 code2wav tail 多数是在等 codec chunks，不是 vocoder compute 瓶颈 |
| offline runner -> vLLM engine admission | 原始 vLLM c=8 admission span 33.3s/44.0s；prebuild w4 降到 4.09s/4.89s | vLLM 原始 c=8 主要是 host prompt build/feed 限制；w4 是诊断，不是 online serving parity |

这张矩阵的核心结论是：当前 SGLang-Omni 的 stage 连接没有形成主要瓶颈。真正要处理的是
c=8/c=16 admission 压力、资源争用，以及 talker AR 的生成尾部。
机器可读证据在：
`/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/stage_interaction_summary.json`

### 4.2 优化开关证据 Ledger

下面这张表用于回答“优化到底开了什么、为什么不是弱 baseline”：

| Runtime | 开关/设置 | 作用 | 证据边界 |
| --- | --- | --- | --- |
| SGLang | `--thinker-cuda-graph on`、`--talker-cuda-graph on`、`--talker-torch-compile on` | 让 Thinker/Talker 走 warmed graph/compile 路径 | warmed c=4 latency/RTF 优于优化版 vLLM；cold compile 单独报告 |
| SGLang | `NO_CODE2WAV_TORCH_COMPILE=0`、`TORCHDYNAMO_DISABLE=0` | 保持 code2wav 编译路径 | decode 约 13-17ms/window，不是当前主瓶颈 |
| SGLang | thinker/talker `max_running_requests=8` | 覆盖 c=4-c=8 推荐窗口，并暴露 c=16 饱和点 | c=8 QPS 2.540 达峰，c=16 下降到 2.407 |
| SGLang | `PREPROCESSING_MAX_CONCURRENCY=1` | 当前 H20 96GB 布局下的安全 admission 点 | preproc=2 QPS 下降 35.4%，preproc=4 OOM/失败 |
| vLLM | Qwen3.5-capable 镜像、compile mode、CUDA graph、code2wav compile | 确保比较对象是优化版 vLLM，而不是保守 eager baseline | 严格 warmed c=4 对比使用该优化 artifact |
| vLLM | `max_num_seqs=4/8`、prefix caching、chunked prefill、shared-memory transfer | 使用 vLLM-Omni 侧推荐性能路径 | 原始 c1/c4/c8 engine-side stage 边界很小 |
| vLLM | `--prebuild-prompts --prebuild-workers 4` | 去掉 offline runner 的串行 prompt build/feed admission | runner QPS 从 0.1420 到 0.2127，但仍不能当 online serving parity |

## 5. vLLM c=8 诊断边界

vLLM 使用 Qwen3.5-capable 镜像：

`tongyi-duanwu-registry-vpc.cn-beijing.cr.aliyuncs.com/dashscope/dashllm:cuda129_cp312_test_vl_13589`

vLLM c=8 能正常跑通，但 offline runner 的 wall throughput 不能直接当作 online serving
吞吐对比，因为 runner 会在 `omni.add_request()` 前进行本地视频 decode / sampling 和
prompt 构建。我们额外做了 prebuilt-prompt 诊断：

| vLLM c=8 Artifact | Completed | Accuracy | Prompt Build Wall | Runner Wall | Runner QPS | Engine QPS | Admission Span Avg/P95 | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| original c8 | 50/50 | 66.0% | n/a | 308.2s | 0.1622 | 0.1622 | 33.3s / 44.0s | prompt-feed limited |
| prebuild w1 | 50/50 | 66.0% | 249.3s | 352.1s | 0.1420 | 0.5391 | 4.44s / 5.43s | prompt admission 被移除，runner 仍慢 |
| prebuild w4 | 50/50 | 66.0% | 129.2s | 235.1s | 0.2127 | 0.5360 | 4.09s / 4.89s | 当前最强 vLLM offline 诊断 |

结论边界：w4 证明 vLLM 原始 c=8 的主要问题是 host prompt build/feed admission，也证明
parallel prompt prebuild 是正确优化方向。但 engine QPS 仍约 0.536，明显低于 SGLang
c=8 的 2.540 req/s；而且 prebuild artifact 尚未计算 WER。因此它是强诊断证据，不是
严格的 c=8 online serving parity 声明。

## 6. 复现入口

建议合作方按这个顺序复现：

1. 先跑完整审计，确认本地 evidence package 完整：

```bash
cd /home/gangouyu/sglang-omni

python3 -m benchmarks.eval.run_qwen35_omni_report_audit \
  --root /home/gangouyu/sglang-omni \
  --summary-output results/qwen35_report_audit_20260619/audit_run_summary.json
```

2. 查看审计摘要：

`/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/audit_run_summary.json`

当前状态：

- one-command audit: pass
- claim verifier: 17/17 pass
- requirement coverage: 34/34 pass
- preflight: 62 checks, 0 required failures, 1 optional Whisper host-cache warning
- manifest current: 196 records, minimum 180, 0 missing
- headline scorecard: ready=true, 9/9 checks
- acceptance matrix: ready=true, 17/17 rows
- confidence ledger: ready=true, 12/12 entries, high=9, medium=3, unsupported=0
- SGLang optimization lock: ready=true, 26/26 checks
- vLLM optimization lock: ready=true, 22/22 checks
- vLLM online parity protocol: ready=true, 18/18 checks, online_parity_proven=false
- repro command manifest: ready=true, 63 commands / 7 phases
- rerun acceptance contract: ready=true, 17/17 checks, 18 rules, 34 required
  return-evidence files, 27 command-to-return-evidence rows, no command/file gaps
- 合作方复跑结果必须用
  `qwen35_omni_collaborator_rerun_validation_sheet_zh_20260621.md` 填写环境、
  指标、回传证据和是否允许替换 headline

环境快照：

`/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/environment_snapshot.json`

它记录 GPU inventory、Docker image IDs、git state、模型/数据路径和当前审计摘要。

3. 查看复现清单和完整报告：

`/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_reproduction_checklist_zh_20260621.md`

完整报告和可复现命令：

`/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_stress_performance_plan_20260621.md`

4. 需要重新生成关键表格时：

```bash
python3 -m benchmarks.eval.summarize_qwen35_omni_report_artifacts \
  --root /home/gangouyu/sglang-omni \
  --json-output results/qwen35_report_audit_20260619/tables_summary.json
```

5. 需要复现 vLLM c=8 prebuild w4 诊断时：

```bash
RUN_ROOT="/home/gangouyu/sglang-omni/results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuildw4_20260620_$(date +%H%M%S)" \
MAX_SAMPLES=50 MAX_CONCURRENCY=8 MAX_NUM_SEQS=8 \
RUN_TAG=ci50_offline_compile_c8_mns8_prebuildw4_20260620 \
EXTRA_ARGS="--prebuild-prompts --prebuild-workers 4" \
bash results/qwen35_vllm_videoamme_ci50_opt_20260618_162319/run_vllm_videoamme_ci5_offline_compile.sh
```

## 7. 当前不能过度声明的部分

- 官方 SeedTTS full-set 数据当前没有本地 cache，因此 headline 中不使用 SeedTTS 完整集
  数字。报告中只提供 Video-AMME spoken-reference 的 SeedTTS-compatible smoke path。
- ci-50/stress/synthetic 证据不能直接外推到完整线上流量；更大 Video-AMME 或真实线上流量
  需要同口径复跑、stage/tail/quality gate 和 caveat gate 全绿。
- vLLM c=8 prebuild w4 是 offline diagnostic，不等于 online serving benchmark。
  若要做严格 c=8 serving parity 声明，还需要在线 ingress 版本，并补 WER/ASR。
- c=16 当前不是推荐运行点。它是压力边界测试，用来证明 queue/admission 饱和。

## 8. 建议对外表述

推荐对外使用的表述：

> 在 8x H20、本地 Video-AMME ci-50 语音输出 workload 上，优化后的 SGLang-Omni
> Qwen3.5 在 warmed c=4 与优化版 vLLM 的严格对比中取得更低 latency/RTF，并保持
> accuracy/WER 不退化。SGLang-Omni 的吞吐峰值出现在 c=8；主要后续优化方向是
> talker AR 和 c=8 附近 admission 策略，而不是 code2wav。vLLM c=8 的 offline
> runner 主要受 host prompt build/feed admission 影响，parallel prebuild 能显著改善
> runner wall，但仍需 online ingress + WER/ASR 才能作为严格 serving-throughput 对比。
> ci-50/stress/synthetic 证据不能直接外推到完整线上流量；更大 Video-AMME 或真实线上流量
> 需要同口径复跑、stage/tail/quality gate 和 caveat gate 全绿。

这段表述对应的机器化边界在：

`/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/confidence_ledger.json`
