# Qwen3.5-Omni SGLang-Omni 中文技术报告

生成时间 UTC：`2026-06-21T02:00:27.153531+00:00`。
工作目录：`/home/gangouyu/sglang-omni`。

定位：这是一份可以直接发给合作高校的中文技术报告正文。它不引入手工数字，
所有核心结论来自已经通过 full audit 的 JSON 证据和随包报告。

## 1. Executive Summary

- 严格横向 headline 使用 warmed c=4：SGLang-Omni 在 latency mean、latency p95、RTF mean、RTF p95 均优于优化版 vLLM，同时 accuracy/WER 不退化。
- SGLang 推荐服务窗口是 c=4 到 c=8；c=8 是当前吞吐峰值，c=16 是压力边界。
- 短/长文本语音输出路径覆盖 thinker/talker/code2wav；`length_regime_coverage.json` 机器 gate 通过，long c=8 仍快于实时。
- stage 连接本身不是当前主瓶颈；stage budget 显示主要压力来自 talker AR tail、c=8/c=16 admission/queue，以及 vLLM offline prompt-feed admission。
- vLLM baseline 使用 Qwen3.5-capable 镜像和 compile/CUDA graph/cache/prebuild 证据，不是弱 baseline；但 vLLM c=8 prebuild w4 仍只能作为 offline diagnostic。

## 2. 环境和方法

- 模型：`qwen3_5_omni_23b_final_multilingual_all_voice_bf16_0315`。
- GPU 合同：`8x NVIDIA H20 / CUDA 12.8`。
- SGLang 镜像：`frankleeeee/sglang-omni:dev`。
- vLLM 镜像：`tongyi-duanwu-registry-vpc.cn-beijing.cr.aliyuncs.com/dashscope/dashllm:cuda129_cp312_test_vl_13589`。
- 主 workload：Video-AMME ci-50，视频 + spoken question，输出 text + speech。
- 压力覆盖：SGLang c=1/2/4/8/16；synthetic short/long text-to-speech c=1/4/8；vLLM c=1/c=4/c=8 和 c=8 prebuild diagnostic。
- 指标口径：latency 是 client-observed end-to-end；RTF 是 latency / generated audio duration；WER 使用 offline Whisper large-v3 或等价 ASR 路径复算。

### 2.1 指标和口径速查

| 术语 | 本报告口径 | 容易误读的地方 |
| --- | --- | --- |
| `c` | benchmark client concurrency；SGLang 压力扫 c=1/2/4/8/16，synthetic speech 扫 c=1/4/8。 | c=16 可运行不等于推荐服务点；本报告推荐窗口是 c=4-c=8。 |
| warmed / skip-first | strict headline 使用 warmed c=4 slice，跳过 cold compile / CUDA graph capture 的前若干请求。 | 不要把 cold-start 编译开销和 warmed serving latency 混在同一 headline。 |
| latency | client observed end-to-end latency，覆盖请求到 text+speech 输出。 | 它不是单个 kernel 或单个 stage 的时间；stage 表用于拆解来源。 |
| RTF | latency / generated audio duration；小于 1 表示快于实时生成。 | 长文本 RTF 低不代表 QPS 高，两者分别看实时性和吞吐。 |
| QPS | 当前 workload 与并发下的完成吞吐。 | SGLang c=8 是当前吞吐峰值；c=16 QPS 回落且 tail 变差。 |
| WER / accuracy | WER 用 offline ASR 链路复算；accuracy 来自 Video-AMME ci-50 任务判分。 | 性能数字替换前必须保留同口径 WER/ASR 和 accuracy 验收。 |
| stage boundary / handoff | 相邻 stage 的连接状态，例如 talker->code2wav stream hop。 | 当前 SGLang handoff 健康；不要把连接健康误写成主瓶颈。 |
| queue estimate | preprocessing lifecycle 中扣除实际 preprocess 后的 admission/queue 估计。 | 低并发无排队估计；c=8/c=16 queue 显性化才是高并发压力信号。 |
| offline diagnostic | 用来定位瓶颈的离线 runner 证据，例如 vLLM c=8 prebuild w4。 | offline diagnostic 不能直接升级成 online serving parity。 |
| share-ready with caveat | 当前包可分享，但带明确边界：ci-50/stress/synthetic、vLLM c=8 online parity、SeedTTS full-set 不越界。 | caveat 是外发口径的一部分，不是证据失败。 |

### 2.2 Runtime fairness / 镜像与优化锁定

这张表回答 baseline 是否公平：SGLang 和 vLLM 都锁定镜像 digest，且 vLLM 使用 compile/CUDA graph/cache/encoder/prebuild 证据，不是保守弱 baseline。
同时它把 strict headline 与 vLLM c=8 offline diagnostic 的边界分开，避免把诊断结果误写成 online serving parity。

| Runtime / scope | Image digest | Optimized recipe evidence | 可以声明 | 不能声明 / 升级条件 |
| --- | --- | --- | --- | --- |
| SGLang-Omni serving c4-c8 | `sha256:be7e72126f525c3767008a73de16f400f974a09db431ded3c52bd48370941a84` | `NO_CODE2WAV_TORCH_COMPILE=0`; `TORCHDYNAMO_DISABLE=0`; `--thinker-cuda-graph on`; `--talker-cuda-graph on`; `--talker-torch-compile on`; max-running=8; 16GiB preprocessing cache; `PREPROCESSING_MAX_CONCURRENCY=1` | compiled/graph SGLang recipe with serial preprocessing and 16GiB preprocessing cache; c4-c8 warmed serving; c8 peak throughput; c16 saturation boundary | 不能把 c16 包装成推荐服务点；preproc=2/4 是当前反例 |
| vLLM strict headline c4 | `sha256:e71dc281e0882896f81e3471206312b7c31ba4b6ab56ed9def0d4f1392a8c4ba` | `VLLM_ENABLE_TORCH_COMPILE=True`; `enforce_eager=False`; `FULL_AND_PIECEWISE` CUDA graph; prefix cache + chunked prefill; shared-memory hidden buffer; encoder compile/batch | optimized warmed c4 apples-to-apples headline only; baseline 是优化版，不是弱 baseline | 只用于 warmed c=4 apples-to-apples headline，不能外推成 c=8 online parity |
| vLLM c8 prebuild diagnostic | `sha256:e71dc281e0882896f81e3471206312b7c31ba4b6ab56ed9def0d4f1392a8c4ba` | `--prebuild-prompts --prebuild-workers 4`; same optimized image and log-stage/admission diagnostics | prebuild w4 is optimized offline diagnostic, not online parity; 可说 prebuild 明显缓解 offline prompt-feed admission | online_parity_proven=`False`；升级需要 online ingress + WER/ASR + stage boundary 复核 |

## 3. 严格 SGLang-vLLM 对比

严格横向比较只使用 warmed skip-first-4 的 c=4 slice，避免 cold compile / CUDA graph capture 影响。

| Runtime | n | Accuracy | Latency Mean | Latency P95 | RTF Mean | RTF P95 | WER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SGLang-Omni optimized | 46 | 67.4% | 1.743s | 3.328s | 1.3536 | 2.4023 | 4.12% |
| vLLM optimized | 46 | 63.0% | 2.093s | 3.525s | 1.4677 | 3.0717 | 7.44% |

相对 vLLM，SGLang-Omni mean latency 低 16.7%，p95 latency 低 5.6%，mean RTF 低 7.8%，p95 RTF 低 21.8%。

### 3.1 c=4 指标口径说明

报告里有两个 c=4：一个是 strict warmed c=4 headline slice，一个是 SGLang pressure sweep c=4。
它们用途不同，所以 n、accuracy/WER、latency/RTF 不要求逐项相同；复现和答辩时按下表选择引用口径。

| c=4 slice | 用途 | n | Accuracy/WER | Latency mean/p95 | RTF mean/p95 | Artifact | 外推边界 |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| strict warmed c=4 headline | SGLang-vLLM 横向比较，只引用这一行做 headline | 46 | 67.4%/4.12% | 1.743s/3.328s | 1.3536/2.4023 | videoamme_results.json | 用于 cross-runtime apples-to-apples；不要和 stress sweep 逐项相减 |
| SGLang stress c=4 | SGLang 内部 pressure sweep，用于服务窗口和 stage scaling | 50 | 70.0%/3.85% | 1.929s/3.633s | 1.4015/2.4983 | videoamme_results.json | 用于 SGLang c=1/2/4/8/16 scaling；不要拿来替代 strict vLLM headline |

## 4. SGLang 单并发和高并发压力结论

| Pressure | 决策 | Stage/瓶颈判断 | 关键数字 |
| --- | --- | --- | --- |
| c=1 | 推荐服务窗口 | 低/中并发主 tail 是 talker_ar；stage handoff 和 code2wav decode 不是主瓶颈。 | acc=70.0%, WER=3.85%, lat_mean=1.316s, lat_p95=2.406s, QPS=0.760, hop_p95=15.5ms, decode_p95=17.8ms |
| c=2 | 推荐服务窗口 | 低/中并发主 tail 是 talker_ar；stage handoff 和 code2wav decode 不是主瓶颈。 | acc=70.0%, WER=3.85%, lat_mean=1.508s, lat_p95=3.124s, QPS=1.315, hop_p95=16.1ms, decode_p95=20.2ms |
| c=4 | 推荐服务窗口 | 低/中并发主 tail 是 talker_ar；stage handoff 和 code2wav decode 不是主瓶颈。 | acc=70.0%, WER=3.85%, lat_mean=1.929s, lat_p95=3.633s, QPS=2.036, hop_p95=17.8ms, decode_p95=21.8ms |
| c=8 | 推荐吞吐峰值 | 当前高并发甜点；admission/queue 开始显性化，但 talker->code2wav hop 仍健康。 | acc=70.0%, WER=3.23%, lat_mean=3.064s, lat_p95=5.853s, QPS=2.540, hop_p95=20.4ms, decode_p95=25.9ms |
| c=16 | 压力边界 | 吞吐回落且 queue/admission 饱和；只作为压力边界，不做默认服务点。 | acc=70.0%, WER=2.88%, lat_mean=6.066s, lat_p95=7.846s, QPS=2.407, hop_p95=19.7ms, decode_p95=23.9ms |

读法：c=1/c=2/c=4 主要是 talker AR tail；c=8 达到当前吞吐峰值；c=16 虽可运行但 admission/queue 饱和，不能作为推荐服务点。

## 5. 短/长文本语音输出结论

| Pressure | 决策 | Stage/瓶颈判断 | 关键数字 |
| --- | --- | --- | --- |
| short c=1 | 语音输出回归保护 | 短文本语音用于验证 thinker/talker/code2wav 输出路径；code2wav 边界保持小。 | text_words=12, audio=4.2s, lat_mean=0.866s, RTF=0.2052, QPS=1.154, hop_p95=14.9ms, decode_p95=14.2ms |
| short c=4 | 语音输出回归保护 | 短文本语音用于验证 thinker/talker/code2wav 输出路径；code2wav 边界保持小。 | text_words=12, audio=4.3s, lat_mean=1.768s, RTF=0.4105, QPS=2.218, hop_p95=20.1ms, decode_p95=22.3ms |
| short c=8 | 语音输出回归保护 | 短文本语音用于验证 thinker/talker/code2wav 输出路径；code2wav 边界保持小。 | text_words=12, audio=4.3s, lat_mean=2.638s, RTF=0.6257, QPS=2.983, hop_p95=21.2ms, decode_p95=24.3ms |
| long c=1 | 语音输出回归保护 | 长文本/长语音主要压 talker AR 和 chunk cadence；long c=8 仍快于实时。 | text_words=139, audio=51.9s, lat_mean=9.168s, RTF=0.1766, QPS=0.109, hop_p95=15.0ms, decode_p95=14.2ms |
| long c=4 | 语音输出回归保护 | 长文本/长语音主要压 talker AR 和 chunk cadence；long c=8 仍快于实时。 | text_words=139, audio=52.6s, lat_mean=17.551s, RTF=0.3338, QPS=0.227, hop_p95=20.4ms, decode_p95=21.6ms |
| long c=8 | 语音输出回归保护 | 长文本/长语音主要压 talker AR 和 chunk cadence；long c=8 仍快于实时。 | text_words=139, audio=52.3s, lat_mean=25.799s, RTF=0.4932, QPS=0.303, hop_p95=24.0ms, decode_p95=18.2ms |

读法：synthetic speech 用来隔离 thinker/talker/code2wav。长文本 c=8 仍快于实时，说明长输出路径没有靠牺牲语音一致性换吞吐。
机器审计入口：`results/qwen35_report_audit_20260619/length_regime_coverage.json` ready=`True`，checks=`10/10`，rows=`7`；short=`74 chars`，long=`944 chars`，long c=8 RTF p95=`0.5001`；max hop/decode p95=`24.0ms/24.3ms`。
边界：这条证据证明 ci-50 target length 与 synthetic short/long guardrail 自洽，不能外推为完整线上流量或 official SeedTTS full-set headline。

### 5.1 Serving/capacity 决策矩阵

这张表把压力点翻译成运行选择：哪些可以做服务窗口，哪些只能做压力边界或诊断证据。
它同时给出对应 stage guard，方便复跑时判断新数字是否仍可替换当前结论。

| 压力/场景 | 运行选择 | 可承诺指标 | Stage guard | 不要做 |
| --- | --- | --- | --- | --- |
| Video-AMME c=1-c2 | latency-first / 单并发到低并发 | QPS=0.760-1.315; lat_p95=2.406s-3.124s; RTF_p95=2.0199-1.9308 | 主 tail 是 talker_ar；handoff/decode p95 仍小 | 不要把低并发 tail 当成高并发 admission 问题 |
| Video-AMME c=4 | balanced serving / strict headline 参照点 | QPS=2.036; lat_p95=3.633s; RTF_p95=2.4982; queue=191.1ms / 9.9% | talker_ar tail 为主，queue 仍可控；strict SGLang-vLLM 对比只用 warmed c=4 slice | 不要把 stress c=4 和 strict warmed c=4 混成同一组数字 |
| Video-AMME c=8 | throughput edge / 当前高并发甜点 | QPS=2.540; lat_p95=5.853s; RTF_p95=4.3924; queue=937.4ms / 30.6% | admission queue 开始显性化，但 QPS 仍是当前峰值，handoff 不是瓶颈 | 不要继续加 preprocessing 并发；preproc=2/4 已是反例 |
| Video-AMME c=16 | saturation evidence / 压力边界 | QPS=2.407; lat_p95=7.846s; RTF_p95=10.4087; queue=4090.5ms / 67.4% | 吞吐低于 c=8，queue 占比大幅上升，RTF tail 明显变差 | 不要作为默认服务点，也不要写成高并发更优 |
| Synthetic short c=8 | 短文本语音高并发 guard | QPS=2.983; lat_p95=2.828s; RTF_p95=0.7440; audio=4.3s; hop_p95=21.2ms | 短文本仍快于实时；code2wav decode 占比小 | 不要把短文本结论外推成长文本吞吐 |
| Synthetic long c=8 | 长文本/长语音 realtime guard | QPS=0.303; lat_p95=26.318s; RTF_p95=0.5001; audio=52.3s; hop_p95=24.0ms | 长文本 c=8 RTF_p95 仍小于 1；压力主要进入 talker cadence | 不要把 vocoder decode 当成优先瓶颈 |
| vLLM c=8 prebuild w4 | optimized offline diagnostic | runner_QPS=0.2127; engine_QPS=0.5360; admission_p95=4891.5ms | prebuild 移除大部分 prompt-feed admission，暴露后续 engine/talker tail | 不要升级为 online serving parity；需要 online ingress + WER/ASR 复核 |

## 6. Stage Breakdown 和连接瓶颈

### 6.1 Pressure × Stage Heatmap

这张表先把所有压力点压到一页：单/低并发、高并发、短/长文本、vLLM original/prebuild 都用同一组 stage 列阅读。
它来自 `pressure_stage_heatmap.json`，不引入新的 benchmark 数字；用途是快速回答“这个压力到底压到了哪个 stage、连接是不是瓶颈”。

| Pressure | Runtime | Key metrics | Stage hotspot | Connection verdict | Decision / Caveat |
| --- | --- | --- | --- | --- | --- |
| Video-AMME c=1 | sglang | QPS=0.760; lat_p95=2.406s; RTF_p95=2.0199 | preproc_lifecycle=294.8ms / 22.4%; queue=无排队估计 ; talker=444.1ms / 33.7%; top_stage=talker_ar 444/1516ms | talker->code2wav hop_p95=15.5ms; handoff is not the current bottleneck ; decode=14.3ms / 1.1%; collect=26.4ms | latency-first guard / Do not describe low-concurrency Talker tail as high-concurrency queue saturation. |
| Video-AMME c=2 | sglang | QPS=1.315; lat_p95=3.124s; RTF_p95=1.9308 | preproc_lifecycle=334.1ms / 22.2%; queue=无排队估计 ; talker=525.8ms / 34.9%; top_stage=talker_ar 526/2024ms | talker->code2wav hop_p95=16.1ms; handoff is not the current bottleneck ; decode=16.0ms / 1.1%; collect=31.9ms | latency-first guard / Do not describe low-concurrency Talker tail as high-concurrency queue saturation. |
| Video-AMME c=4 | sglang | QPS=2.036; lat_p95=3.633s; RTF_p95=2.4982 | preproc_lifecycle=486.9ms / 25.2%; queue=191.1ms / 9.9% ; talker=663.1ms / 34.4%; top_stage=talker_ar 663/2628ms | talker->code2wav hop_p95=17.8ms; handoff is not the current bottleneck ; decode=16.0ms / 0.8%; collect=45.6ms | balanced serving / Do not mix stress c4 and strict warmed c4 metrics. |
| Video-AMME c=8 | sglang | QPS=2.540; lat_p95=5.853s; RTF_p95=4.3924 | preproc_lifecycle=1226.6ms / 40.0%; queue=937.4ms / 30.6% ; talker=982.7ms / 32.1%; top_stage=preprocessing 1227/2164ms | talker->code2wav hop_p95=20.4ms; handoff is not the current bottleneck ; decode=17.2ms / 0.6%; collect=68.1ms | throughput edge / Do not widen preprocessing concurrency without redesigning admission and placement. |
| Video-AMME c=16 | sglang | QPS=2.407; lat_p95=7.846s; RTF_p95=10.4087 | preproc_lifecycle=4395.1ms / 72.5%; queue=4090.5ms / 67.4% ; talker=815.6ms / 13.4%; top_stage=preprocessing 4395/5884ms | talker->code2wav hop_p95=19.7ms; handoff is not the current bottleneck ; decode=16.7ms / 0.3%; collect=61.0ms | saturation boundary / Do not present c16 as the high-concurrency optimum. |
| Synthetic short text c=1 | sglang | QPS=1.154; lat_p95=0.924s; RTF_p95=0.2112; words=12; audio_mean=4.2s | synthetic speech isolates thinker/talker/code2wav; no Video-AMME preprocessing queue claim ; talker=849.0ms / 98.0%; RTF below 1.0 | talker->code2wav hop_p95=14.9ms; handoff delta stays small versus Talker cadence ; decode=12.6ms / 1.5%; collect=27.0ms | short-text speech guardrail / Do not replace full-set or Video-AMME headline with synthetic-only evidence. |
| Synthetic short text c=4 | sglang | QPS=2.218; lat_p95=2.056s; RTF_p95=0.4388; words=12; audio_mean=4.3s | synthetic speech isolates thinker/talker/code2wav; no Video-AMME preprocessing queue claim ; talker=1722.6ms / 97.4%; RTF below 1.0 | talker->code2wav hop_p95=20.1ms; handoff delta stays small versus Talker cadence ; decode=16.5ms / 0.9%; collect=57.9ms | short-text speech guardrail / Do not replace full-set or Video-AMME headline with synthetic-only evidence. |
| Synthetic short text c=8 | sglang | QPS=2.983; lat_p95=2.828s; RTF_p95=0.7440; words=12; audio_mean=4.3s | synthetic speech isolates thinker/talker/code2wav; no Video-AMME preprocessing queue claim ; talker=2205.9ms / 83.6%; RTF below 1.0 | talker->code2wav hop_p95=21.2ms; handoff delta stays small versus Talker cadence ; decode=15.3ms / 0.6%; collect=73.4ms | short-text speech guardrail / Do not replace full-set or Video-AMME headline with synthetic-only evidence. |
| Synthetic long text c=1 | sglang | QPS=0.109; lat_p95=9.465s; RTF_p95=0.1776; words=139; audio_mean=51.9s | synthetic speech isolates thinker/talker/code2wav; no Video-AMME preprocessing queue claim ; talker=9091.2ms / 99.2%; RTF below 1.0 | talker->code2wav hop_p95=15.0ms; handoff delta stays small versus Talker cadence ; decode=12.7ms / 0.1%; collect=27.8ms | long-text speech guardrail / Do not replace full-set or Video-AMME headline with synthetic-only evidence. |
| Synthetic long text c=4 | sglang | QPS=0.227; lat_p95=18.025s; RTF_p95=0.3373; words=139; audio_mean=52.6s | synthetic speech isolates thinker/talker/code2wav; no Video-AMME preprocessing queue claim ; talker=17463.9ms / 99.5%; RTF below 1.0 | talker->code2wav hop_p95=20.4ms; handoff delta stays small versus Talker cadence ; decode=14.0ms / 0.1%; collect=56.1ms | long-text speech guardrail / Do not replace full-set or Video-AMME headline with synthetic-only evidence. |
| Synthetic long text c=8 | sglang | QPS=0.303; lat_p95=26.318s; RTF_p95=0.5001; words=139; audio_mean=52.3s | synthetic speech isolates thinker/talker/code2wav; no Video-AMME preprocessing queue claim ; talker=25572.3ms / 99.1%; RTF below 1.0 | talker->code2wav hop_p95=24.0ms; handoff delta stays small versus Talker cadence ; decode=14.5ms / 0.1%; collect=81.7ms | long-text speech guardrail / Do not replace full-set or Video-AMME headline with synthetic-only evidence. |
| vLLM-c4 | vllm | QPS=0.154; lat_p95=3.525s; RTF_p95=3.0717 | runner_overhead=76.7%; admission_p95=19135.8ms ; thinker->talker p95=1.0ms; talker->code2wav drain_p95=17.5ms | prompt-feed dominates before engine boundaries ; talker/code2wav drain_p95=17.5ms | offline prompt-feed diagnostic / Do not promote offline diagnostic rows to online serving parity without online ingress plus WER/ASR. |
| vLLM-c8 | vllm | QPS=0.162; lat_p95=3.260s; RTF_p95=3.1987 | runner_overhead=81.8%; admission_p95=43972.7ms ; thinker->talker p95=1.0ms; talker->code2wav drain_p95=16.0ms | prompt-feed dominates before engine boundaries ; talker/code2wav drain_p95=16.0ms | offline prompt-feed diagnostic / Do not promote offline diagnostic rows to online serving parity without online ingress plus WER/ASR. |
| vLLM-c8-prebuild-w1 | vllm | QPS=0.539; lat_p95=7.009s; RTF_p95=6.2581 | runner_overhead=77.6%; admission_p95=5425.0ms ; thinker->talker p95=4.0ms; talker->code2wav drain_p95=88.7ms | prebuild removes most admission span and exposes later engine/talker tail ; talker/code2wav drain_p95=88.7ms | optimized offline diagnostic / Do not promote offline diagnostic rows to online serving parity without online ingress plus WER/ASR. |
| vLLM-c8-prebuild-w4 | vllm | QPS=0.536; lat_p95=7.730s; RTF_p95=7.0869 | runner_overhead=65.6%; admission_p95=4891.5ms ; thinker->talker p95=3.9ms; talker->code2wav drain_p95=123.4ms | prebuild removes most admission span and exposes later engine/talker tail ; talker/code2wav drain_p95=123.4ms | optimized offline diagnostic / Do not promote offline diagnostic rows to online serving parity without online ingress plus WER/ASR. |

### 6.2 SGLang Video-AMME stage latency budget

| c | Lat Mean | RTF | QPS | Preproc lifecycle | Queue est | Talker avg | Code2wav decode | Hop p95 | Diagnosis |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 1316.0ms | 1.0490 | 0.760 | 294.8ms / 22.4% | 无排队估计 | 444.1ms / 33.7% | 14.3ms / 1.1% | 15.5ms | talker_ar_tail |
| 2 | 1508.0ms | 1.0816 | 1.315 | 334.1ms / 22.2% | 无排队估计 | 525.8ms / 34.9% | 16.0ms / 1.1% | 16.1ms | talker_ar_tail |
| 4 | 1929.0ms | 1.4015 | 2.036 | 486.9ms / 25.2% | 191.1ms / 9.9% | 663.1ms / 34.4% | 16.0ms / 0.8% | 17.8ms | talker_ar_tail |
| 8 | 3064.0ms | 2.2141 | 2.540 | 1226.6ms / 40.0% | 937.4ms / 30.6% | 982.7ms / 32.1% | 17.2ms / 0.6% | 20.4ms | admission_queue_plus_talker_tail |
| 16 | 6066.0ms | 4.8489 | 2.407 | 4395.1ms / 72.5% | 4090.5ms / 67.4% | 815.6ms / 13.4% | 16.7ms / 0.3% | 19.7ms | saturation_boundary |

读法：低/中并发下 talker AR 占比约三分之一；c=8 时 queue estimate 已占 latency 的 30.6%，但仍是吞吐峰值；c=16 queue estimate 到 67.4%，所以是 saturation boundary。

### 6.3 短/长文本语音 stage latency budget

| Scenario | c | Audio | Lat Mean | RTF | QPS | Talker avg | Code2wav decode | Hop p95 | Diagnosis |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| long | 1 | 51.9s | 9168.0ms | 0.1766 | 0.109 | 9091.2ms / 99.2% | 12.7ms / 0.1% | 15.0ms | faster_than_realtime |
| long | 4 | 52.6s | 17551.0ms | 0.3338 | 0.227 | 17463.9ms / 99.5% | 14.0ms / 0.1% | 20.4ms | faster_than_realtime |
| long | 8 | 52.3s | 25799.0ms | 0.4932 | 0.303 | 25572.3ms / 99.1% | 14.5ms / 0.1% | 24.0ms | long_speech_talker_ar_dominant_but_faster_than_realtime |
| short | 1 | 4.2s | 866.0ms | 0.2052 | 1.154 | 849.0ms / 98.0% | 12.6ms / 1.5% | 14.9ms | faster_than_realtime |
| short | 4 | 4.3s | 1768.0ms | 0.4105 | 2.218 | 1722.6ms / 97.4% | 16.5ms / 0.9% | 20.1ms | faster_than_realtime |
| short | 8 | 4.3s | 2638.0ms | 0.6257 | 2.983 | 2205.9ms / 83.6% | 15.3ms / 0.6% | 21.2ms | faster_than_realtime |

读法：synthetic short/long 把 thinker/talker/code2wav 路径隔离出来。短文本和长文本的 code2wav decode 占比都很小；长文本 c=8 的 RTF=0.4932，仍快于实时，瓶颈描述应落在 talker AR cadence 而不是 vocoder decode。

### 6.4 vLLM offline stage latency budget

| Workload | Runner QPS | Engine QPS | Runner overhead | Admission avg/p95 | Encoder p95 | Thinker->Talker p95 | Talker->C2W p95 | Scope | Diagnosis |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| vLLM-c4 | 0.1536 | 0.1536 | 76.7% | 15110.8ms/19135.8ms | 44.2ms | 1.0ms | 17.5ms | strict_c4_only | prompt_feed_limited |
| vLLM-c8 | 0.1622 | 0.1622 | 81.8% | 33314.0ms/43972.7ms | 43.8ms | 1.0ms | 16.0ms | offline_diagnostic_only | prompt_feed_limited |
| vLLM-c8-prebuild-w1 | 0.1420 | 0.5391 | 77.6% | 4439.7ms/5425.0ms | 41.3ms | 4.0ms | 88.7ms | offline_diagnostic_only | engine_or_workload_limited |
| vLLM-c8-prebuild-w4 | 0.2127 | 0.5360 | 65.6% | 4089.0ms/4891.5ms | 46.2ms | 3.9ms | 123.4ms | offline_diagnostic_only | engine_or_workload_limited |

读法：vLLM original c=4/c=8 的 p95 encoder、thinker->talker、talker->code2wav 并不大，主问题在 offline runner 到 engine admission 的 prompt build/feed。prebuild w4 把 admission span 降下来后，才暴露后续 engine/talker/code2wav tail，因此它是诊断证据，不是线上 parity 证据。

### 6.5 Stage route verdict

| Route | Runtime | 裁决 | 优化重点 | 对外安全说法 |
| --- | --- | --- | --- | --- |
| admission -> preprocessing | sglang | recommended_window_with_saturation_guard | Keep c4-c8 as the serving window; treat c16 as queue/admission saturation, not a better high-concurrency point. | 可以说当前高并发瓶颈是 admission/queue 与 talker tail 叠加；不能把 c16 包装成推荐配置。 |
| admission -> preprocessing -> talker -> code2wav | sglang | talker_ar_tail | Optimize Talker AR cadence/batching before code2wav compute for these regimes. | 可以说低/中并发主要是 talker AR tail；code2wav 不是优先优化对象。 |
| admission -> preprocessing_queue -> preprocessing -> talker -> code2wav | sglang | recommended_window_with_saturation_guard | Keep c4-c8 as the serving window; treat c16 as queue/admission saturation, not a better high-concurrency point. | 可以说当前高并发瓶颈是 admission/queue 与 talker tail 叠加；不能把 c16 包装成推荐配置。 |
| code2wav_collect -> code2wav_decode | sglang | decode_not_bottleneck | Do not optimize vocoder decode first; collect wait reflects Talker chunk cadence. | 可以说 code2wav decode 稳定且小；collect wait 是等待 chunk，不是 vocoder 算力瓶颈。 |
| offline_runner -> engine_admission | vllm | vllm_offline_prompt_feed_limited | Use prebuilt prompts or online ingress before making c8 cross-runtime parity claims. | 可以说 vLLM original c8 被 offline prompt build/feed admission 限制；不能说这是 online serving parity。 |
| offline_runner -> engine_admission -> encoder -> thinker -> talker -> code2wav | vllm | optimized_offline_diagnostic | Use prebuild w4 as the strongest offline diagnostic; require online ingress plus WER/ASR before parity replacement. | 可以说 prebuild w4 解除主要 admission 问题并暴露 engine/talker tail；不能当作最终线上 parity。 |
| talker -> code2wav_stream | sglang | handoff_healthy | Do not optimize the stream handoff first; p95 hop stays small across measured SGLang pressure. | 可以说 SGLang talker->code2wav 连接健康；当前瓶颈不在连接本身。 |

关键结论：不要把健康 handoff 说成瓶颈。`talker -> code2wav_stream` 和 `code2wav_collect -> code2wav_decode` 当前主要是健康/非瓶颈；c=8/c=16 的压力来自 admission/queue 与 talker tail 叠加。

### 6.6 Pressure transition 矩阵

这张表把相邻压力档位的变化量直接摊开，用来回答压力到底传到了哪个 stage 边界。
它来自 `stage_boundary_bottleneck_ledger.json`，不引入新的 benchmark 数字。

| ID | Runtime | Workload | Transition | Verdict | Key evidence | Decision |
| --- | --- | --- | --- | --- | --- | --- |
| pressure-sglang-c1-c2 | sglang | Video-AMME ci-50 | c1 -> c2 | scales_without_boundary_bottleneck | QPS +73.0%; latency_p95 +29.8%; RTF_p95 -4.4%; queue_delta=无排队估计; queue_share_after=无排队估计; hop_p95_delta +0.6ms; decode_p95_delta +2.4ms | Throughput improves without moving the main bottleneck to stage handoff or decode. |
| pressure-sglang-c2-c4 | sglang | Video-AMME ci-50 | c2 -> c4 | scales_without_boundary_bottleneck | QPS +54.8%; latency_p95 +16.3%; RTF_p95 +29.4%; queue_delta=无排队估计; queue_share_after 39.3%; hop_p95_delta +1.7ms; decode_p95_delta +1.6ms | Throughput improves without moving the main bottleneck to stage handoff or decode. |
| pressure-sglang-c4-c8 | sglang | Video-AMME ci-50 | c4 -> c8 | usable_high_concurrency_window | QPS +24.8%; latency_p95 +61.1%; RTF_p95 +75.8%; queue_delta +746.3ms; queue_share_after 76.4%; hop_p95_delta +2.6ms; decode_p95_delta +4.1ms | Keep c8 as the throughput-oriented serving edge; queue pressure is visible but throughput still improves. |
| pressure-sglang-c8-c16 | sglang | Video-AMME ci-50 | c8 -> c16 | saturation_boundary | QPS -5.2%; latency_p95 +34.1%; RTF_p95 +137.0%; queue_delta +3153.1ms; queue_share_after 93.1%; hop_p95_delta -0.7ms; decode_p95_delta -2.0ms | Do not use c16 as the serving optimum: throughput falls while admission queue and p95 RTF rise. |
| pressure-synthetic-short-c1-c4 | sglang | synthetic_short | c1 -> c4 | short_text_scales_below_realtime | QPS +92.2%; latency_p95 +122.5%; RTF_p95_after 0.4; talker_p95_delta +1125.1ms; hop_p95_delta +5.2ms; decode_avg_delta +3.9ms | Synthetic speech remains below real-time; the handoff delta stays small, so length pressure maps to Talker cadence rather than code2wav decode. |
| pressure-synthetic-short-c4-c8 | sglang | synthetic_short | c4 -> c8 | short_text_scales_below_realtime | QPS +34.5%; latency_p95 +37.5%; RTF_p95_after 0.7; talker_p95_delta +472.8ms; hop_p95_delta +1.1ms; decode_avg_delta -1.2ms | Synthetic speech remains below real-time; the handoff delta stays small, so length pressure maps to Talker cadence rather than code2wav decode. |
| pressure-synthetic-long-c1-c4 | sglang | synthetic_long | c1 -> c4 | long_text_realtime_guard_holds | QPS +108.3%; latency_p95 +90.4%; RTF_p95_after 0.3; talker_p95_delta +8563.3ms; hop_p95_delta +5.4ms; decode_avg_delta +1.3ms | Synthetic speech remains below real-time; the handoff delta stays small, so length pressure maps to Talker cadence rather than code2wav decode. |
| pressure-synthetic-long-c4-c8 | sglang | synthetic_long | c4 -> c8 | long_text_realtime_guard_holds | QPS +33.5%; latency_p95 +46.0%; RTF_p95_after 0.5; talker_p95_delta +8251.4ms; hop_p95_delta +3.6ms; decode_avg_delta +0.5ms | Synthetic speech remains below real-time; the handoff delta stays small, so length pressure maps to Talker cadence rather than code2wav decode. |
| pressure-vllm-vLLM-c4-to-vLLM-c8 | vllm | Video-AMME ci-50 offline | vLLM-c4 -> vLLM-c8 | offline_prompt_feed_limited | wall_qps +5.6%; engine_qps +5.6%; admission_p95 +129.8%; runner_overhead +5.1pp; talker_drain_p95_delta -1.5ms; after_diagnosis=prompt_feed_limited | Do not use original c8 wall QPS as online parity; admission span grows sharply. |
| pressure-vllm-vLLM-c8-to-vLLM-c8-prebuild-w4 | vllm | Video-AMME ci-50 offline | vLLM-c8 -> vLLM-c8-prebuild-w4 | diagnostic_bottleneck_shift | wall_qps +31.1%; engine_qps +230.4%; admission_p95 -88.9%; runner_overhead -16.2pp; talker_drain_p95_delta +107.4ms; after_diagnosis=engine_or_workload_limited | Prebuild removes most admission span and exposes later engine/talker tail; keep offline caveat. |
| pressure-vllm-vLLM-c8-prebuild-w1-to-vLLM-c8-prebuild-w4 | vllm | Video-AMME ci-50 offline | vLLM-c8-prebuild-w1 -> vLLM-c8-prebuild-w4 | runner_parallelism_helps_wall_not_engine | wall_qps +49.8%; engine_qps -0.6%; admission_p95 -9.8%; runner_overhead -12.0pp; talker_drain_p95_delta +34.7ms; after_diagnosis=engine_or_workload_limited | w4 improves runner wall time while engine QPS stays flat; not a serving-parity headline. |

读法：SGLang `c8 -> c16` 是 saturation boundary，因为吞吐下降但 admission queue 和 RTF tail 上升；synthetic long `c4 -> c8` 仍快于实时，说明长文本压力主要进入 Talker cadence；vLLM `c8 -> prebuild-w4` 是 offline diagnostic 的瓶颈转移，不能提升为 online parity。

### 6.7 Route 复现索引

| Route | Stage rows | Raw artifact anchors | Rerun command IDs | jq entry |
| --- | ---: | --- | --- | --- |
| admission -> preprocessing | 5 | results/qwen35_sglang_mr8_stress_20260619/benchmark_audio_50_c1_warm_profile_skipwer/videoamme_results.json, results/qwen35_sglang_mr8_stress_20260619/request_profile_c1_warm_profile_skipwer.json, results/qwen35_sglang_mr8_stress_20260619/benchmark_audio_50_c2_warm_profile_skipwer/videoamme_results.json, ... (+7) | sglang_videoamme_stress, build_stage_interactions, build_stage_boundary_bottleneck_ledger, build_stage_drilldown_index, ... (+4) | jq '.rows[] \| select(.route_key == "admission -> preprocessing")' results/qwen35_report_audit_20260619/stage_reproduction_drilldown.json |
| admission -> preprocessing -> talker -> code2wav | 2 | results/qwen35_sglang_mr8_stress_20260619/benchmark_audio_50_c1_warm_profile_skipwer/videoamme_results.json, results/qwen35_sglang_mr8_stress_20260619/request_profile_c1_warm_profile_skipwer.json, results/qwen35_sglang_mr8_stress_20260619/benchmark_audio_50_c2_warm_profile_skipwer/videoamme_results.json, ... (+1) | sglang_videoamme_stress, build_report_tables, build_stage_latency_budget, build_stage_drilldown_index, ... (+4) | jq '.rows[] \| select(.route_key == "admission -> preprocessing -> talker -> code2wav")' results/qwen35_report_audit_20260619/stage_reproduction_drilldown.json |
| admission -> preprocessing_queue -> preprocessing -> talker -> code2wav | 3 | results/qwen35_sglang_mr8_stress_20260619/benchmark_audio_50_c4_profile_skipwer/videoamme_results.json, results/qwen35_sglang_mr8_stress_20260619/request_profile_c4_profile_skipwer.json, results/qwen35_sglang_mr8_stress_20260619/benchmark_audio_50_c8_profile_skipwer/videoamme_results.json, ... (+3) | sglang_videoamme_stress, build_report_tables, build_stage_latency_budget, build_stage_drilldown_index, ... (+4) | jq '.rows[] \| select(.route_key == "admission -> preprocessing_queue -> preprocessing -> talker -> code2wav")' results/qwen35_report_audit_20260619/stage_reproduction_drilldown.json |
| code2wav_collect -> code2wav_decode | 5 | results/qwen35_sglang_mr8_stress_20260619/benchmark_audio_50_c1_warm_profile_skipwer/videoamme_results.json, results/qwen35_sglang_mr8_stress_20260619/request_profile_c1_warm_profile_skipwer.json, results/qwen35_sglang_mr8_stress_20260619/benchmark_audio_50_c2_warm_profile_skipwer/videoamme_results.json, ... (+7) | sglang_videoamme_stress, build_stage_interactions, build_stage_boundary_bottleneck_ledger, build_stage_drilldown_index, ... (+4) | jq '.rows[] \| select(.route_key == "code2wav_collect -> code2wav_decode")' results/qwen35_report_audit_20260619/stage_reproduction_drilldown.json |
| offline_runner -> engine_admission | 5 | results/qwen35_vllm_videoamme_ci50_offline_compile_c1_mns8_20260619_20260619_220617/benchmark_audio_50_c1_offline_compile/videoamme_results.json, results/qwen35_vllm_videoamme_ci50_offline_compile_c1_mns8_20260619_20260619_220617/run.log, results/qwen35_vllm_videoamme_ci50_official_talker_compile_c4_20260618_202106/benchmark_audio_50_c4_offline_compile/videoamme_results.json, ... (+7) | vllm_c1_original, summarize_vllm_log_stages, diagnose_vllm_admission, build_stage_interactions, ... (+8) | jq '.rows[] \| select(.route_key == "offline_runner -> engine_admission")' results/qwen35_report_audit_20260619/stage_reproduction_drilldown.json |
| offline_runner -> engine_admission -> encoder -> thinker -> talker -> code2wav | 4 | results/qwen35_vllm_videoamme_ci50_official_talker_compile_c4_20260618_202106/benchmark_audio_50_c4_offline_compile/videoamme_results.json, results/qwen35_vllm_videoamme_ci50_official_talker_compile_c4_20260618_202106/run.log, results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_20260619_20260619_222434/benchmark_audio_50_c8_offline_compile/videoamme_results.json, ... (+5) | vllm_c1_original, summarize_vllm_log_stages, diagnose_vllm_admission, build_stage_latency_budget, ... (+7) | jq '.rows[] \| select(.route_key == "offline_runner -> engine_admission -> encoder -> thinker -> talker -> code2wav")' results/qwen35_report_audit_20260619/stage_reproduction_drilldown.json |
| preprocessing -> encoder_thinker | 1 | results/qwen35_sglang_mr8_stress_20260619/benchmark_audio_50_c8_profile_skipwer/videoamme_results.json, results/qwen35_sglang_mr8_stress_20260619/request_profile_c8_profile_skipwer.json | sglang_videoamme_stress, build_stage_interactions, build_stage_boundary_bottleneck_ledger, build_stage_drilldown_index, ... (+4) | jq '.rows[] \| select(.route_key == "preprocessing -> encoder_thinker")' results/qwen35_report_audit_20260619/stage_reproduction_drilldown.json |
| talker -> code2wav | 5 | results/qwen35_vllm_videoamme_ci50_offline_compile_c1_mns8_20260619_20260619_220617/benchmark_audio_50_c1_offline_compile/videoamme_results.json, results/qwen35_vllm_videoamme_ci50_offline_compile_c1_mns8_20260619_20260619_220617/run.log, results/qwen35_vllm_videoamme_ci50_official_talker_compile_c4_20260618_202106/benchmark_audio_50_c4_offline_compile/videoamme_results.json, ... (+7) | vllm_c1_original, summarize_vllm_log_stages, diagnose_vllm_admission, build_stage_interactions, ... (+8) | jq '.rows[] \| select(.route_key == "talker -> code2wav")' results/qwen35_report_audit_20260619/stage_reproduction_drilldown.json |
| talker -> code2wav_stream | 11 | results/qwen35_sglang_mr8_stress_20260619/benchmark_audio_50_c1_warm_profile_skipwer/videoamme_results.json, results/qwen35_sglang_mr8_stress_20260619/request_profile_c1_warm_profile_skipwer.json, results/qwen35_sglang_mr8_stress_20260619/benchmark_audio_50_c2_warm_profile_skipwer/videoamme_results.json, ... (+19) | sglang_videoamme_stress, build_stage_interactions, build_stage_boundary_bottleneck_ledger, build_stage_drilldown_index, ... (+5) | jq '.rows[] \| select(.route_key == "talker -> code2wav_stream")' results/qwen35_report_audit_20260619/stage_reproduction_drilldown.json |

读法：每条 route 都可以沿 `stage_route_decision_matrix.json` 回到 `stage_reproduction_drilldown.json`、raw artifact 和 rerun command ID；这也是报告里 stage 结论可复核的最短路径。

## 7. vLLM c=8 诊断边界

| Pressure | 决策 | Stage/瓶颈判断 | 关键数字 |
| --- | --- | --- | --- |
| original c=8 | 诊断证据 | 原始 vLLM c=8 主要是 host prompt build/feed admission 受限，不是 online parity。 | runner_overhead=81.8%, admission_avg=33314.0ms, engine_QPS=0.1622 |
| prebuild c=8 workers=4 | 优化后 offline 诊断 | prebuild w4 是当前最强 vLLM offline 诊断；仍需 online ingress + WER 才能做 c=8 parity。 | prompt_wall=129.2s, runner_QPS=0.2127, engine_QPS=0.5360, admission_avg=4089.0ms |

vLLM c=8 original 的 offline runner 会在 engine admission 前本地构建 multimodal prompt，因此原始 wall QPS 主要反映 prompt build/feed。prebuild w4 显著缩短 admission span，但仍缺 online ingress + WER/ASR 复核，所以不能升级为 strict c=8 online parity。

## 8. 优化锁和反例

- SGLang optimization lock：ready=`True`，checks=`26/26`，推荐窗口 `c4-c8 warmed serving; c8 peak throughput; c16 saturation boundary`。
- vLLM optimization lock：ready=`True`，checks=`22/22`，c=8 边界 `prebuild w4 is optimized offline diagnostic, not online parity`。
- vLLM online parity protocol：ready=`True`，online_parity_proven=`False`。

### 8.1 当前 best measured recipe 裁决

| 主题 | 当前裁决 | 可说法 | 不能说 | 替换条件 |
| --- | --- | --- | --- | --- |
| SGLang recipe | 当前审计环境里的 best measured recipe | compiled/graph path + serial preprocessing + 16GiB preprocessing cache；服务窗口 c=4-c=8，c=8 是当前吞吐峰值 | 不能说已经搜索完所有 future kernel、placement、admission policy，不能承诺任意环境全局最优 | 新 recipe 必须补齐 c=4/c=8/c=16、WER、stage interaction、acceptance、final readiness 后才能替换 |
| SGLang anti-recipe | preproc=2/4 当前不能作为优化方向 | preproc=2 回退，preproc=4 失败/OOM；当前应先管 admission、placement 和 shared-resource contention | 不能把 preprocessing 并发越高越好当成优化结论 | 只有 admission/placement/memory 同步重设计后重新评估 |
| vLLM baseline | optimized baseline，不是弱 baseline | Qwen3.5-capable image + compile/CUDA graph/cache/prebuild evidence；c=4 用于 strict headline，c=8 prebuild w4 是 offline diagnostic | 不能把 c=8 prebuild w4 说成 online serving parity | 需要 online ingress + WER/ASR + stage boundary 复核后才能升级 |

| Pressure | 决策 | Stage/瓶颈判断 | 关键数字 |
| --- | --- | --- | --- |
| PREPROCESSING_MAX_CONCURRENCY=2 at c=8 | 反例：性能回退 | preproc=2 把 admission 问题转成共享资源 contention，不能作为当前优化方向。 | baseline_QPS=2.540, preproc2_QPS=1.642, baseline_lat=3.064s, preproc2_lat=4.579s |
| PREPROCESSING_MAX_CONCURRENCY=4 at c=8 | 反例：失败/OOM | preproc=4 有失败/OOM 风险，当前 recipe 禁用。 | failed=7, accuracy=60.0% |

优化顺序建议：先守住 compiled/graph recipe 和 serial preprocessing，再优化 talker AR cadence/batching、admission 策略和 online vLLM parity 入口；不要把 preproc=2/4 当作当前 recipe。

## 9. 复现入口

最短复核命令：

```bash
cd /home/gangouyu/sglang-omni

python3 -m benchmarks.eval.run_qwen35_omni_report_audit \
  --root /home/gangouyu/sglang-omni \
  --summary-output results/qwen35_report_audit_20260619/audit_run_summary.json
```

- Repro command manifest：ready=`True`，commands=`63`，phases=`7`。
- SGLang/vLLM 具体重跑命令：`benchmarks/reports/qwen35_omni_reproduction_checklist_zh_20260621.md`。
- 接收方一条命令收包快检：`bash benchmarks/eval/qwen35_omni_receiver_quickcheck.sh`；它串起 checksum、tarball-mode validation、receiver smoke、extracted-only validation 和 external standalone validation。
- 单项 share package validator：`python3 -m benchmarks.eval.validate_qwen35_omni_share_package --root /home/gangouyu/sglang-omni --strict --json-output results/qwen35_report_audit_20260619/share_package_validation.json`。
- 现场只读证据查询 smoke：`bash benchmarks/eval/qwen35_omni_evidence_query_cards_smoke.sh --root /home/gangouyu/sglang-omni --mode host`；解包后用 `bash "$BUNDLE_ROOT/benchmarks/eval/qwen35_omni_evidence_query_cards_smoke.sh" --root "$BUNDLE_ROOT" --mode portable`。
- Release seal：ready=`True`，checks=`14/14`，tarball identity 由 `results/qwen35_report_audit_20260619/share_release_seal.json` 记录和校验，evidence-query host/portable pass=`21/19`。

关键命令可以直接从 machine-readable manifest 抽取，避免复制长命令时改错参数：

```bash
cd /home/gangouyu/sglang-omni
jq -r --arg id vllm_c4_original \
  '.commands[] | select(.id == $id) | .command' \
  results/qwen35_report_audit_20260619/repro_command_manifest.json
```

| Scope | Command ID | Phase | 期望结果 |
| --- | --- | --- | --- |
| 全量证据验证 | `run_full_audit` | audit_first | ok=true; claims 17/17; coverage 34/34; preflight 62 checks with 0 required failures; manifest >=180 records with 0 missing; chart source consistency ready=true; objective completion audit has 0 required failures; final readiness ready=true; runtime image contract ready=true; rerun acceptance contract ready=true; checkpoint watchlist ready=true; stage latency budget ready=true; stage boundary bottleneck ledger ready=true; metric provenance index ready=true; stage reproduction drilldown ready=true; claim metric crosswalk ready=true; objective requirement crosswalk ready=true with 8 optimization candidates; share bundle ready=true; tarball validation 17/17; receiver smoke ready=true; extracted-only validation 13/13; standalone validation 8/8; receiver quickcheck contract 15/15 with WER/ASR path guard, evidence-query CLI/docs, stage dictionary crosswalk, and final completion route; final completion audit ready=true with 0 required failures; release seal ready=true. |
| SGLang 服务 | `launch_sglang_optimized` | sglang_serving | Server listens on port 8161 after warmup; c<=8 is the recommended serving envelope for this recipe. |
| SGLang c=1/2/4/8/16 | `sglang_videoamme_stress` | sglang_stress | c=8 is the current throughput peak; c=16 is saturation evidence; code2wav remains non-bottleneck. |
| SGLang 短/长文本语音 | `sglang_synthetic_text_to_speech` | sglang_stress | Short input remains 74 chars / 12 words; long input remains 944 chars / 139 words; long c=8 RTF stays below 1.0. |
| SGLang WER | `sglang_recompute_wer` | quality_validation | Corpus WER remains stable across c=1/2/4/8/16 and does not trade quality for throughput. |
| vLLM c=1 baseline | `vllm_c1_original` | vllm_baseline | Completes Video-AMME ci-50 with the Qwen3.5-capable vLLM image and max_num_seqs=8. |
| vLLM strict c=4 baseline | `vllm_c4_original` | vllm_baseline | Completes Video-AMME ci-50 with the Qwen3.5-capable vLLM image; warmed c=4 is the strict apples-to-apples headline comparison slice. |
| vLLM c=8 original diagnostic | `vllm_c8_original` | vllm_baseline | Original c=8 remains prompt-feed/admission limited and is not used as online serving parity. |
| vLLM c=8 prebuild w4 | `vllm_c8_prebuild_w4` | vllm_baseline | Completes 50/50; runner QPS improves versus prebuild w1; engine QPS remains around 0.536 in the current checkpoint. |
| vLLM stage log 汇总 | `summarize_vllm_log_stages` | audit_regeneration | vLLM log-stage rows exist for c1/c4/c8 and prebuild w1/w4. |
| vLLM admission 诊断 | `diagnose_vllm_admission` | audit_regeneration | Original c4/c8 are prompt-feed limited; prebuild w4 is the optimized offline diagnostic. |
| Stage latency budget | `build_stage_latency_budget` | audit_regeneration | ready=true, 12/12 checks pass, with 5 SGLang, 6 synthetic, and 4 vLLM stage-budget rows. |
| Stage boundary ledger | `build_stage_boundary_bottleneck_ledger` | audit_regeneration | ready=true, 12/12 checks pass, all 37 stage boundary rows have evidence, decision, and claim scope, and 11 pressure transition rows cover concurrency, long/short text, and vLLM diagnostics. |
| Stage drilldown 复现 | `build_stage_reproduction_drilldown` | audit_regeneration | ready=true, 52 stage rows, at least 11 route rows, raw artifacts, metric row links, jq queries, and rerun command IDs are present. |
| Stage route 裁决 | `build_stage_route_decision_matrix` | audit_regeneration | ready=true, 11 route rows, 52 covered stage rows, route-level decisions, safe talking points, raw artifacts, jq queries, rerun command IDs, and the 15-row pressure-stage heatmap are present. |
| build_tail_confidence_appendix | `build_tail_confidence_appendix` | audit_regeneration | ready=true, 13/13 checks pass, 18 per-sample distribution rows and 9 bootstrap comparison rows cover strict c4, stress, synthetic, and vLLM diagnostic cases. |

## 10. 可分享边界

- 可以说：warmed c=4 下 SGLang-Omni 优于优化版 vLLM，并且 accuracy/WER 不退化。
- 可以说：当前 SGLang 推荐 c=4-c=8，c=8 是吞吐峰值，c=16 是压力边界。
- 可以说：stage handoff 当前健康，code2wav decode 不是优先瓶颈。
- 不要说：vLLM c=8 prebuild w4 已证明 online parity。
- 不要说：preprocessing 并发越高越好。
- 不要说：官方 SeedTTS full-set 已是 headline evidence；当前只提供本地 smoke path。

### 10.1 现场答辩速查

这张表来自 `defense_claim_matrix.json`，用于现场把“能说什么、怎么复跑、失败时怎么撤回”放在同一个口径里。
同一个 JSON 还包含 13 个 Q&A 问题到 10 条 defense claim 的 `qna_question_rows` 映射，
方便从现场提问一跳进入机器证据、复跑命令和撤回条件。
完整展开版在 `benchmarks/reports/qwen35_omni_defense_qna_zh_20260621.md`。

快速抽取命令：

```bash
jq '.rows[] | {claim, allowed_wording, rerun_command_ids, failure_decision}' \
  results/qwen35_report_audit_20260619/defense_claim_matrix.json
```

| 现场追问 | 可说法 | Evidence | 复跑命令/入口 | 失败时裁决 |
| --- | --- | --- | --- | --- |
| SGLang warmed c=4 优于优化版 vLLM | 当前 8x H20、Video-AMME ci-50、warmed c=4 严格对比中，SGLang latency/RTF 更好且质量不退化。 | claims_verification.json, headline_scorecard.json, qwen35_omni_runtime_comparison_contract_zh_20260621.md | `run_full_audit`, `vllm_c1_original`, `vllm_c4_original`, `vllm_c8_original`, ... (+1) | claims 或 headline 失败时不得沿用 headline，进入 rerun acceptance 替换评审。 |
| vLLM baseline 不是弱 baseline | vLLM 使用 Qwen3.5-capable 镜像和 compile/CUDA graph、prefix/chunked prefill、shared-memory transfer、encoder compile/batch 等优化证据。 | vllm_optimization_lock.json, runtime_image_contract.json, vllm_log_stage_summary.json | `build_vllm_optimization_lock`, `build_runtime_image_contract`, `summarize_vllm_log_stages`, `vllm_c1_original`, ... (+1) | image/optimization lock 失败时只能说现有 vLLM 证据不可复核，不能说 baseline 公平。 |
| SGLang c=8 是当前高并发峰值 | c=8 是当前 recipe 的吞吐峰值，c=16 是压力边界，不是推荐默认点。 | acceptance_matrix.json, stage_latency_budget.json, stage_interaction_summary.json | `sglang_videoamme_stress`, `build_acceptance_matrix`, `build_stage_latency_budget` | c=8 不再为峰值时先按 rerun delta triage 定位 admission/queue，不直接改主报告数字。 |
| short/long text-to-speech 已覆盖 | short 74 chars / 12 words，long 944 chars / 139 words，c=1/4/8 均覆盖，long c=8 仍快于实时。 | length_regime_coverage.json, tables_summary.json, stage_latency_budget.json, ... (+1) | `sglang_synthetic_text_to_speech`, `build_report_tables`, `build_stage_latency_budget` | 输入形状或 long c=8 RTF 失败时，相关长短文结论不得替换或外推。 |
| stage handoff 没有卡住 | talker 到 code2wav 的 stream hop p95 约 15-24ms，当前不是主瓶颈。 | stage_interaction_summary.json, stage_boundary_bottleneck_ledger.json, qwen35_omni_stage_causal_graph_zh_20260621.md | `build_stage_interactions`, `build_stage_boundary_bottleneck_ledger`, `build_stage_causal_graph` | handoff health 失败时，不能继续说 stage 连接健康，先补 profile drilldown。 |
| code2wav decode 不是当前 compute bottleneck | decode 平均约 14-17ms/window，collect wait 更多是在等 talker chunk cadence。 | claims_verification.json, stage_latency_budget.json, stage_boundary_bottleneck_ledger.json | `sglang_videoamme_stress`, `sglang_synthetic_text_to_speech`, `build_stage_boundary_bottleneck_ledger` | decode 成为主项时，当前 code2wav-not-bottleneck 结论必须撤回或重写。 |
| 朴素提高 preprocessing 并发是负优化 | preproc=2 回退，preproc=4 失败；当前应先管 admission、placement 和 shared-resource contention。 | acceptance_matrix.json, sglang_optimization_lock.json, stage_interaction_summary.json | `build_sglang_optimization_lock`, `build_acceptance_matrix`, `build_stage_boundary_bottleneck_ledger` | 新候选 recipe 必须补齐 c=4/c=8/c=16、WER、profile 和稳定性证据后再评审。 |
| vLLM c=8 prebuild w4 只是 offline diagnostic | prebuild w4 改善 runner prompt build/feed，但没有证明 online serving parity。 | vllm_admission_diagnosis.json, vllm_online_parity_protocol.json, qwen35_omni_runtime_comparison_contract_zh_20260621.md | `vllm_c8_prebuild_w4`, `build_vllm_online_parity_protocol`, `build_runtime_comparison_contract` | online_parity_proven=false 时不得把 c=8 prebuild 写成 online parity。 |
| WER/quality 没有为性能让步 | SGLang stress WER 稳定，strict c=4 WER/accuracy 不劣于 vLLM。 | claims_verification.json, headline_scorecard.json, acceptance_matrix.json | `sglang_recompute_wer`, `verify_report_claims`, `build_headline_scorecard` | WER 或 ASR 路径不一致时，不得只替换 latency/RTF headline。 |
| 当前包可分享但仍有边界 | share-ready 是带 caveat 的阶段稿；更大数据/真实流量外推、SeedTTS full-set 和 vLLM c=8 online parity 不能越界。 | final_readiness_audit.json, confidence_ledger.json, qwen35_omni_caveat_adjudication_matrix_zh_20260621.md | `run_full_audit`, `validate_share_bundle_package`, `validate_share_bundle_receiver_smoke` | 任一 package/readiness gate 失败时先修包，不讨论性能结论。 |

## 11. 证据入口

- 本报告 gate：ready=`True`，checks=`16/16`。
- Final readiness：ready=`True`，checks=`49/49`。
- Manifest：records=`196`，missing=`0`。
- Share bundle：ready=`True`，records=`122`。
- Share package validation：ready=`True`，checks=`17/17`。
- Receiver smoke validation：ready=`True`，checks=`17/17`。
- External standalone validation：ready=`True`，checks=`8/8`。
- Share release seal：ready=`True`，checks=`14/14`，adjacent hashed=`14/16`；完整 tarball identity 只在 release seal JSON 中校验。
- Evidence-query smoke：ready=`True`，host/portable pass=`21/19`。
- Defense claim matrix：ready=`True`，claim_rows=`10`，question_rows=`13`，failure_decisions=`10`。
- Tail confidence appendix：ready=`True`，rows=`18`，strict_c4_sglang_p95=`3.3280250000000002`，strict_c4_vllm_p95=`3.5251249999999996`。
- `results/qwen35_report_audit_20260619/share_package_validation.json`
- `results/qwen35_report_audit_20260619/share_package_receiver_smoke_validation.json`
- `results/qwen35_report_audit_20260619/share_package_external_standalone_validation.json`
- `results/qwen35_report_audit_20260619/share_release_seal.json`
- `results/qwen35_report_audit_20260619/evidence_query_cards_smoke_summary.json`
- `results/qwen35_report_audit_20260619/headline_scorecard.json`
- `results/qwen35_report_audit_20260619/regime_decision_matrix.json`
- `results/qwen35_report_audit_20260619/stage_latency_budget.json`
- `results/qwen35_report_audit_20260619/pressure_stage_heatmap.json`
- `results/qwen35_report_audit_20260619/stage_boundary_bottleneck_ledger.json`
- `results/qwen35_report_audit_20260619/stage_route_decision_matrix.json`
- `results/qwen35_report_audit_20260619/stage_reproduction_drilldown.json`
- `results/qwen35_report_audit_20260619/defense_claim_matrix.json`
- `results/qwen35_report_audit_20260619/metric_provenance_index.json`
- `results/qwen35_report_audit_20260619/tail_confidence_appendix.json`
- `results/qwen35_report_audit_20260619/repro_command_manifest.json`
- `benchmarks/reports/qwen35_omni_tail_confidence_appendix_zh_20260621.md`
- `benchmarks/reports/qwen35_omni_share_package_index_zh_20260621.md`
