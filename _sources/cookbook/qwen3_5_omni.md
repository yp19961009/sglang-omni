# Qwen3.5-Omni：架构、实现与延迟对比

本文整理自[《sglang-omni 支持 qwen35-omni》](https://alidocs.dingtalk.com/i/nodes/2Amq4vjg89jyZdNnCmvmkE3wW3kdP0wQ)，记录 Qwen3.5-Omni 的组件划分、框架接入方式和历史单请求性能测量。

性能数据对应下文列出的版本，不代表本仓库当前 HEAD 的测试结果；本文整理时没有重新运行这些测量。PAI-vLLM 修复跨进程拷贝前后的数据分别保留，比较时应区分两个基线。

## Qwen3.5-Omni 的组件

Qwen3.5-Omni 的在线推理包含以下组件，而非单个 LLM 推理过程。

| 组件 | 输入 | 输出 | 主要计算特征 |
| --- | --- | --- | --- |
| Processor | 文本、视频帧、音频 waveform | token、pixel values、audio features | CPU 密集，包含 token replacement 和特征处理 |
| Vision Encoder | 视频/图像 tensor | 视觉 embedding | GPU 计算密集，输入 payload 大 |
| Audio Encoder | 音频 feature | 音频 embedding | GPU 计算密集 |
| Thinker | token 和多模态 embedding | 文本 token、hidden state | 长 prompt prefill + AR decode |
| Talker | Thinker 文本/hidden state、codec feedback | 16 组 codec code | 带外部文本输入的反馈式 AR |
| Code2Wav | codec code stream | 24 kHz waveform | 流式窗口解码 |

## 框架的工程扩展方式

原文比较版本中的 PAI-vLLM 可复用 vLLM EngineCore、KV cache 和 continuous batching，适合固定拓扑下接入模型。这里关注的是继续增加组件和部署形态时的扩展成本。

PAI-vLLM 的主要扩展单元是定制 Engine；SGLang-Omni 的主要扩展单元是 Stage 和数据流拓扑。

| 维度 | PAI-vLLM（原文比较版本） | SGLang-Omni | 主要影响 |
| --- | --- | --- | --- |
| 拓扑与编排 | 在 `OmniLLMEngine` 中命令式连接各组件 | 用 `StageConfig` 声明路由、fan-in 和 stream | 新增分支时，前者修改中心 Engine，后者主要调整 Stage 配置与路由函数 |
| 调度与放置 | Thinker/Talker 使用独立 Engine，但 MM Encoder 与 Thinker 共享调度 | 每个 Stage 可拥有独立 Scheduler，并声明进程和 GPU | 各组件可以分别 batching、放置和调优 |

因此，SGLang-Omni 接入新模型时主要实现模型组件、Stage 拓扑和 Stage 间输入输出约定，通用请求生命周期、通信与编排能力由框架复用。

## Qwen3.5-Omni 的实现与优化

[模型实现代码](../../sglang_omni/models/qwen3_5_omni/)按以下流程拆分：

```text
Processor → Encoder → Thinker → Talker → Code2Wav
```

Processor、Encoder 和 Code2Wav 补齐模型专用逻辑；Thinker/Talker 注册为 SGLang 原生模型，并通过 ModelRunner/Scheduler 接入 forward。由此复用 SGLang 的 KV cache、continuous batching、sampling、CUDA graph/compile，以及 SGLang-Omni 的 Stage 编排、流式传输和请求生命周期管理。

## 历史性能测量

### 测量配置与版本

- 模型：23B，BF16。
- Thinker-only：单卡 H20。
- Thinker + Talker：Thinker 使用一张 H20，Talker + Code2Wav 使用另一张 H20，共两张。
- PAI-vLLM 原始基线：`origin/dev/qwenc_perf_v2`，commit `35d35d51bfd195436379275638aac122eea28f2e`。
- SGLang-Omni：`origin/develop`，commit `85eda8194a3be5fe846fb060052faafca9ad1691`。
- PAI-vLLM 更新后的基线：`dev/qwenc_perf_v2`，commit `7f8cf8cc0557ba7bf6f01d0de31065063c7388e0`，修复了跨进程拷贝问题。

以下均为单请求测量，延迟单位为毫秒（ms），token 数另行标注。“SGLang 变化”沿用原记录，以对应 PAI-vLLM 基线为参照，负值表示 SGLang-Omni 延迟更低。原表的差值和百分比可能因四舍五入与展示值直接计算的结果略有不同。

### Thinker-only：原始基线

| 区间 | SGLang-Omni | PAI-vLLM | SGLang 变化 |
| --- | ---: | ---: | --- |
| HF Processor | 119.442 | 138.794 | -13.9% |
| HF end → critical Encoder start | 57.859 | 292.000 | -80.2% |
| Critical/combined Encoder | 346.574 | 328.354 | +5.5% |
| Encoder end → first token | 375.996 | 339.667 | +10.7% |
| Post-media/HF start → first token | 899.871 | 1098.251 | -18.1% |
| Prompt / completion tokens | 8698 / 6 | 8698 / 4 | 原记录称文本输出相同 |

原记录将约 200 ms 的差距主要归因于 PAI-vLLM 前处理结果到 Encoder 之间的跨进程数据传输。这里的 `Post-media/HF start → first token` 从媒体处理之后的 HF 处理起点计时，不能直接当作完整请求的端到端 TTFT。两边 completion token 数不同，原记录的“文本输出相同”也不能作为相同生成 token 数的证明。

### Thinker + Talker：原始基线

| 区间 | SGLang-Omni | PAI-vLLM | SGLang 变化 |
| --- | ---: | ---: | --- |
| HF Processor | 103.5 | 139.4 | -35.9 ms |
| Request → first text token | 926.7 | 1169.1 | -242.3 ms（-20.7%） |
| First text → Talker prefill start | 17.8 | 20.9\* | -3.1 ms |
| Talker prefill start → first codec | 36.1 | 64.5 | -28.4 ms |
| First codec → fourth codec | 38.3 | 35.2 | +3.1 ms |
| Fourth codec → first audio | 37.3 | 23.6 | +13.7 ms |
| First text → first audio | 129.5 | 144.1 | -14.6 ms（-10.2%） |
| Request → first audio | 1056.2 | 1313.1 | -256.9 ms（-19.6%） |

原记录将首音频延迟（TTFA）的差距也主要归因于首文本 token 之前、前处理到 Encoder 之间的跨进程数据传输。

### Thinker + Talker：PAI-vLLM 修复跨进程拷贝后

| 区间 | SGLang-Omni | PAI-vLLM 更新后 | SGLang 变化 |
| --- | ---: | ---: | --- |
| HF Processor | 103.5 | 189.5 | -86.0 ms |
| Request → first text token | 926.7 | 871.4 | +55.3 ms（+6.3%） |
| First text → Talker prefill start | 17.8 | 21.1\* | -3.3 ms |
| Talker prefill start → first codec | 36.1 | 63.4 | -27.3 ms |
| First codec → fourth codec | 38.3 | 34.6 | +3.7 ms |
| Fourth codec → first audio | 37.3 | 23.6 | +13.7 ms |
| First text → first audio | 129.5 | 142.8 | -13.3 ms（-9.3%） |
| Request → first audio | 1056.2 | 1014.2 | +42.0 ms（+4.1%） |

更新后的比较中，SGLang-Omni 的 TTFA 为 1056.2 ms，PAI-vLLM 为 1014.2 ms，即 SGLang-Omni 高 42.0 ms（约 4.1%）。首文本之后到首音频的阶段，SGLang-Omni 仍低 13.3 ms。不能用更新前的约 19.6% TTFA 优势概括更新后的结果。

\* 原文对 `20.9` 和 `21.1` 标有星号，但未给出对应注释，此处保留标记。
