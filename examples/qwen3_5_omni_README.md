# Qwen3.5-Omni Bring-Up

这份说明记录当前 sglang-omni 里的 Qwen3.5-Omni 启动路径。模型权重
还没有落地时，可以先用这里的命令检查参数、placement 和 benchmark 条件。

## Model Layout

默认入口接受 root checkpoint：

```bash
export MODEL_DIR=/myapp/models/Qwen3.5-Omni
```

如果 checkpoint 是拆分目录，当前实现会自动把 thinker、image encoder、
audio encoder 切到 `$MODEL_DIR/thinker`，并把 talker 切到
`$MODEL_DIR/talker_lm` 或 `$MODEL_DIR/talker`，前提是对应子目录下有
`config.json`。自动探测优先 `talker_lm`，以对齐 vLLM perf_v2 server/RTC
脚本；没有 `talker_lm` 时再尝试 `talker`。talker 和 code2wav 也支持显式指定：

```text
$MODEL_DIR/
  config.json
  thinker/config.json
  talker_lm/config.json
  # older/smaller split checkpoints may use:
  # talker/config.json
  qwen3_5_omni_codec_decode_online_0306/config.yaml
  # older reference checkpoints may use:
  # qwen3_5_omni_codec_decode_online_0226/config.yaml
```

`code2wav` 会优先自动查找 `qwen3_5_omni_codec_decode_online_0306`，
也兼容 `qwen3_5_omni_codec_decode_online_0226`，然后继续尝试
`code2wav`、`codec_decoder`、`dac`、`codec` 这些目录名。
codec EOS 会先从 root config 读取，再 fallback 到 `talker_lm/config.json`
或 `talker/config.json`。
`--talker-model-path` / `--talker-path` 和 `--code2wav-model-path` /
`--code2wav-path` 在 example launcher 与通用 `serve` 入口里都可用。
vLLM 启动命令里的 `--thinker-model $MODEL_DIR/thinker` 也可直接传给通用
`serve`：如果同时有 root `--model`/`--model-path`，它会作为兼容 no-op；
如果只传 `--thinker-model`，则会把该路径当作启动 model path。
如果沿用 vLLM perf_v2 配置里的 `code2wav_model_folder`，也可以传
`--code2wav-model-folder qwen3_5_omni_codec_decode_online_0306`；它会按
`$MODEL_DIR/<folder>` 解析，显式 `--code2wav-model-path` 优先。
vLLM `qwen_omni_v35.py` 默认会传 `--code2wav-model code2wav`；Qwen3.5
speech launcher 会把这种单个相对目录名按 `$MODEL_DIR/code2wav` 解析，
而不是当成依赖当前工作目录的相对 checkpoint 路径。
code2wav 的 `--code2wav-enable-torch-compile`、
`--no-code2wav-torch-compile`、`--code2wav-enable-dynamic-chunk` 和
`--no-code2wav-dynamic-chunk` 这些 example launcher 风格开关，也可以在
通用 `serve` 入口里使用。Qwen3.5 默认对齐 vLLM perf_v2，开启
code2wav torch.compile；排查启动或编译预热问题时可以显式传
`--no-code2wav-torch-compile` 关闭。
vLLM 的 `--enable-torch-compile` / `--no-enable-torch-compile` 也会按
code2wav torch.compile alias 处理。
Qwen3.5 speech launcher 的服务级默认音色对齐 vLLM perf_v2，
默认为 `--voice-type f245`。请求里的 `audio.voice`、顶层 `voice_type` 或
`parameters.voice_type` 仍会覆盖这个默认值。如果需要服务级打开文本规范化，
可以传 `--enable-tn`；需要显式关闭时传 `--disable-tn`。
通用 `sglang_omni.cli serve` 在 Qwen3.5 speech/colocated 配置下也会默认
使用 `voice_type=f245`，并接受同样的 `--voice-type`、`--enable-tn` 和
`--disable-tn` 覆盖；text-only 配置不会注入 talker 默认值。
speech launcher 也对齐 vLLM server 的默认生成参数：
`--max-tokens 2048`、`--seed 0`、`temperature=1e-6`、`top_k=1`、`top_p=0.8`。
单个请求里的 `max_tokens`、`seed`、`temperature` 等字段仍然优先。
通用 `sglang_omni.cli serve` 在 Qwen3.5 speech/colocated 配置下也会注入
同一组服务级生成默认值；text-only 和其他模型仍保持各自默认。需要覆盖时
可以显式传 `--max-tokens`、`--seed`、`--temperature`、`--top-k`、
`--top-p`、`--repetition-penalty`、`--frequency-penalty` 或
`--presence-penalty`。
vLLM perf_v2 里的 `dtype: bfloat16` 可以直接迁移成 `--dtype bfloat16`；
如果要分阶段调试，也可以用 `--thinker-dtype` / `--talker-dtype` 覆盖。
`mamba_cache_dtype: float32` 可以用 `--mamba-cache-dtype float32` 传入；
它映射到 SGLang core 的 `mamba_ssm_dtype`。
`mamba_cache_mode` 是 vLLM 的混合 Mamba cache 策略参数；当前 SGLang
ServerArgs 没有等价入口，因此 `--mamba-cache-mode none` 会被当作显式
no-op 接受，而 `light/all` 会提前报错。vLLM perf_v2 里带
`mamba_cache_mode=light` 的 disaggregated prefill/decode profile，需要后续
单独接 SGLang KV transfer / hybrid cache 路径后再迁移。
同理，`kv_transfer_config` 和 `enable_disaggregated_prefilling` 不是普通
吞吐调优项，而是 vLLM KV connector 的 producer/consumer 部署语义。当前
`--kv-transfer-config '{}'` 与 `--enable-disaggregated-prefilling 0` 可作为
显式 no-op；非空 `kv_transfer_config` 或 truthy
`enable_disaggregated_prefilling` 会提前报错。等后续实现 SGLang 的
prefill/decode 分离与 KV transfer 映射后，再迁移
`400b_nvfp4_prefill/decode` 这类 profile。
vLLM profile 里的 `distributed_executor_backend=mp` 会被消费为显式 no-op；
`tensor_parallel_size` 在通用 `serve` 入口里等价于 `--thinker-tp-size`，
vLLM RPC 命令里的 `--thinker-tensor-parallel-size` 也是同一个参数的 alias。
多卡时仍需要配合 `--thinker-visible-devices` 指定实际 GPU。两个 example
launcher 只安全接受 `tensor_parallel_size=1`，多卡 TP 建议走通用入口。
`kv_cache_dtype=auto` 可作为 no-op；`tq4/fp8` 等量化 KV cache dtype 会提前
报错，因为当前 SGLang Qwen3.5 路径还没有等价 KV cache 存储映射。
`enable_expert_parallel=true` 也会提前报错，后续需要单独映射到 SGLang MoE
并行策略后才能用于真实性能对比。`max_mm_len` 会映射到 preprocessing
阶段的 `max_seq_len` guard，用来限制多模态输入预算；它必须小于等于当前
有效的 thinker context length，且不会降低 thinker 后端自身的上下文上限。
当 `override_video_max_pixels=true` 时，preprocessing 会按
`max_mm_len * IMAGE_FACTOR^2` 派生视频 `total_pixels` 预算，并允许该预算
覆盖默认 `VIDEO_MAX_PIXELS` 上限；显式 `--video-max-pixels` /
`--video-total-pixels` 仍可用于更细的 SGLang 侧配置。
thinker-only 与 speech launcher 都支持这些 thinker 侧 alias。
`max_model_len` 对应这里的 thinker 上下文上限；默认对齐 vLLM
`qwen_omni_v35.py` 在线 speech profile 的 `192000`。如果要跑 256k
thinker-only eval 或更长上下文，可以用 `--max-model-len 262144` 或更明确的
`--thinker-max-seq-len 262144` 显式覆盖。
vLLM perf 配置里的 `max_seq_len_to_capture` 可以用
`--max-seq-len-to-capture 262144` 原样带上；当前 SGLang 路径会消费并校验
这个参数，但不需要单独的 capture length，因此它是显式 no-op。
`max_num_batched_tokens` 可以用 `--max-num-batched-tokens 32768` 传入；
它会设置 SGLang AR stage 的 `max_prefill_tokens`，并在 chunked prefill
未关闭时同步作为 `chunked_prefill_size`。
`block_size` 对应 SGLang KV cache `page_size`，可以用
`--block-size 16` 或 `--page-size 16`。
vLLM perf 配置里的 `compilation_config` 可以用
`--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","use_inductor":false}'`
传入。`FULL_DECODE_ONLY` 会按 SGLang 默认 CUDA graph 行为处理；如果显式传
`{"cudagraph_mode":"none"}` 会映射成 `disable_cuda_graph=True`。当前不支持
vLLM 的 `use_inductor=true` 或 fuse pass 配置，启动时会提前报错。
`send_chunk_size` 对应 code2wav 流式音频 chunk，可以用
`--send-chunk-size 8` 或 `--code2wav-stream-chunk-size 8`。
vLLM server 的 `--sample-rate 24000` 会作为 `--code2wav-sample-rate`
alias 处理，控制支持该参数的 code2wav stage 输出采样率。
另外兼容 vLLM perf_v2 的 `--enable-torch-compile-first-chunk`、
`--odeint-method {euler,rk4}`、`--odeint-method-relaxed` 和
`--batched-chunk`、`--code2wav-frequency {50hz,25hz}`，以及
`--code2wav-dit-quantization fp8`。
这些参数也有 `--code2wav-...` 命名空间版本；
当前内置 Next DAC 没有 DIT/ODE hook 时会安全忽略，后续接入暴露
`enable_torch_compile(first_chunk)`、`code2wav_dit_model.cfm_model`
或 `chunk_size/bs_mel`/`dit_quant` 的真实在线 decoder 后会自动生效。

## Preflight

真实权重拷到 ECS 后，先跑本地 preflight。它只检查 config 与文件布局，
不会加载模型权重。除了 thinker/talker/code2wav 文件，它也会提示
processor、tokenizer 和 chat template 资产是否缺失；这些是
`AutoProcessor.from_pretrained(..., local_files_only=True)` 启动时会用到的
文件：

```bash
python scripts/qwen35_omni_preflight.py \
  --model-path "$MODEL_DIR"
```

拆分 code2wav 目录时：

```bash
python scripts/qwen35_omni_preflight.py \
  --model-path "$MODEL_DIR" \
  --code2wav-model-path "$MODEL_DIR/qwen3_5_omni_codec_decode_online_0306"
```

如果当前只做 thinker-only bring-up，可以跳过 talker/code2wav 检查：

```bash
python scripts/qwen35_omni_preflight.py \
  --model-path "$MODEL_DIR" \
  --text-only
```

两个 Qwen3.5 launcher 也支持在启动前直接加 `--preflight`。这样会先做本地
checkpoint 布局检查，通过后才继续拉起多进程 server：

```bash
python examples/run_qwen3_5_omni_speech_server.py \
  --model-path "$MODEL_DIR" \
  --code2wav-model-path "$MODEL_DIR/code2wav" \
  --preflight \
  --port 8008
```

如果要测 voice clone，可以提前检查预抽好的 xvector 目录。这个检查只验证
`feat.pkl`/`info.json` 是否存在、`info.json` 是否能解析；不会在 preflight
阶段 `pickle.load(feat.pkl)`，真正的 prompt code 兼容性仍在请求构建阶段解析：

```bash
python scripts/qwen35_omni_preflight.py \
  --model-path "$MODEL_DIR" \
  --xvector-info-path /path/to/xvector-info
```

同样的参数也可以挂在 speech launcher 或通用 `serve --preflight` 后面；
`--xvector-info-path`、`--voice-clone-info-path`、`--voice-clone-path`
是等价别名，且可以传多次：

```bash
python examples/run_qwen3_5_omni_speech_server.py \
  --model-path "$MODEL_DIR" \
  --preflight \
  --voice-clone-info-path /path/to/ref-a \
  --voice-clone-info-path /path/to/ref-b \
  --port 8008
```

如果确认 `feat.pkl` 是可信本地资产，也可以显式加
`--validate-xvector-pickle`，提前检查它是否包含 `prompt_code`、`ref_code`
或 `prompt_speaker_codes` 等兼容 code key。
当前 SGLang Qwen3.5 路径复用 prompt speaker prefix 支持这类 prompt
codec code；vLLM 的 xvector-only zero-shot 分支还没有独立实现，因此只有
`xvector`、没有 prompt codec code 的资产会在严格预检中报错。

还没有模型、但已经从 vLLM perf_v2 拿到 `engine_args` 时，可以先做 profile
兼容性 preflight。它不会加载模型，只会把参数分成 `mapped`、`noop`、
`warning`、`unsupported` 和 `unknown`，并给出能迁移到 SGLang 的 CLI 参数。
`warning` 表示配置能继续迁移但语义不是完全等价；`unsupported` 或
`unknown` 都会让 preflight 失败。通过时，输出末尾的
`[mapped-cli]` 行就是已经映射好的参数片段，可以拼到 `serve` 或 example
launcher 后面继续调试：

```bash
python scripts/qwen35_omni_preflight.py \
  --vllm-profile /path/to/qwen35_perf_profile.json
```

如果只想粘一段 JSON，也可以直接传：

```bash
python scripts/qwen35_omni_preflight.py \
  --vllm-engine-args-json '{"dtype":"bfloat16","max_model_len":192000}'
```

带 Qwen3.5 MTP `speculative_config` 的 profile 也可以先按当前 base thinker
AR 路径做兼容性检查，不需要修改原始 vLLM profile 文件：

```bash
python scripts/qwen35_omni_preflight.py \
  --vllm-profile /path/to/qwen3.5-omni/23b_fp8_mtp_vc/h20.config \
  --disable-mtp
```

两个 example launcher 也可以直接读取 vLLM perf_v2 profile。profile 会先做
同一套 preflight 检查；通过后，`mapped` 参数会作为启动默认值注入，用户在
命令行里显式传的参数优先级更高。speech launcher 会保留 talker/code2wav
相关字段，thinker-only launcher 会过滤这些 speech-only 字段，只使用 thinker
和 preprocessing 相关配置：

```bash
python examples/run_qwen3_5_omni_speech_server.py \
  --vllm-profile /myapp/vllm/util/vllmgen/configs/qwen3.5-omni/23b_fp8/h20.config \
  --model-path "$MODEL_DIR" \
  --port 8008
```

```bash
python examples/run_qwen3_5_omni_server.py \
  --vllm-profile /myapp/vllm/util/vllmgen/configs/qwen3.5-omni/23b_fp8/h20.config \
  --model-path "$MODEL_DIR" \
  --port 8008
```

通用 `sglang_omni.cli serve` 入口也可以直接读取 profile，适合需要
`--thinker-tp-size`、`--thinker-visible-devices` 或 colocated YAML 等更细
配置的场景。profile 注入的是默认值；命令行里显式传的 `--model-path`、
`--talker-gpu`、`--code2wav-stream-chunk-size` 等参数会覆盖 profile。
vLLM 控制脚本里的 `--serve-port` 也可直接作为 `--port` alias：

```bash
python -m sglang_omni.cli serve \
  --vllm-profile /myapp/vllm/util/vllmgen/configs/qwen3.5-omni/23b_fp8/h20.config \
  --model-path "$MODEL_DIR" \
  --thinker-model "$MODEL_DIR/thinker" \
  --serve-port 8008 \
  --talker-gpu 1 \
  --code2wav-gpu 1 \
  --voice-type f245 \
  --temperature 0.000001 \
  --top-k 1 \
  --top-p 0.8 \
  --host 0.0.0.0 \
```

### 离线推理与精度对齐脚本

`scripts/qwen35_omni_alignment.py` 可以用同一段音频顺序跑 vLLM baseline
和 SGLang-Omni speech server，保存两侧文本、WAV、ASR transcript 和
`alignment_report.md`。例如宿主机 `/home/gangouyu` bind 到容器 `/myapp`
时，可以直接用 vLLM 镜像和已有 SGLang 容器编排：

```bash
VLLM_IMAGE="tongyi-duanwu-registry-vpc.cn-beijing.cr.aliyuncs.com/dashscope/"
VLLM_IMAGE="${VLLM_IMAGE}dashllm:cuda129_cp312_test_vl_13589"

python3 scripts/qwen35_omni_alignment.py \
  --backend compare \
  --model-path "$MODEL_DIR" \
  --prompt '请用两到三句话详细描述这段音频的内容。' \
  --vllm-root '' \
  --vllm-docker-image "$VLLM_IMAGE" \
  --vllm-docker-mount /home/gangouyu:/myapp \
  --disable-vllm-mtp \
  --launch-sglang \
  --sglang-python python \
  --sglang-container b5f665f3d883 \
  --sglang-port 8101 \
  --asr-container b5f665f3d883 \
  --asr-model base \
  --asr-language zh \
  --voice-type tina \
  --max-tokens 256 \
  --no-code2wav-torch-compile \
  --output-dir results/qwen35_omni_alignment
```

如果 SGLang 服务已经启动，可以不传 `--launch-sglang`，改用
`--sglang-base-url http://127.0.0.1:8101`。默认测试音频来自 vLLM
Qwen3.5-Omni 单测里的公开样本；也可以用 `--audio-path /path/to/input.wav`
替换成本地音频。ASR 默认走 `openai-whisper`，已有内部 ASR 命令时可传
`--asr-command 'your_asr --audio {audio}'`。如果 vLLM 不跑 Docker，也可以
用 `--vllm-python /path/to/vllm-env/bin/python` 指向对应 Python 环境。
当前 vLLM 镜像里建议用 `--vllm-root ''` 走镜像内置的已编译 vLLM 包；
`--disable-vllm-mtp` 用于对齐当前 SGLang-Omni 已实现的 base thinker AR
主链路，后续接入 SGLang speculative/MTP 后再打开 MTP 做性能路径对齐。

vLLM server/offline demo 脚本里的 `host`、`port` / `serve_port`、
`text_only` 和 `omni_video_fps` 会分别映射到通用 `serve` 的
`--host`、`--port`、`--text-only` 和 `--video-fps`。
`prompt`、`num_prompts`、`do_wave`、`id_start_index`、`conversation`、
`token_prompt`、`use_torchvision`、`do_warmup`、`multi_waveforms`、
`translation_prefix`、`asr_language`、`prompt_file` 和 RTC chunk/test-file
控制项会在 profile preflight 中作为兼容 no-op 接受：这些字段只控制 demo
输入、离线请求数量、音频输出开关、请求 id、warmup 或输出保存，不应该变成
SGLang server 的全局启动参数。`max_tokens`、`seed` 和
`talker_quantization` 会映射到通用 `serve` 的服务级默认生成参数和 talker
stage quantization；单个 OpenAI 请求体里的同名采样参数仍然优先。
`legacy_omni_video=true`、`vl_fastv=true` 和
`enable_rtc_simulate=true` 会显示为 `unsupported`，因为这些会改变 vLLM 的
视频/RTC 执行路径，当前还没有等价的 SGLang serve/profile 映射。
`model_to_run=talker/code2wav`、`v6d_send_socket`、`v6d_recv_socket`、
`remote_endpoint`、`pd_prefill_host`、`pd_prefill_rpc_port`、
`pd_prefill_endpoint`、`v6d_prefill_recv_socket`，以及大于 1 的
`thinker_data_parallel_size` / `code2wav_data_parallelism` 会显示为
`unsupported`。这些属于 vLLM RPC 单组件启动、PD prefill 或多副本部署语义；
当前 SGLang-Omni Qwen3.5 路径还是一个统一 pipeline serve 入口，后续接入
SGLang KV transfer / stage replica placement 后再迁移。

带 `mamba_cache_mode=light`、非空 `kv_transfer_config`、
`enable_disaggregated_prefilling=true` 或 `enable_expert_parallel=true` 的
profile 会显示为 `unsupported`。这些参数会改变 KV transfer 或 MoE worker
placement 形态，当前实现不会静默降级成普通 SGLang 启动。非空
`speculative_config` 默认也会显示为 `unsupported`；唯一例外是同时设置
`disable_mtp=true` 时，Qwen3.5 MTP speculative config 会被当作显式 no-op，
用于先跑 base thinker AR 路径。
`trust_remote_code`、`disable_log_stats`、`enable_prompt_embeds` 和
`mm_processor_cache_type` 会作为 vLLM profile 兼容字段接受；当前 SGLang
Qwen3.5 路径分别由本地 wrapper/config shim、benchmark 外部统计、stage
内部 prompt-embedding 约定和普通 preprocessing 执行来承接，因此这些字段
不会生成额外 CLI 参数。

vLLM perf_v2 的部分 Qwen3.5-Omni 配置会启用
`speculative_config.method=qwen3_omni_next_thinker_mtp`，并在 checkpoint 里带
`thinker.mtp.*` 权重。当前 sglang-omni 这版先实现 base thinker AR +
talker/code2wav 主链路，MTP/draft 权重会被 thinker loader 跳过；preflight
检测到 `mtp_num_hidden_layers` 或 `thinker.mtp.*` index 时会给 warning，便于
后续拿真实模型做性能对齐时继续接 SGLang speculative 路径。
为了方便迁移 vLLM 启动命令，`--disable-mtp` 在两个 example launcher 和
通用 `serve` 入口里都会被接受；它在当前实现中是显式 no-op，因为 MTP 本来
就没有开启。`--speculative-config '{}'` 也会作为显式 no-op 接受。如果 vLLM
profile 里带
`{"method":"qwen3_omni_next_thinker_mtp","num_speculative_tokens":4}`，同时
传 `--disable-mtp` 时会忽略该 MTP 配置并继续启动 base thinker；不传
`--disable-mtp` 时仍会提前报错，避免把 MTP profile 当成已启用的性能路径。

## Thinker-Only

先跑 thinker-only，能最快验证 Qwen3OmniNext 架构注册、预处理和
image/audio encoder 加载：

```bash
python examples/run_qwen3_5_omni_server.py \
  --model-path "$MODEL_DIR" \
  --gpu-thinker 0 \
  --gpu-image-encoder 0 \
  --gpu-audio-encoder 0 \
  --gpu-memory-utilization 0.6 \
  --video-fps 2 \
  --video-max-frames 128 \
  --video-min-frames 4 \
  --video-max-pixels 401408 \
  --image-max-pixels 401408 \
  --port 8008
```

## Speech Disaggregated

分离部署建议先用 GPU0 跑 thinker 和 encoder，GPU1 跑 talker/code2wav。
如果不显式传 `--gpu-code2wav`，Qwen3.5 launcher 默认也会让 code2wav
跟随 talker，避免把 codec 解码压到 thinker GPU：
Qwen3.5 默认 `max_running_requests=32`，对齐 vLLM perf_v2 H20 profile 的
`max_num_seqs=32`；压测时可以用 `--max-running-requests` 或 vLLM alias
`--max-num-seqs` 全局调整，
或用 `--thinker-max-running-requests` / `--talker-max-running-requests`
分别覆盖 thinker/talker。
从 vLLM 启动命令迁移时，也可以继续传 `--enable-prefix-caching`、
`--enable-chunked-prefill` / `--no-enable-chunked-prefill`、`--enforce-eager`、
`--thinker-enforce-eager`、`--talker-enforce-eager` 和
`--thinker-quantization {none,fp8,nvfp4}`、
`--talker-quantization {none,fp8,nvfp4}`。这些参数会转成 SGLang
`server_args_overrides`：prefix caching 对应 radix cache，eager 对应关闭
CUDA graph，chunked prefill 使用 Qwen3.5 H20 profile 的 32768 chunk size。
vLLM 的 `--gpu-memory-utilization`、
`--thinker-gpu-memory-utilization`、`--talker-gpu-memory-utilization`
也可继续使用，会映射到 SGLang 的 `mem_fraction_static`；
同名下划线版本在 example launcher 和通用 `serve` 入口都兼容。
vLLM perf 配置里的 `thinker_visible_devices`、
`talker_visible_devices`、`code2wav_visible_devices` 也可以按 CLI 形式传入：
`--thinker-visible-devices '[0,1]'` 会映射成 thinker TP placement，
`--talker-visible-devices '[2]'` 和 `--code2wav-visible-devices '[2]'`
会映射成对应单卡 stage GPU。
Qwen3.5 默认 `limit_mm_per_prompt={"audio":960,"image":960,"video":960}`，
对齐 vLLM `qwen_omni_v35.py` 在线 speech profile 的服务级多模态数量 guard。
如果要复现 256k thinker eval profile，可以显式传
`{"audio":2048,"image":2048,"video":2048}`。需要更小的压测上限时，
可以传 vLLM JSON 形式 `--limit-mm-per-prompt '{"image":2,"video":1}'`，
也可以用 `--limit-mm-per-prompt-image` /
`--limit-mm-per-prompt-video` / `--limit-mm-per-prompt-audio` 分别覆盖。

```bash
python examples/run_qwen3_5_omni_speech_server.py \
  --model-path "$MODEL_DIR" \
  --model-name qwen3.5-omni \
  --gpu-thinker 0 \
  --gpu-talker 1 \
  --gpu-code2wav 1 \
  --gpu-image-encoder 0 \
  --gpu-audio-encoder 0 \
  --gpu-memory-utilization 0.6 \
  --talker-gpu-memory-utilization 0.8 \
  --video-fps 2 \
  --video-max-frames 128 \
  --video-min-frames 4 \
  --video-max-pixels 401408 \
  --image-max-pixels 401408 \
  --limit-mm-per-prompt '{"image":960,"video":960,"audio":960}' \
  --max-running-requests 32 \
  --port 8008
```

拆分 checkpoint 可以加：

```bash
python examples/run_qwen3_5_omni_speech_server.py \
  --model-path "$MODEL_DIR" \
  --talker-model-path "$MODEL_DIR/talker_lm" \
  --code2wav-model-folder qwen3_5_omni_codec_decode_online_0306 \
  --gpu-thinker 0 \
  --gpu-talker 1 \
  --gpu-code2wav 1 \
  --port 8008
```

## Speech Colocated

colocated 需要给每个 stage 配 `total_gpu_memory_fraction`，因此推荐走
YAML 配置。当前 H20 初始 profile 在：
`examples/configs/qwen3_5_omni_colocated_h20.yaml`。

```bash
python -m sglang_omni.cli serve \
  --config examples/configs/qwen3_5_omni_colocated_h20.yaml \
  --colocate \
  --preflight \
  --model-path "$MODEL_DIR" \
  --model-name qwen3.5-omni \
  --video-fps 2 \
  --video-max-frames 128 \
  --video-min-frames 4 \
  --video-max-pixels 401408 \
  --image-max-pixels 401408 \
  --max-running-requests 32 \
  --host 0.0.0.0 \
  --port 8008
```

真实模型落地后，需要根据显存峰值重新校准 YAML 里的 stage memory
fraction；这份 profile 只是 H20 上的保守起点。

`--preflight` 会在 YAML merge 和 CLI 覆盖之后、真正拉起多进程 server 之前
检查 Qwen3.5 checkpoint 布局；如果 `--code2wav-model-path` 没有显式传入，
它会读取最终 `code2wav` stage 的 `factory_args.code2wav_model_path`。

这些 `--video-*` / `--image-*` 参数是服务级默认 preprocessing 条件；benchmark
请求里显式传入的 `video_fps/video_min_frames/video_max_frames/video_max_pixels`
或 OpenAI image content part 上的 `min_pixels/max_pixels` 仍然可以逐请求覆盖。
YAML 里的 `preprocessing.runtime.image_max_pixels`、`video_fps` 等字段也会走
同一条 runtime_arg_map 注入路径。

如果 colocated 使用拆分 checkpoint，也可以在同一个 CLI 里覆盖 code2wav
参数：

```bash
python -m sglang_omni.cli serve \
  --config examples/configs/qwen3_5_omni_colocated_h20.yaml \
  --colocate \
  --preflight \
  --model-path "$MODEL_DIR" \
  --model-name qwen3.5-omni \
  --talker-model-path "$MODEL_DIR/talker_lm" \
  --code2wav-model-path \
    "$MODEL_DIR/qwen3_5_omni_codec_decode_online_0306" \
  --code2wav-enable-torch-compile \
  --code2wav-enable-dynamic-chunk \
  --code2wav-dynamic-chunk-sizes 2,4,8 \
  --code2wav-dynamic-chunk-steps "8 4 1" \
  --port 8008
```

## Video-AMME Benchmark

和 Qwen3-Omni 对比时，尽量保持同一个数据集、并发、视频采样和输出模式。
benchmark 请求会把 video/audio 放进 OpenAI `messages[].content` parts；
音频输出同时带 `modalities: ["text", "audio"]` 和 vLLM-style
`enable_audio_output: true` / `do_wave: true`，因此同一套脚本可以更直接地打
sglang-omni 或 vLLM perf_v2 的 Qwen3.5 server。
如果只测性能、不需要 WER，保留 `--skip-wer`：

```bash
python -m benchmarks.eval.benchmark_omni_videoamme \
  --model qwen3.5-omni \
  --port 8008 \
  --repo-id zhaochenyang20/Video_AMME_ci \
  --max-samples 50 \
  --max-concurrency 8 \
  --video-fps 2 \
  --video-max-frames 128 \
  --video-min-frames 4 \
  --video-max-pixels 401408 \
  --enable-audio \
  --skip-wer \
  --output-dir results/videoamme_qwen35_talker_c8
```

## Quick Smoke Request

OpenAI-compatible audio output 请求应显式带 `modalities`，这样同一个服务
可以区分 text-only 和 text+audio：

也兼容 vLLM server/offline 风格的
`enable_audio_output`/`return_audio`/`do_wave` 布尔开关；
如果同时传入，`modalities` 优先。
OpenAI `audio.voice` 会映射到 Qwen3.5 talker 的 `voice_type`，用于选择
音色；`audio.language`/`audio.language_id`、`audio.voice_style` 和
`audio.instruction` 也会在没有显式 params 覆盖时传给 talker。`audio.format`
只影响响应编码/封装语义，不参与 talker prefill。
voice clone 可以通过 `audio.xvector_info`、`audio.voice_clone_info` 或
`audio.voice_clone.path` 传入，最终复用 talker 的 prompt speaker prefix 路径。
目录模式下会读取 `feat.pkl` + `info.json`，并兼容 `prompt_code`、
`prompt_speaker_codes`、`prompt_codes` 等常见 code key。
OpenAI `tools` 会透传给 Qwen3.5 chat template，用于复现 vLLM
`function-call`/`mcp-call` prompt 渲染；服务端只渲染工具 schema，不执行工具。
旧版 OpenAI `functions` 会先包装成 `{"type": "function", "function": ...}`。
多轮工具调用消息里的 `name`、`tool_calls`、`tool_call_id` 和旧版 assistant
`function_call` 也会保留，避免 assistant/tool/function 历史在 chat template
渲染时丢上下文。
OpenAI `metadata` 会作为请求追踪/压测标签保留到 `GenerateRequest.metadata`；
服务端派生出的 `audio_config`、媒体输入和工具 schema 等内部字段会覆盖同名
metadata key，避免外部标签误改预处理语义。
OpenAI 标准 `user` 以及 sglang-omni 扩展 `request_id` 也会进入 metadata，
便于把 HTTP 请求、coordinator 阶段日志和压测样本串起来。
OpenAI content part 里的 `type: "input_text"` 会和普通 `type: "text"` 一样
归一成聊天文本，兼容 Responses 风格请求构造器。
OpenAI video content part 上的 `fps`/`min_frames`/`max_frames`/`min_pixels`/
`max_pixels`/`total_pixels`/`use_audio_in_video` 会提升为请求级 video 预处理
参数；如果请求顶层已经传入 `video_fps` 等同类字段，则顶层字段优先。
OpenAI image content part 上的 `min_pixels`/`max_pixels` 会提升为
`image_min_pixels`/`image_max_pixels` 并传给 HF processor 的 `images_kwargs`。
请求顶层或 `parameters` 里的 `image_url`/`image_urls`、`input_image`/
`input_images`、`video_url`/`video_urls`、`input_video`/`input_videos`
也会分别归一成 `images`/`videos`，方便复用 vLLM perf_v2 风格请求构造器。
这些字段既可以是普通路径/URL，也可以是 OpenAI-style `{"url": "..."}`
或 `{"data": "...", "format": "mp4"}`；后者会转成已有 resource connector
能加载的 `data:<media>/<format>;base64,...`。
OpenAI body 顶层的 `use_audio_in_video`、`max_frames`、`fps` 等字段即使
进入 `OmniRequest.params`，也会先提升到 preprocessor 的输入侧。
多个 video content part 上的 `use_audio_in_video` 会按视频顺序归一成 bool
list，用于只给需要的那几个视频抽音轨。
`dependent_audio`/`video_dependent_audio` 作为 vLLM-compatible 开启信号兼容；
sglang-omni 的 CPU preprocessor 不会把它传给 HF video processor。OpenAI
content part 里的 audio/video 会按 prompt 顺序进入 audio 特征列表，避免
视频音轨和独立音频错位。普通顶层 `videos`/`audios` 混用时，Qwen3.5 也会
按模板里的 video 占位符优先顺序，把视频音轨排在独立音频前面。

也可以接近 vLLM perf_v2 的离线输入形态：`multi_modal_data.image/video/audio`
会提升到 sglang-omni 顶层 `images/videos/audios`；`mm_processor_kwargs`
里的 `fps`、`max_frames`、`use_audio_in_video`、`dependent_audio`、
`video_metadata` 以及嵌套 `videos_kwargs/images_kwargs/audio_kwargs`
会按白名单提升为预处理参数。`TextPrompt.prompt` 会作为已经渲染好的 raw
prompt 使用，不会再次套 chat template；`TokensPrompt.prompt_token_ids`
会保留调用侧 token ids，只让 HF processor 负责多模态特征抽取。
vLLM perf_v2 example 里的请求级 `enable_tn` 也会保留到
`OmniRequest.params`，当前先作为兼容开关透传。
非流式 audio 响应会同时保留 OpenAI-style `choices[0].message.audio`，并额外
返回 vLLM-style 顶层 `audio: {"data": "...", "format": "wav"}`，方便同一套
压测脚本对比 vLLM 与 sglang-omni。

```bash
curl http://localhost:8008/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.5-omni",
    "modalities": ["text", "audio"],
    "audio": {
      "voice": "Cherry",
      "language": "zh-CN",
      "format": "wav"
    },
    "messages": [
      {"role": "user", "content": "用一句话介绍你自己"}
    ],
    "max_tokens": 128
  }'
```

## Streaming TTFT

首音频指标可以复用 `benchmark_omni_streaming_ttft.py`。脚本会同时传
`modalities: ["text", "audio"]`、vLLM-style
`enable_audio_output: true` 和 `do_wave: true`，
并兼容 sglang-omni `delta.audio` 与 vLLM perf_v2 顶层
`chat.completion.audio` SSE 事件。Qwen3.5 speech launcher 已经有服务级默认
音色；一般可以不传 `--voice`，需要指定时再传：

```bash
python benchmarks/eval/benchmark_omni_streaming_ttft.py \
  --base-url http://localhost:8008 \
  --model qwen3.5-omni \
  --label qwen35_stream_ttf_audio \
  --max-tokens 256 \
  --warmup 2 \
  --repeats 10
```

## Validation

本地 mirror 校验：

```bash
bash scripts/validate_qwen35_remote_copy.sh
```

同步到 ECS 并在容器内跑回归：

```bash
RUN_REMOTE_VALIDATE=1 bash scripts/sync_qwen35_to_ecs.sh
```
