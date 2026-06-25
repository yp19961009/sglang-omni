# Qwen3.5-Omni 外部复现 Handoff Runbook

状态：2026-06-21 evidence-ready 交付稿；更新后的目标不再等待 6.21 晚上。
工作目录：`/home/gangouyu/sglang-omni`。
用途：给合作高校或外部 reviewer 一个最短、可执行、可验收的复现入口。

## 路径映射约定

本文命令默认仓库根目录为 `/home/gangouyu/sglang-omni`，这是当前证据包的生成路径。
接收方如果把仓库或解包目录放在别的位置，先在 shell 中改下面变量，再复制命令：

```bash
export HOST_REPO="${HOST_REPO:-/home/gangouyu/sglang-omni}"
export SMOKE_DIR="${SMOKE_DIR:-/tmp/qwen35_omni_receiver_smoke_final}"
export EXTRACTED_BUNDLE="${EXTRACTED_BUNDLE:-${SMOKE_DIR}/qwen35_omni_share_bundle_20260621}"
```

host 侧审计、tarball 校验和报告再生成都在 `$HOST_REPO` 下执行；解包目录只用于阅读和
extracted-only 校验，不用于替换源仓库里的 benchmark artifact。
接收方如果先从 tarball/解包目录阅读，先打开路径手册：
`/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_receiver_package_path_map_zh_20260621.md`。
如果只需要先复制收包校验、full audit 和复跑入口命令，打开一页式命令卡：
`/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_receiver_command_card_zh_20260621.md`。

相关文档：

- 接收方路径手册：
  `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_receiver_package_path_map_zh_20260621.md`
- 接收方命令卡：
  `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_receiver_command_card_zh_20260621.md`
- 主报告：
  `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_stress_performance_plan_20260621.md`
- 分享包索引：
  `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_share_package_index_zh_20260621.md`
- 复现清单：
  `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_reproduction_checklist_zh_20260621.md`
- 合作方复跑验收表：
  `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_collaborator_rerun_validation_sheet_zh_20260621.md`
- 复跑差异定位矩阵：
  `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_rerun_delta_triage_zh_20260621.md`
- 指标来源索引：
  `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_metric_source_map_zh_20260621.md`
- Stage 指标字典：
  `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_stage_metric_dictionary_zh_20260621.md`
- Stage latency budget：
  `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_stage_latency_budget_zh_20260621.md`
- Stage boundary bottleneck ledger：
  `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_stage_boundary_bottleneck_ledger_zh_20260621.md`
- Stage 因果图：
  `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_stage_causal_graph_zh_20260621.md`
- Stage 因果图 JSON：
  `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/stage_causal_graph.json`
- Stage 复现实操 Drilldown：
  `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_stage_reproduction_drilldown_zh_20260621.md`
- Stage route 裁决矩阵：
  `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_stage_route_decision_matrix_zh_20260621.md`
- SGLang 优化锁定矩阵：
  `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_sglang_optimization_lock_zh_20260621.md`
- vLLM 优化锁定矩阵：
  `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_vllm_optimization_lock_zh_20260621.md`
- Final completion audit：
  `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_final_completion_audit_zh_20260621.md`
  机器证据：`/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/final_completion_audit.json`

## 0. 收包快检

如果接收方拿到的是便捷 tarball，先不要读结论，先在仓库根目录做一条
quickcheck：

1. 运行 `qwen35_omni_receiver_quickcheck.sh`，它会串起 checksum、tarball-mode
   validation、receiver-smoke validation、extracted-only validation 和 standalone validation。
2. 打开 share package index，从五分钟阅读顺序进入报告。
   如果接收方只想先复制命令，先打开
   `benchmarks/reports/qwen35_omni_receiver_command_card_zh_20260621.md`。

```bash
HOST_REPO="${HOST_REPO:-/home/gangouyu/sglang-omni}"
SMOKE_DIR="${SMOKE_DIR:-/tmp/qwen35_omni_receiver_smoke_final}"
EXTRACT_DIR="${EXTRACT_DIR:-/tmp/qwen35_omni_share_bundle_final}"
STANDALONE_DIR="${STANDALONE_DIR:-/tmp/qwen35_omni_external_standalone_bundle_validation_handoff}"
cd "$HOST_REPO"

bash benchmarks/eval/qwen35_omni_receiver_quickcheck.sh
```

快检通过时应看到：

- `sha256sum` 输出 `OK`。
- `share_package_receiver_smoke_validation.json` 中 `ready=true`。
- `receiver_smoke_ready=true`。
- tarball-mode validation 为 `17/17`。
- nested extracted-only validation 为 `13/13`。
- `share_package_external_standalone_validation.json` 中 `ready=true`，standalone checks 为 `8/8`。
- tarball 和 nested extracted-only 的资产 evidence 中
  `report_quality_offenders=[]`、`chart_quality_offenders=[]`。

其中 `report_quality_offenders=[]` 表示随包 `share_report` Markdown 没有裸 hash、
坏表格或坏展示 token；`chart_quality_offenders=[]` 表示随包 `share_charts` CSV/SVG
可解析、非空且结构可渲染。

如果 reviewer 想先用机器查询复核核心结论、stage 连接和长短文 coverage，可以在
host 仓库根目录追加运行 evidence query card smoke：

```bash
HOST_REPO="${HOST_REPO:-/home/gangouyu/sglang-omni}"
cd "$HOST_REPO"

bash benchmarks/eval/qwen35_omni_evidence_query_cards_smoke.sh \
  --root "$HOST_REPO" \
  --mode host
```

解包后也可以在 bundle 内用 portable 模式跑同一套查询：

```bash
BUNDLE_ROOT="${BUNDLE_ROOT:-${EXTRACTED_BUNDLE}}"

bash "$BUNDLE_ROOT/benchmarks/eval/qwen35_omni_evidence_query_cards_smoke.sh" \
  --root "$BUNDLE_ROOT" \
  --mode portable
```

这条 smoke 会查询 `length_regime_coverage.json`、headline/stage/runtime 证据和
share package gate；它是 reviewer 快速自证材料，不改变下游
`34` 个 return evidence hard contract。

快检失败分流：

| 失败点 | 先看证据 | 裁决 |
| --- | --- | --- |
| checksum FAIL | `.tar.gz.sha256` 和 `sha256sum -c` 输出 | 不进入报告阅读；先重传 tarball/checksum。 |
| tarball validation FAIL | `results/qwen35_report_audit_20260619/share_package_validation.json` | 不发送为最终包；修复缺失/mismatch 后重建 tarball。 |
| receiver smoke FAIL | `results/qwen35_report_audit_20260619/share_package_receiver_smoke_validation.json` | 不替换任何 benchmark 数字；先确认安全解包和 nested extracted-only gate。 |
| extracted-only validation FAIL | `results/qwen35_report_audit_20260619/share_package_validation_extracted.json` | 说明解包目录或随包文件不自洽；回到路径手册确认 `BUNDLE_ROOT` / `HOST_REPO`。 |
| standalone validation FAIL | `results/qwen35_report_audit_20260619/share_package_external_standalone_validation.json` | 说明随包 validator 不能在干净 `/tmp` 独立运行；先修包，不进入外部复现。 |
| quality offenders 非空 | validation JSON 里的 `report_quality_offenders` / `chart_quality_offenders` | 先修 Markdown/CSV/SVG 质量，再重跑 full audit 和 release seal。 |

如果只收到解包后的目录，也可以直接在解包目录运行 extracted-only 校验：

```bash
HOST_REPO="${HOST_REPO:-/home/gangouyu/sglang-omni}"
SMOKE_DIR="${SMOKE_DIR:-/tmp/qwen35_omni_receiver_smoke_final}"
EXTRACTED_BUNDLE="${EXTRACTED_BUNDLE:-${SMOKE_DIR}/qwen35_omni_share_bundle_20260621}"
cd "$EXTRACTED_BUNDLE"

python3 benchmarks/eval/validate_qwen35_omni_share_package.py \
  --root "$PWD" \
  --extracted-only \
  --strict \
  --json-output "$HOST_REPO/results/qwen35_report_audit_20260619/share_package_validation_extracted.json"
```

解包目录只用于阅读和接收方校验；需要重跑 benchmark 或更新证据时，回到
`$HOST_REPO` 仓库根目录执行后续命令。

## 1. 验收顺序

最短验收路径分四步：

1. 先在 host 上跑 full audit，确认当前证据包自洽。
2. 再看主报告、中文简报、pressure matrix、stage metric dictionary，确认结论和 stage 口径。
3. 需要重跑时，先复现 SGLang 主压测和短/长文本/语音，再复现 vLLM baseline/prebuild diagnostic。
4. 重跑后刷新 tables、claims、stage interaction、headline、acceptance、confidence、preflight、coverage、manifest，并用复跑验收表登记环境、指标和差异；如果数字偏移，先按 rerun delta triage 定位 stage/boundary，再只替换通过 gate 的数字。

如果只是 review 当前 checkpoint，不需要先重跑所有 benchmark；full audit 和 manifest
已经能证明当前报告引用的 artifact 是否齐全、数字是否自洽。

## 2. Host 侧第一条命令

```bash
HOST_REPO="${HOST_REPO:-/home/gangouyu/sglang-omni}"
cd "$HOST_REPO"

python3 -m benchmarks.eval.run_qwen35_omni_report_audit \
  --root "$HOST_REPO" \
  --summary-output results/qwen35_report_audit_20260619/audit_run_summary.json
```

当前 checkpoint 的接受门槛：

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
- repro command manifest `ready=true`，63 条命令、7 个阶段
- command reference hygiene `ready=true`，结构化 rerun command IDs 均可解析
- final readiness audit `ready=true`，49/49 checks，0 required failures
- final completion audit `ready=true`，`final_completion_audit.json` required failures 为 0；`completion_allowed_now=true` 表示可以进入最终完成裁决
- share bundle manifest `ready=true`
- share package validation `ready=true`，17/17 checks，0 required failures
- external standalone validation `ready=true`，8/8 checks，0 required failures
- package asset quality evidence `report_quality_offenders=[]`，`chart_quality_offenders=[]`
- length regime coverage `ready=true`，7 rows / 10 checks；short prompt 为 74 chars，long prompt 为 944 chars，long c=8 RTF p95 小于 1
- receiver quickcheck contract `ready=true`，15/15 checks，6-step wrapper，4 receiver JSONs，completion-gate docs 为 6，WER/ASR docs 为 3，stage dictionary crosswalk needles 为 8，0 required failures
- evidence query card smoke 可在 host/portable 模式自证核心查询、stage gate 和长短文 coverage gate
- receiver quickcheck 会同时覆盖 checksum、tarball-mode validation、receiver-smoke validation、extracted-only validation 和 standalone validation

唯一允许的现有 warning 是 host 侧可选 Whisper cache：
`/root/.cache/whisper/large-v3.pt` 不存在。它只影响 host 侧直接跑 offline
WER；如果容器内已有权重或改用 ASR router，不影响报告其他 gate。

## 3. 环境和镜像确认

复现该报告默认环境：

- GPU：8x NVIDIA H20。
- SGLang-Omni 镜像：`frankleeeee/sglang-omni:dev`。
- vLLM 镜像：
  `tongyi-duanwu-registry-vpc.cn-beijing.cr.aliyuncs.com/dashscope/dashllm:cuda129_cp312_test_vl_13589`。
- 模型路径：
  `/myapp/models/qwen3_5_omni_23b_final_multilingual_all_voice_bf16_0315`。
- Video-AMME cache：
  `/myapp/data/videoamme`。

环境证据在：
`/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/environment_snapshot.json`。

如果接收方机器不是 8x H20，可以重跑，但不要直接把吞吐、RTF 和 stage
tail 数字替换为当前报告结论；需要在报告中把硬件差异单独标注。

## 4. SGLang 复现主线

进入 SGLang container 后先启动服务：

```bash
cd /myapp/sglang-omni

NO_CODE2WAV_TORCH_COMPILE=0 \
TORCHDYNAMO_DISABLE=0 \
SGLANG_OMNI_VIDEO_PREPROCESS_CACHE_MAX_BYTES=17179869184 \
SGLANG_OMNI_VIDEO_PREPROCESS_CACHE_MAX_ENTRIES=64 \
EXTRA_ARGS="--thinker-cuda-graph on --talker-cuda-graph on --talker-torch-compile on --thinker-max-running-requests 8 --talker-max-running-requests 8" \
bash examples/launch_qwen35_omni_speech_server_container.sh
```

随后按复现清单 section 2 跑 Video-AMME ci-50 c=1/2/4/8/16，并为每个
run 采集 request profile。复现时重点看三件事：

- c=8 应是当前 recipe 的吞吐峰值。
- c=16 应显示 admission/queueing 饱和，不推荐作为默认运行点。
- `code2wav_decode` 不应成为主要 compute bottleneck。

短/长文本输入 + 语音输出按复现清单 section 3 跑 synthetic speech c=1/4/8。短文本
应是 74 chars / 12 words，长文本应是 944 chars / 139 words。长语音的关键验收是
long c=8 仍快于实时，且 tail 主要落在 talker AR，而不是 code2wav decode。

WER 按复现清单 section 4 在 serving 压测结束后离线跑，避免 ASR 和 Qwen3.5 serving
抢 GPU。

## 5. vLLM 复现主线

严格 apples-to-apples headline 使用 warmed c=4 artifact：

- `results/qwen35_vllm_videoamme_ci50_official_talker_compile_c4_20260618_202106/run.log`
- `results/qwen35_vllm_videoamme_ci50_official_talker_compile_c4_20260618_202106/benchmark_audio_50_c4_offline_compile/vllm_videoamme_report.md`

复跑 c=1/c=8 original offline 和 c=8 prebuild w4 diagnostic 时使用复现清单
section 5 的命令。对外解释时保持四条边界：

- vLLM c=4 是严格 warmed 对比点。
- vLLM 优化锁定矩阵证明该 baseline 使用 Qwen3.5-capable 镜像、compile/CUDA graph、
  prefix/chunked prefill、shared-memory transfer 和 prebuild w4 证据，不是弱 baseline。
- vLLM original c=8 主要证明 offline runner prompt build/feed admission 受限。
- vLLM c=8 `--prebuild-prompts --prebuild-workers 4` 是当前最强 offline diagnostic，
  不是 online serving parity 结论。

## 6. 重跑后刷新证据

重跑任意 SGLang 或 vLLM artifact 后，按这个顺序刷新：

```bash
HOST_REPO="${HOST_REPO:-/home/gangouyu/sglang-omni}"
cd "$HOST_REPO"

python3 -m benchmarks.eval.run_qwen35_omni_report_audit \
  --root "$HOST_REPO" \
  --summary-output results/qwen35_report_audit_20260619/audit_run_summary.json
```

如果 full audit 失败，不要把新数字写入分享稿。先看：

- `results/qwen35_report_audit_20260619/claims_verification.json`
- `results/qwen35_report_audit_20260619/stage_interaction_summary.json`
- `results/qwen35_report_audit_20260619/acceptance_matrix.json`
- `results/qwen35_report_audit_20260619/confidence_ledger.json`
- `results/qwen35_report_audit_20260619/repro_command_manifest.json`
- `results/qwen35_report_audit_20260619/command_reference_hygiene.json`
- `results/qwen35_report_audit_20260619/final_readiness_audit.json`
- `results/qwen35_report_audit_20260619/share_bundle_manifest.json`
- `results/qwen35_report_audit_20260619/rerun_delta_triage.json`
- `results/qwen35_report_audit_20260619/stage_reproduction_drilldown.json`
- `results/qwen35_report_audit_20260619/stage_route_decision_matrix.json`
- `results/qwen35_report_audit_20260619/preflight_repro.json`

## 7. Stage 读表规则

现场讨论 breakdown 时先分清五层时间：

- client latency：用户看到的端到端请求时间。
- stage lifecycle：`stage_input_received->stage_complete`，包含该 stage 内部排队。
- actual compute：例如 `preprocess_start->preprocess_end` 或
  `code2wav_decode_start->code2wav_decode_end`。
- handoff：例如 `talker_to_code2wav_hop`。
- collect wait：例如 `code2wav_window_collect`，代表 code2wav 等 talker codec chunk。

需要从结论追到原始 artifact 时，先读 Stage 因果图里的 manifest-backed 原始证据
Drilldown；那里把 stage 归因、瓶颈转移、连接健康度和 `manifest.json` 里的 raw path
证据清单串在一起。需要讲 route-level 裁决和优化动作时，先读 Stage route 裁决矩阵；
需要按单个 stage row 复核时，再读 Stage 复现实操 Drilldown；
它把 `stage_row_id`、metric provenance row、jq 查询、raw artifact 和 rerun command ID
串在一起。

判定原则：

- lifecycle 变长但 actual compute 不变，优先看 queue/admission/backpressure。
- collect wait 变长但 decode 稳定，优先看 talker 输出节奏。
- handoff p95 稳定，不能说 stage 连接卡住。
- vLLM offline wall QPS 必须和 engine-side latency 分开解释。

## 8. 替换报告数字的红线

只有同时满足以下条件，才可以把重跑数字替换进最终分享稿：

- full audit 通过。
- SGLang/vLLM headline 仍满足 warmed c=4 latency/RTF 和 WER/accuracy 边界。
- SGLang stress 仍覆盖 c=1/2/4/8/16。
- short/long text-to-speech 仍覆盖 c=1/4/8。
- stage interaction summary 四个关键布尔结论仍为 `true`。
- acceptance matrix 仍为 `17/17`。
- confidence ledger 仍无 unsupported claim。
- final completion audit 仍为 `ready=true`，且只有 2026-06-21 18:00 UTC+08:00 后才允许 `completion_allowed_now=true`。
- 新硬件、新镜像、新数据 cache 或新模型路径都在 environment snapshot 和 manifest 中可见。

## 9. 失败分流

常见失败处理：

- Docker image 或 GPU warning：先判断是否只是接收方机器没有部署完整复现环境。
- Whisper cache warning：可忽略；需要 WER 时补 cache 或改用 ASR router。
- Manifest missing：先补 artifact 或从报告中移除对应引用。
- Claims failed：不要改文案绕过，先定位是哪条性能边界被新数据推翻。
- Coverage missing：说明报告包里某个用户要求没有可追踪证据，必须补文档或补机器证据。
- vLLM c=8 变好：先确认是否是 prebuild/offline diagnostic，不要直接改成 online parity。

## 10. 对外一句话版本

当前可以高置信对外说：在 8x H20、本地 Video-AMME ci-50 speech-output
workload、warmed c=4 严格对比上，优化后的 SGLang-Omni Qwen3.5 比优化版
vLLM 更快且 WER/accuracy 不退化；SGLang c=8 是当前吞吐峰值；高并发主要瓶颈在
admission/queueing 和 talker AR tail，不在 code2wav decode 或 talker->code2wav
连接。
