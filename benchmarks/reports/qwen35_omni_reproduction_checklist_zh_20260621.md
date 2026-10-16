# Qwen3.5-Omni 性能报告复现清单

状态：证据门已就绪的复现清单；更新后的目标不再等待 6.21 晚间，后续变更必须重跑 full audit。
适用目录：`/home/gangouyu/sglang-omni`。
主报告：
`/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_stress_performance_plan_20260621.md`
分享包索引：
`/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_share_package_index_zh_20260621.md`
接收方命令卡：
`/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_receiver_command_card_zh_20260621.md`
原始需求-证据映射：
`/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_requirement_evidence_map_zh_20260621.md`
压力条件总表：
`/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_pressure_matrix_zh_20260621.md`
数字来源索引：
`/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_metric_source_map_zh_20260621.md`
Stage 指标字典：
`/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_stage_metric_dictionary_zh_20260621.md`
Stage 因果图：
`/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_stage_causal_graph_zh_20260621.md`
Stage 因果图 JSON：
`/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/stage_causal_graph.json`
Stage 复现实操 Drilldown：
`/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_stage_reproduction_drilldown_zh_20260621.md`
Stage route 裁决矩阵：
`/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_stage_route_decision_matrix_zh_20260621.md`
长短输入/输出覆盖：
`/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_length_regime_coverage_zh_20260621.md`
Caveat 裁决矩阵：
`/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_caveat_adjudication_matrix_zh_20260621.md`
SGLang 优化锁定矩阵：
`/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_sglang_optimization_lock_zh_20260621.md`
vLLM 优化锁定矩阵：
`/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_vllm_optimization_lock_zh_20260621.md`
外部复现 handoff runbook：
`/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_external_handoff_runbook_zh_20260621.md`
合作方复跑验收表：
`/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_collaborator_rerun_validation_sheet_zh_20260621.md`
复跑差异定位矩阵：
`/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_rerun_delta_triage_zh_20260621.md`
复跑耗时/算力预算：
`/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_rerun_time_budget_zh_20260621.md`
答辩 Q&A：
`/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_defense_qna_zh_20260621.md`
优化 playbook：
`/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_optimization_playbook_zh_20260621.md`
分享 deck 提纲：
`/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_share_deck_outline_zh_20260621.md`

## 0. 路径映射约定

本文的 host 侧命令默认在 `/home/gangouyu/sglang-omni` 运行。合作方如果挂载到别的路径，
先设置 `HOST_REPO`；需要手动解包验证时再设置 `EXTRACT_DIR`：

```bash
export HOST_REPO="${HOST_REPO:-/home/gangouyu/sglang-omni}"
export EXTRACT_DIR="${EXTRACT_DIR:-/tmp/qwen35_omni_share_bundle_repro}"
```

host 侧 audit、report regeneration、tarball validation 都以 `$HOST_REPO` 为准；
container 内的 `/myapp/sglang-omni` 和 `/myapp/data/videoamme` 是容器挂载路径，不要和
host 侧 `$HOST_REPO` 混用。vLLM 复跑里需要绝对输出目录时，用
`"${HOST_REPO}/results/..."` 拼接，避免把新 artifact 写到旧机器路径。

## 1. 先验证证据包

在 host 上运行：

```bash
HOST_REPO="${HOST_REPO:-/home/gangouyu/sglang-omni}"
cd "$HOST_REPO"

python3 -m benchmarks.eval.run_qwen35_omni_report_audit \
  --root "$HOST_REPO" \
  --summary-output results/qwen35_report_audit_20260619/audit_run_summary.json
```

期望结果：

- `ok=true`
- claims: `17/17`
- coverage: `34/34`
- preflight: `62` checks, `0` required failures
- manifest current: `196` records, minimum `180`, `0` missing
- SGLang optimization lock: `ready=true`, checks `26/26`
- repro command manifest: `ready=true`, `63` commands / `7` phases
- command reference hygiene: `ready=true`, structured rerun command IDs resolve against `repro_command_manifest.json`
- metric provenance index: `ready=true`, metric rows link raw artifacts and rerun commands
- claim metric crosswalk: `ready=true`, defense claims link to concrete metric rows
- rerun time budget: `ready=true`, `9` budget rows, `6` timed rows, `0` required failures
- final readiness: `ready=true`, `49/49` checks, `0` required failures
- headline scorecard: `ready=true`, checks `9/9`
- acceptance matrix: `ready=true`, rows `17/17`
- confidence ledger: `ready=true`, entries `12/12`，high `9`、medium `3`、unsupported `0`
- environment snapshot: 8x NVIDIA H20，SGLang/vLLM Docker image 均 `ok=true`

关键 JSON：

- `results/qwen35_report_audit_20260619/audit_run_summary.json`
- `results/qwen35_report_audit_20260619/environment_snapshot.json`
- `results/qwen35_report_audit_20260619/manifest.json`
- `results/qwen35_report_audit_20260619/coverage_matrix.json`
- `results/qwen35_report_audit_20260619/claims_verification.json`
- `results/qwen35_report_audit_20260619/headline_scorecard.json`
- `results/qwen35_report_audit_20260619/metric_provenance_index.json`
- `results/qwen35_report_audit_20260619/claim_metric_crosswalk.json`
- `results/qwen35_report_audit_20260619/objective_requirement_crosswalk.json`
- `results/qwen35_report_audit_20260619/acceptance_matrix.json`
- `results/qwen35_report_audit_20260619/confidence_ledger.json`
- `results/qwen35_report_audit_20260619/repro_command_manifest.json`
- `results/qwen35_report_audit_20260619/command_reference_hygiene.json`
- `results/qwen35_report_audit_20260619/final_readiness_audit.json`
- `results/qwen35_report_audit_20260619/share_bundle_manifest.json`
- `results/qwen35_report_audit_20260619/share_package_external_standalone_validation.json`
- `results/qwen35_report_audit_20260619/stage_interaction_summary.json`
- `results/qwen35_report_audit_20260619/stage_reproduction_drilldown.json`
- `results/qwen35_report_audit_20260619/stage_route_decision_matrix.json`
- `results/qwen35_report_audit_20260619/length_regime_coverage.json`
- `results/qwen35_report_audit_20260619/rerun_time_budget.json`

## 2. 复现 SGLang-Omni 主压测

进入 SGLang container 后启动服务：

```bash
cd /myapp/sglang-omni

NO_CODE2WAV_TORCH_COMPILE=0 \
TORCHDYNAMO_DISABLE=0 \
SGLANG_OMNI_VIDEO_PREPROCESS_CACHE_MAX_BYTES=17179869184 \
SGLANG_OMNI_VIDEO_PREPROCESS_CACHE_MAX_ENTRIES=64 \
EXTRA_ARGS="--thinker-cuda-graph on --talker-cuda-graph on --talker-torch-compile on --thinker-max-running-requests 8 --talker-max-running-requests 8" \
bash examples/launch_qwen35_omni_speech_server_container.sh
```

服务 warmup 完成后跑 Video-AMME ci-50 c=1/2/4/8/16：

```bash
cd /myapp/sglang-omni

for C in 1 2 4 8 16; do
  RUN_ID="c${C}_profile_skipwer"
  RUN_ROOT="results/qwen35_sglang_mr8_stress_20260619"
  OUT_DIR="${RUN_ROOT}/benchmark_audio_50_${RUN_ID}"

  curl -s http://127.0.0.1:8161/start_request_profile \
    -H "Content-Type: application/json" \
    -d "{\"run_id\":\"${RUN_ID}\",\"event_dir\":\"/myapp/sglang-omni/${RUN_ROOT}/events\"}"

  HF_HOME=/myapp/data/videoamme \
  HF_DATASETS_CACHE=/myapp/data/videoamme/datasets \
  HF_HUB_OFFLINE=1 \
  python -m benchmarks.eval.benchmark_omni_videoamme \
    --model qwen3_5-omni --port 8161 \
    --repo-id zhaochenyang20/Video_AMME_ci \
    --output-dir "${OUT_DIR}" \
    --max-samples 50 --max-concurrency "${C}" \
    --max-tokens 256 --temperature 0.0 \
    --video-fps 2 --video-max-frames 128 --video-max-pixels 401408 \
    --enable-audio --audio-voice m02 --skip-wer --disable-tqdm

  curl -s http://127.0.0.1:8161/stop_request_profile \
    -H "Content-Type: application/json" \
    -d "{\"run_id\":\"${RUN_ID}\"}"

  python -m sglang_omni.profiler \
    "/myapp/sglang-omni/${RUN_ROOT}/events" \
    --format json \
    --out "/myapp/sglang-omni/${RUN_ROOT}/request_profile_${RUN_ID}.json"
done
```

本报告 checkpoint 中 c=1/c=2 使用 warmed artifact 名：
`c1_warm_profile_skipwer`、`c2_warm_profile_skipwer`。如果完全重跑，可按新 run id
重新生成表格；如果复核当前报告数字，应使用 manifest 中的既有 artifact。

期望形态：

- c=8 是吞吐峰值，约 `2.540 req/s`
- c=16 吞吐低于 c=8，约 `2.407 req/s`
- c=1/2/4 主要 `talker_ar` tail；c=8/c=16 出现 preprocessing admission / queueing
- `code2wav_decode` 不是主要瓶颈

## 3. 复现短/长文本输入 + 语音输出

短文本固定为 74 chars / 12 words，长文本固定为 944 chars / 139 words。复跑后
`tables_summary.json`、`share_charts/synthetic_short_long_speech.csv` 和
`length_regime_coverage.json` 应保留这两列，用于证明“长短文”口径不是只看音频时长。

```bash
cd /myapp/sglang-omni

for SCENARIO in short long; do
  for C in 1 4 8; do
    RUN_ID="${SCENARIO}_c${C}_profile"
    RUN_ROOT="results/qwen35_synthetic_speech_20260619"
    OUT_DIR="${RUN_ROOT}/${SCENARIO}_c${C}"
    SAMPLES=16
    if [ "${SCENARIO}" = "long" ]; then SAMPLES=8; fi

    curl -s http://127.0.0.1:8161/start_request_profile \
      -H "Content-Type: application/json" \
      -d "{\"run_id\":\"${RUN_ID}\",\"event_dir\":\"/myapp/sglang-omni/${RUN_ROOT}/events\"}"

    python -m benchmarks.eval.benchmark_qwen35_speech_synthetic \
      --model qwen3_5-omni --port 8161 \
      --scenario "${SCENARIO}" --samples-per-scenario "${SAMPLES}" \
      --output-dir "${OUT_DIR}" \
      --max-concurrency "${C}" \
      --voice m02 --max-tokens 1024 --temperature 0.0 \
      --disable-tqdm

    curl -s http://127.0.0.1:8161/stop_request_profile \
      -H "Content-Type: application/json" \
      -d "{\"run_id\":\"${RUN_ID}\"}"

    python -m sglang_omni.profiler \
      "/myapp/sglang-omni/${RUN_ROOT}/events" \
      --format json \
      --out "/myapp/sglang-omni/${RUN_ROOT}/request_profile_${RUN_ID}.json"
  done
done
```

期望形态：

- long c=8 仍快于实时：约 52.3s 生成音频，约 25.8s mean latency，RTF 约 0.4932
- long speech 主要瓶颈是 talker AR，不是 code2wav decode

## 4. 复现 SGLang WER

serving 压测结束后再跑 WER，避免 ASR 与 Qwen3.5 serving 抢 GPU。先记录接收方
ASR/WER 路径，host 侧 Whisper cache 缺失只是 optional warning，不是 serving benchmark
失败：

```bash
HOST_REPO="${HOST_REPO:-/home/gangouyu/sglang-omni}"
cd "$HOST_REPO"

if [ -f /root/.cache/whisper/large-v3.pt ]; then
  echo "WER_ASR_PATH=local_whisper_cache"
  echo "WHISPER_CACHE=/root/.cache/whisper/large-v3.pt"
else
  echo "WER_ASR_PATH=asr_router_or_container_cache_required"
  echo "LOCAL_WHISPER_CACHE_MISSING=/root/.cache/whisper/large-v3.pt"
  echo "Use the ASR router command in qwen35_omni_stress_performance_plan_20260621.md section 14.2, or run WER inside a container with cached large-v3 weights."
fi
```

然后在有 cached large-v3 权重或 ASR router 的环境中重算：

```bash
cd /myapp/sglang-omni

python -m benchmarks.eval.compute_audio_consistency_from_results \
  results/qwen35_sglang_mr8_stress_20260619/benchmark_audio_50_c1_warm_profile_skipwer/videoamme_results.json \
  results/qwen35_sglang_mr8_stress_20260619/benchmark_audio_50_c2_warm_profile_skipwer/videoamme_results.json \
  results/qwen35_sglang_mr8_stress_20260619/benchmark_audio_50_c4_profile_skipwer/videoamme_results.json \
  results/qwen35_sglang_mr8_stress_20260619/benchmark_audio_50_c8_profile_skipwer/videoamme_results.json \
  results/qwen35_sglang_mr8_stress_20260619/benchmark_audio_50_c16_profile_skipwer/videoamme_results.json \
  --path-root /myapp/sglang-omni \
  --local-whisper-model large-v3 \
  --asr-device cuda:1 \
  --output-name whisper_large_v3_local_wer.json \
  --lang en
```

期望形态：

- c=1/2/4/8/16 corpus WER 稳定在约 2.88%-3.85%
- throughput 峰值不是通过牺牲语音一致性得到的

## 5. 复现 vLLM baseline

vLLM 使用镜像：

```text
tongyi-duanwu-registry-vpc.cn-beijing.cr.aliyuncs.com/dashscope/dashllm:cuda129_cp312_test_vl_13589
```

严格 warmed c=4 对比使用既有 artifact：

```bash
HOST_REPO="${HOST_REPO:-/home/gangouyu/sglang-omni}"
cd "$HOST_REPO"

less results/qwen35_vllm_videoamme_ci50_official_talker_compile_c4_20260618_202106/run.log
less results/qwen35_vllm_videoamme_ci50_official_talker_compile_c4_20260618_202106/benchmark_audio_50_c4_offline_compile/vllm_videoamme_report.md
```

复跑 c=1/c=8：

```bash
HOST_REPO="${HOST_REPO:-/home/gangouyu/sglang-omni}"
cd "$HOST_REPO"

MAX_SAMPLES=50 MAX_CONCURRENCY=1 MAX_NUM_SEQS=8 \
RUN_TAG=ci50_offline_compile_c1_mns8_20260619 \
bash results/qwen35_vllm_videoamme_ci50_opt_20260618_162319/run_vllm_videoamme_ci5_offline_compile.sh

MAX_SAMPLES=50 MAX_CONCURRENCY=8 MAX_NUM_SEQS=8 \
RUN_TAG=ci50_offline_compile_c8_mns8_20260619 \
bash results/qwen35_vllm_videoamme_ci50_opt_20260618_162319/run_vllm_videoamme_ci5_offline_compile.sh
```

复跑当前最强 vLLM c=8 offline diagnostic：

```bash
HOST_REPO="${HOST_REPO:-/home/gangouyu/sglang-omni}"
cd "$HOST_REPO"

RUN_ROOT="${HOST_REPO}/results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuildw4_20260620_$(date +%H%M%S)" \
MAX_SAMPLES=50 MAX_CONCURRENCY=8 MAX_NUM_SEQS=8 \
RUN_TAG=ci50_offline_compile_c8_mns8_prebuildw4_20260620 \
EXTRA_ARGS="--prebuild-prompts --prebuild-workers 4" \
bash results/qwen35_vllm_videoamme_ci50_opt_20260618_162319/run_vllm_videoamme_ci5_offline_compile.sh
```

期望形态：

- warmed c=4 SGLang latency/RTF 优于 vLLM，WER 不退化
- vLLM original c=8 诊断为 `prompt_feed_limited`
- vLLM prebuild w4 完成 50/50，accuracy 66.0%，runner QPS 约 0.2127，engine QPS
  约 0.5360，admission span 约 4.09s / 4.89s
- 该 vLLM w4 结果是 offline diagnostic，不是 online serving parity 结论

## 6. 重新生成报告表格

跑完或复核 artifact 后，重新生成表格：

```bash
cd /home/gangouyu/sglang-omni

python3 -m benchmarks.eval.summarize_qwen35_omni_report_artifacts \
  --root /home/gangouyu/sglang-omni \
  --json-output results/qwen35_report_audit_20260619/tables_summary.json
```

复核 claims：

```bash
python3 -m benchmarks.eval.verify_qwen35_omni_report_claims \
  --root /home/gangouyu/sglang-omni \
  --json-output results/qwen35_report_audit_20260619/claims_verification.json
```

复核 coverage：

```bash
python3 -m benchmarks.eval.summarize_qwen35_report_coverage \
  --root /home/gangouyu/sglang-omni \
  --strict \
  --json-output results/qwen35_report_audit_20260619/coverage_matrix.json
```

复核 stage interaction 机器摘要：

```bash
python3 -m benchmarks.eval.summarize_qwen35_stage_interactions \
  --root /home/gangouyu/sglang-omni \
  --json-output results/qwen35_report_audit_20260619/stage_interaction_summary.json
```

复核 headline scorecard：

```bash
python3 -m benchmarks.eval.build_qwen35_omni_headline_scorecard \
  --root /home/gangouyu/sglang-omni \
  --json-output results/qwen35_report_audit_20260619/headline_scorecard.json
```

复核 metric provenance index：

```bash
python3 -m benchmarks.eval.build_qwen35_omni_metric_provenance_index \
  --root /home/gangouyu/sglang-omni \
  --strict \
  --json-output results/qwen35_report_audit_20260619/metric_provenance_index.json
```

复核 claim metric crosswalk：

```bash
python3 -m benchmarks.eval.build_qwen35_omni_claim_metric_crosswalk \
  --root /home/gangouyu/sglang-omni \
  --strict \
  --json-output results/qwen35_report_audit_20260619/claim_metric_crosswalk.json
```

复核 objective requirement crosswalk：

```bash
python3 -m benchmarks.eval.build_qwen35_omni_objective_requirement_crosswalk \
  --root /home/gangouyu/sglang-omni \
  --strict \
  --json-output results/qwen35_report_audit_20260619/objective_requirement_crosswalk.json
```

复核 acceptance matrix：

```bash
python3 -m benchmarks.eval.build_qwen35_omni_acceptance_matrix \
  --root /home/gangouyu/sglang-omni \
  --json-output results/qwen35_report_audit_20260619/acceptance_matrix.json
```

复核 confidence ledger：

```bash
python3 -m benchmarks.eval.build_qwen35_omni_confidence_ledger \
  --root /home/gangouyu/sglang-omni \
  --json-output results/qwen35_report_audit_20260619/confidence_ledger.json
```

复核原始目标完成度：

```bash
python3 -m benchmarks.eval.build_qwen35_omni_objective_completion_audit \
  --root /home/gangouyu/sglang-omni \
  --strict \
  --json-output results/qwen35_report_audit_20260619/objective_completion_audit.json
```

复核机器可读复现命令清单：

```bash
python3 -m benchmarks.eval.build_qwen35_omni_repro_command_manifest \
  --root /home/gangouyu/sglang-omni \
  --strict \
  --json-output results/qwen35_report_audit_20260619/repro_command_manifest.json
```

复核复跑差异定位矩阵：

```bash
python3 -m benchmarks.eval.build_qwen35_omni_rerun_delta_triage \
  --root /home/gangouyu/sglang-omni \
  --strict \
  --output benchmarks/reports/qwen35_omni_rerun_delta_triage_zh_20260621.md \
  --json-output results/qwen35_report_audit_20260619/rerun_delta_triage.json
```

复核最终分享 readiness：

```bash
python3 -m benchmarks.eval.build_qwen35_omni_final_readiness \
  --root /home/gangouyu/sglang-omni \
  --strict \
  --json-output results/qwen35_report_audit_20260619/final_readiness_audit.json
```

重建最终状态摘要：

```bash
python3 -m benchmarks.eval.build_qwen35_omni_final_status_summary \
  --root /home/gangouyu/sglang-omni \
  --output benchmarks/reports/qwen35_omni_final_status_summary_zh_20260621.md
```

重建 reviewer 决策矩阵：

```bash
python3 -m benchmarks.eval.build_qwen35_omni_regime_decision_matrix \
  --root /home/gangouyu/sglang-omni \
  --output benchmarks/reports/qwen35_omni_regime_decision_matrix_zh_20260621.md
```

重建 runtime 公平对比合同：

```bash
python3 -m benchmarks.eval.build_qwen35_omni_runtime_comparison_contract \
  --root /home/gangouyu/sglang-omni \
  --output benchmarks/reports/qwen35_omni_runtime_comparison_contract_zh_20260621.md
```

重建 SGLang 优化锁定矩阵：

```bash
python3 -m benchmarks.eval.build_qwen35_omni_sglang_optimization_lock \
  --root /home/gangouyu/sglang-omni \
  --strict \
  --output benchmarks/reports/qwen35_omni_sglang_optimization_lock_zh_20260621.md \
  --json-output results/qwen35_report_audit_20260619/sglang_optimization_lock.json
```

重建 vLLM 优化锁定矩阵：

```bash
python3 -m benchmarks.eval.build_qwen35_omni_vllm_optimization_lock \
  --root /home/gangouyu/sglang-omni \
  --strict \
  --output benchmarks/reports/qwen35_omni_vllm_optimization_lock_zh_20260621.md \
  --json-output results/qwen35_report_audit_20260619/vllm_optimization_lock.json
```

重建 Stage 因果图：

```bash
python3 -m benchmarks.eval.build_qwen35_omni_stage_causal_graph \
  --root /home/gangouyu/sglang-omni \
  --output benchmarks/reports/qwen35_omni_stage_causal_graph_zh_20260621.md \
  --json-output results/qwen35_report_audit_20260619/stage_causal_graph.json \
  --strict
```

重建 Stage 复现实操 Drilldown：

```bash
python3 -m benchmarks.eval.build_qwen35_omni_stage_reproduction_drilldown \
  --root /home/gangouyu/sglang-omni \
  --strict \
  --output benchmarks/reports/qwen35_omni_stage_reproduction_drilldown_zh_20260621.md \
  --json-output results/qwen35_report_audit_20260619/stage_reproduction_drilldown.json
```

重建 Stage route 裁决矩阵：

```bash
python3 -m benchmarks.eval.build_qwen35_omni_stage_route_decision_matrix \
  --root /home/gangouyu/sglang-omni \
  --strict \
  --output benchmarks/reports/qwen35_omni_stage_route_decision_matrix_zh_20260621.md \
  --json-output results/qwen35_report_audit_20260619/stage_route_decision_matrix.json
```

重建 Caveat 裁决矩阵：

```bash
python3 -m benchmarks.eval.build_qwen35_omni_caveat_adjudication_matrix \
  --root /home/gangouyu/sglang-omni \
  --output benchmarks/reports/qwen35_omni_caveat_adjudication_matrix_zh_20260621.md
```

复核分享包文件 hash 清单：

```bash
python3 -m benchmarks.eval.build_qwen35_omni_share_bundle_manifest \
  --root /home/gangouyu/sglang-omni \
  --strict \
  --json-output results/qwen35_report_audit_20260619/share_bundle_manifest.json
```

复核/重建便捷发送 tarball：

```bash
HOST_REPO="${HOST_REPO:-/home/gangouyu/sglang-omni}"
cd "$HOST_REPO"

python3 -m benchmarks.eval.build_qwen35_omni_share_bundle_package \
  --root "$HOST_REPO" \
  --strict \
  --source-manifest results/qwen35_report_audit_20260619/share_bundle_manifest.json \
  --output results/qwen35_report_audit_20260619/qwen35_omni_share_bundle_20260621.tar.gz \
  --json-output results/qwen35_report_audit_20260619/share_bundle_package_manifest.json

sha256sum -c results/qwen35_report_audit_20260619/qwen35_omni_share_bundle_20260621.tar.gz.sha256
```

复核 tarball-mode share package validation：

```bash
HOST_REPO="${HOST_REPO:-/home/gangouyu/sglang-omni}"
cd "$HOST_REPO"

python3 -m benchmarks.eval.validate_qwen35_omni_share_package \
  --root "$HOST_REPO" \
  --strict \
  --json-output results/qwen35_report_audit_20260619/share_package_validation.json
```

复核 receiver smoke validation，一步完成 tarball 校验、安全解包和 nested extracted-only
validation：

```bash
HOST_REPO="${HOST_REPO:-/home/gangouyu/sglang-omni}"
cd "$HOST_REPO"

rm -rf /tmp/qwen35_omni_receiver_smoke_repro
python3 -m benchmarks.eval.validate_qwen35_omni_share_package \
  --root "$HOST_REPO" \
  --strict \
  --receiver-smoke-dir /tmp/qwen35_omni_receiver_smoke_repro \
  --json-output results/qwen35_report_audit_20260619/share_package_receiver_smoke_validation.json
```

复核离仓 extracted-only validation：

```bash
HOST_REPO="${HOST_REPO:-/home/gangouyu/sglang-omni}"
EXTRACT_DIR="${EXTRACT_DIR:-/tmp/qwen35_omni_share_bundle_repro}"
cd "$HOST_REPO"

rm -rf "$EXTRACT_DIR"
mkdir -p "$EXTRACT_DIR"
tar -xzf results/qwen35_report_audit_20260619/qwen35_omni_share_bundle_20260621.tar.gz \
  -C "$EXTRACT_DIR"

python3 benchmarks/eval/validate_qwen35_omni_share_package.py \
  --root "$EXTRACT_DIR/qwen35_omni_share_bundle_20260621" \
  --extracted-only \
  --strict \
  --json-output "$HOST_REPO/results/qwen35_report_audit_20260619/share_package_validation_extracted.json"
```

复核 external standalone validation，证明只用干净解包目录和随包 validator 也能自检：

```bash
HOST_REPO="${HOST_REPO:-/home/gangouyu/sglang-omni}"
cd "$HOST_REPO"

# command_id: validate_external_standalone_share_bundle
python3 -m benchmarks.eval.build_qwen35_omni_external_standalone_bundle_validation \
  --root "$HOST_REPO" \
  --strict \
  --work-dir /tmp/qwen35_omni_external_standalone_bundle_validation_repro \
  --json-output results/qwen35_report_audit_20260619/share_package_external_standalone_validation.json
```

期望形态：

- `share_package_validation.json` 为 `ready=true`，且 tarball-mode validation 为 `17/17`
- `share_package_receiver_smoke_validation.json` 为 `ready=true`，`receiver_smoke_ready=true`
- receiver smoke 中的 nested extracted-only validation 为 `13/13`
- 独立 `share_package_validation_extracted.json` 为 `ready=true`，且 extracted-only validation 为 `13/13`
- `share_package_external_standalone_validation.json` 为 `ready=true`，standalone validation 为 `8/8`
- tarball-mode、receiver smoke 和 extracted-only 的资产 evidence 均显示
  `report_quality_offenders=[]`、`chart_quality_offenders=[]`
- `report_quality_offenders=[]` 覆盖随包 `share_report` Markdown 的裸 hash、坏表格、重复 heading 和坏展示 token；
  `chart_quality_offenders=[]` 覆盖随包 `share_charts` CSV/SVG 可解析、非空和结构可渲染

## 7. 最终接受标准

复现或审计后的结果应满足：

- SGLang warmed c=4 在 latency mean/p95、RTF mean/p95 上优于 vLLM warmed c=4
- SGLang warmed c=4 accuracy >= vLLM warmed c=4，WER <= vLLM warmed c=4
- SGLang stress 覆盖 c=1/2/4/8/16，c=8 是吞吐峰值
- short/long text-to-speech 覆盖 c=1/4/8；短文本 74 chars / 12 words，长文本
  944 chars / 139 words；long c=8 快于实时
- `length_regime_coverage.json` 为 ready=true，7 rows、10/10 checks；long c=8
  RTF p95 小于 1，talker->code2wav hop 和 code2wav decode guard 均健康
- `rerun_time_budget.json` 为 ready=true，9 budget rows、6 timed rows、0 required failures；
  计时段下界明确不包含 server launch/warmup/WER/ASR
- stage breakdown 能解释 c=1/c=4 talker tail、c=8/c=16 preprocessing admission
- `code2wav_decode` 和 `talker_ar -> code2wav` hop 不构成主要瓶颈
- `stage_interaction_summary.json` 证明 stage 连接健康、vLLM c8 prompt-feed 受限、
  preprocessing 并发负优化等关键结论
- `headline_scorecard.json` 为 ready=true，且 9/9 headline checks 全部通过
- `metric_provenance_index.json` 为 ready=true，且 headline、stress、synthetic、
  vLLM diagnostic、acceptance 和 stage 指标均能追到 raw artifact 与复跑命令
- `claim_metric_crosswalk.json` 为 ready=true，且 10 个对外答辩主张均能追到
  concrete metric row、raw artifact 与复跑命令
- `objective_requirement_crosswalk.json` 为 ready=true，且 11 个原始需求行均能追到
  objective row、defense claim、metric row、raw artifact 与复跑命令
- `acceptance_matrix.json` 为 ready=true，且 17/17 pressure/diagnostic rows 全部通过
- `confidence_ledger.json` 为 ready=true，且 12/12 entries 全部通过；high=9、
  medium=3、unsupported=0
- `qwen35_omni_runtime_comparison_contract_zh_20260621.md` 明确 strict c=4
  headline、SGLang scaling、vLLM c=8 offline diagnostic 和无效 parity 比较边界
- `qwen35_omni_sglang_optimization_lock_zh_20260621.md` 和
  `sglang_optimization_lock.json` 明确 SGLang 镜像、compiled/graph recipe、
  c=8 峰值、stage handoff 和 anti-recipe 均锁定，且 26/26 checks 通过
- `qwen35_omni_vllm_optimization_lock_zh_20260621.md` 和
  `vllm_optimization_lock.json` 明确 vLLM 镜像、compile/CUDA graph/cache/prebuild
  开关均锁定，且 22/22 checks 通过
- `qwen35_omni_vllm_online_parity_protocol_zh_20260621.md` 和
  `vllm_online_parity_protocol.json` 明确 vLLM c=8 online parity 升级协议已锁定，
  且当前 `online_parity_proven=false`
- `qwen35_omni_stage_causal_graph_zh_20260621.md` 明确 admission/queue、
  talker cadence、stream hop、code2wav collect/decode 和 vLLM offline admission
  的因果关系
- `qwen35_omni_stage_reproduction_drilldown_zh_20260621.md` 和
  `stage_reproduction_drilldown.json` 给每个 stage row 对应的 jq 查询、
  metric provenance row、raw artifact 和 rerun command ID
- `qwen35_omni_stage_route_decision_matrix_zh_20260621.md` 和
  `stage_route_decision_matrix.json` 把 52 条 stage row 聚合为 11 条 route-level
  裁决、优化动作和安全说法
- `qwen35_omni_caveat_adjudication_matrix_zh_20260621.md` 明确 caveat
  的可分享边界、禁止说法、补跑升级条件和替换数字触发器
- `objective_completion_audit.json` 为 share_ready_with_documented_caveats=true，
  rows=17，required failures=0
- `repro_command_manifest.json` 为 ready=true，且包含 full audit、SGLang、vLLM、
  表格、图表、preflight、coverage、manifest 的必需复跑命令
- `command_reference_hygiene.json` 为 ready=true，且结构化 rerun command IDs
  均可解析到 `repro_command_manifest.json`
- `final_readiness_audit.json` 为 ready=true，且 required failures 为 0
- `share_bundle_manifest.json` 为 ready=true，且推荐发送文件、机器证据和图表资产均有 hash
- `share_bundle_package_manifest.json` 为 ready=true，且便捷 tarball 的 `.sha256` 校验通过
- `share_package_validation.json` 为 ready=true，tarball-mode validation 为 17/17
- `share_package_receiver_smoke_validation.json` 为 ready=true，receiver smoke 和 nested
  extracted-only validation 均通过
- `share_package_validation_extracted.json` 为 ready=true，extracted-only validation 为 13/13
- `share_package_external_standalone_validation.json` 为 ready=true，standalone validation 为 8/8
- share package asset evidence 中 `report_quality_offenders=[]` 且 `chart_quality_offenders=[]`
- preproc=2/preproc=4 负结果存在并被解释
- vLLM optimized image、c8 prompt-feed caveat、prebuild w4 诊断均存在
- SGLang/vLLM 优化开关、anti-recipe 和 base/optimized 边界均有明确证据
- `run_qwen35_omni_report_audit` 全部通过

若以上条件不满足，不应把重跑结果替换为最终分享数字；应先回到 artifact 和 run log
定位差异。
