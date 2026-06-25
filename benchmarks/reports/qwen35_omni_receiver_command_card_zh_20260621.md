# Qwen3.5-Omni 接收方命令卡

状态：2026-06-21 evidence-ready 交付稿；更新后的目标不再等待 6.21 晚上。
用途：给合作高校或外部 reviewer 一个一页式复制入口。完整解释仍以
`qwen35_omni_share_package_index_zh_20260621.md`、`qwen35_omni_external_handoff_runbook_zh_20260621.md`
和 `qwen35_omni_reproduction_checklist_zh_20260621.md` 为准。

## 0. 路径变量

```bash
export HOST_REPO="${HOST_REPO:-/home/gangouyu/sglang-omni}"
export SMOKE_DIR="${SMOKE_DIR:-/tmp/qwen35_omni_receiver_smoke_card}"
export EXTRACT_DIR="${EXTRACT_DIR:-/tmp/qwen35_omni_share_bundle_card}"
export STANDALONE_DIR="${STANDALONE_DIR:-/tmp/qwen35_omni_external_standalone_bundle_validation_card}"
cd "$HOST_REPO"
```

`HOST_REPO` 是可执行 full audit 和性能复跑的仓库根目录。解包目录只用于阅读和
extracted-only validation，不用于替换 benchmark artifact。

## 1. 收包快检

```bash
bash benchmarks/eval/qwen35_omni_receiver_quickcheck.sh
```

现场答辩或收包后想快速自证“查询卡里的命令仍能跑通”，再跑一条只读 smoke：

```bash
bash benchmarks/eval/qwen35_omni_evidence_query_cards_smoke.sh
```

显式区分 host 仓库态和解包 bundle 态时，用下面两条：

```bash
bash benchmarks/eval/qwen35_omni_evidence_query_cards_smoke.sh --root "$HOST_REPO" --mode host

export BUNDLE_ROOT="${BUNDLE_ROOT:-${EXTRACT_DIR}/qwen35_omni_share_bundle_20260621}"
bash "$BUNDLE_ROOT/benchmarks/eval/qwen35_omni_evidence_query_cards_smoke.sh" --root "$BUNDLE_ROOT" --mode portable
```

host 仓库态会同时校验相邻 package validation JSON；解包后的 bundle 态会自动进入
portable 子集，跳过只存在于仓库相邻结果目录的 package-validation 查询卡。这条命令不跑
benchmark，只验证公开查询入口能读到关键证据。
`PASS`/skip 摘要打印在 stdout；`--output` 只保存查询卡 bash block 的 JSON/文本正文。
如果要留档，建议把 stdout 用 `tee` 保存成 `*.summary.out`，把 `--output` 保存成
`*.query.out`。

合作方复跑完成并返回完整仓库证据后，再跑一条 headline 替换评审自检：

```bash
python3 -m benchmarks.eval.qwen35_omni_collaborator_return_check \
  --root "$HOST_REPO" \
  --strict \
  --json-output results/qwen35_report_audit_20260619/collaborator_return_check.json
```

通过标准：

- `sha256sum` 输出 `OK`。
- tarball validation 为 `17/17`。
- receiver smoke validation 为 `17/17`，且 `receiver_smoke_ready=true`。
- extracted-only validation 为 `13/13`。
- standalone validation 为 `8/8`。
- package asset quality evidence 显示 `report_quality_offenders=[]`、
  `chart_quality_offenders=[]`、`identity_hash_offenders=[]`。
- 相邻 `results/qwen35_report_audit_20260619/final_completion_audit.json` 为
  `ready=true`、required failures 为 `0`；更新后的目标不再等待 6.21 晚上，
  `completion_allowed_now=true` 时才可标记完成。可读版为
  `benchmarks/reports/qwen35_omni_final_completion_audit_zh_20260621.md`。
- extracted-only/standalone validation 如果显示 `audit_run_summary.json` 为
  `in_progress=true`，只能在 `recovered_from_in_progress_gates=True` 且包内
  `direct_rerun_delta_triage` 为 `ready=true`、`rows_total>=19`、`checks_passed>=8`、
  `required_failures=0` 时接受；仓库根最终 summary 必须是完成态 `ok=true` 并顶层暴露
  `rerun_delta_triage`。

`qwen35_omni_receiver_quickcheck.sh` 会串起 checksum、tarball-mode validation、
receiver-smoke validation、extracted-only validation 和 standalone validation；`STANDALONE_DIR`
只用于干净 `/tmp` 解包验证。

快检失败分流：

| 失败点 | 先看证据 | 裁决 |
| --- | --- | --- |
| checksum FAIL | `.tar.gz.sha256` 和 `sha256sum -c` 输出 | 不进入报告阅读；先重传 tarball/checksum。 |
| tarball validation FAIL | `results/qwen35_report_audit_20260619/share_package_validation.json` | 不发送为最终包；修复缺失/mismatch 后重建 tarball。 |
| receiver smoke FAIL | `results/qwen35_report_audit_20260619/share_package_receiver_smoke_validation.json` | 不替换任何 benchmark 数字；先确认安全解包和 nested extracted-only gate。 |
| extracted-only validation FAIL | `results/qwen35_report_audit_20260619/share_package_validation_extracted.json` | 说明解包目录或随包文件不自洽；回到路径手册确认 `BUNDLE_ROOT` / `HOST_REPO`。 |
| standalone validation FAIL | `results/qwen35_report_audit_20260619/share_package_external_standalone_validation.json` | 说明随包 validator 不能在干净 `/tmp` 独立运行；先修包，不进入外部复现。 |
| quality offenders 非空 | validation JSON 里的 `report_quality_offenders` / `chart_quality_offenders` | 先修 Markdown/CSV/SVG 质量，再重跑 full audit 和 release seal。 |

## 2. 先读哪些文件

```bash
less benchmarks/reports/qwen35_omni_evidence_query_cards_zh_20260621.md
less benchmarks/reports/qwen35_omni_share_package_index_zh_20260621.md
less benchmarks/reports/qwen35_omni_university_technical_report_zh_20260621.md
less benchmarks/reports/qwen35_omni_one_page_scorecard_zh_20260621.md
less benchmarks/reports/qwen35_omni_length_regime_coverage_zh_20260621.md
less benchmarks/reports/qwen35_omni_rerun_time_budget_zh_20260621.md
less benchmarks/reports/qwen35_omni_stage_reproduction_drilldown_zh_20260621.md
less benchmarks/reports/qwen35_omni_stage_metric_dictionary_zh_20260621.md
less benchmarks/reports/qwen35_omni_stage_latency_budget_zh_20260621.md
less benchmarks/reports/qwen35_omni_stage_boundary_bottleneck_ledger_zh_20260621.md
less benchmarks/reports/qwen35_omni_runtime_comparison_contract_zh_20260621.md
less benchmarks/reports/qwen35_omni_final_completion_audit_zh_20260621.md
```

这些入口分别覆盖：可执行证据查询、阅读顺序、可发中文正文、一页数字、
长短输入/输出覆盖、复跑耗时/算力预算、stage 复现实操 drilldown、
`quick_reproduction_map` 的 5 条答辩 quick route、stage metric dictionary 的
lifecycle/compute/handoff/collect wait 读法、stage latency 占比、
stage boundary 瓶颈、SGLang/vLLM 公平比较边界以及最终 completion gate。

## 3. 复核当前证据包

```bash
python3 -m benchmarks.eval.run_qwen35_omni_report_audit \
  --root "$HOST_REPO" \
  --summary-output results/qwen35_report_audit_20260619/audit_run_summary.json
```

当前 checkpoint 的最低通过标准：

- full audit `ok=true`。
- claims `17/17`，coverage `34/34`。
- final readiness `49/49`，required failures 为 `0`。
- SGLang optimization lock `26/26`。
- vLLM optimization lock `22/22`。
- vLLM online parity protocol `18/18`，且 `online_parity_proven=false`。
- length regime coverage `10/10`，短文本 74 chars / 12 words，长文本
  944 chars / 139 words，long c=8 RTF p95 小于 1。
- rerun time budget `ready=true`，包含 SGLang/vLLM timed lower bound、8-GPU 等效预算和 WER/ASR 未计时边界。
- share package validation `17/17`，extracted-only validation `13/13`，
  standalone validation `8/8`。
- final completion audit `ready=true`，required failures 为 `0`；更新后的目标不再等待
  2026-06-21 18:00 UTC+08:00，`completion_allowed_now=true` 表示可以进入完成裁决。

## 4. 性能复跑入口

SGLang 主压测和 short/long speech 复跑：

```bash
less benchmarks/reports/qwen35_omni_reproduction_checklist_zh_20260621.md
less benchmarks/reports/qwen35_omni_rerun_time_budget_zh_20260621.md
less benchmarks/reports/qwen35_omni_length_regime_coverage_zh_20260621.md
less benchmarks/reports/qwen35_omni_sglang_optimization_lock_zh_20260621.md
```

vLLM baseline 和 c=8 diagnostic 复跑：

```bash
less benchmarks/reports/qwen35_omni_vllm_optimization_lock_zh_20260621.md
less benchmarks/reports/qwen35_omni_vllm_online_parity_protocol_zh_20260621.md
```

如果现场只想按 command id 抽出可执行命令，用机器清单直接查：

```bash
jq -r --arg id sglang_videoamme_stress \
  '.commands[] | select(.id == $id) | .command' \
  results/qwen35_report_audit_20260619/repro_command_manifest.json

jq -r --arg id vllm_c8_prebuild_w4 \
  '.commands[] | select(.id == $id) | .command' \
  results/qwen35_report_audit_20260619/repro_command_manifest.json
```

常用入口：`launch_sglang_optimized`、`sglang_videoamme_stress`、
`sglang_synthetic_text_to_speech`、`check_wer_asr_path`、`sglang_recompute_wer`、
`vllm_c4_original`、`vllm_c8_original`、`vllm_c8_prebuild_w4`、`run_full_audit`。

复跑后填写：

```bash
less benchmarks/reports/qwen35_omni_collaborator_rerun_validation_sheet_zh_20260621.md
less benchmarks/reports/qwen35_omni_rerun_delta_triage_zh_20260621.md
```

## 5. WER/ASR 快查

先看当前 evidence 是否已经证明质量不退化：

```bash
jq '.checks[] | select(.name | test("WER|accuracy"))' \
  results/qwen35_report_audit_20260619/claims_verification.json

jq '.summary.checks.strict_c4_accuracy_wer_preserved' \
  results/qwen35_report_audit_20260619/headline_scorecard.json
```

如果需要在接收方机器上重算 WER，先记录 ASR 路径：

```bash
if [ -f /root/.cache/whisper/large-v3.pt ]; then
  echo "WER_ASR_PATH=local_whisper_cache"
  echo "WHISPER_CACHE=/root/.cache/whisper/large-v3.pt"
else
  echo "WER_ASR_PATH=asr_router_or_container_cache_required"
  echo "LOCAL_WHISPER_CACHE_MISSING=/root/.cache/whisper/large-v3.pt"
  echo "Use the ASR router command in qwen35_omni_stress_performance_plan_20260621.md section 14.2, or run WER inside a container with cached large-v3 weights."
fi
```

host 侧 `/root/.cache/whisper/large-v3.pt` 缺失是 optional warning；接收方可以使用
ASR router 或容器内 cached large-v3 权重。只有同一 ASR/WER 路径和 full audit 全绿时，
WER 复跑结果才可用于替换 headline。机器清单里的同源命令是
`repro_command_manifest.json` 的 `check_wer_asr_path`。

## 6. 数字替换红线

可以进入 headline replacement review 的前提是：同一 8x H20 硬件、同一
SGLang/vLLM image digest、同一 model、同一 Video-AMME cache、同一 ASR/WER 路径，
并且 full audit、rerun acceptance contract、share package validation、receiver smoke、
receiver quickcheck contract、extracted-only validation、standalone validation、final completion audit
和 release seal 全绿。

不能替换 headline 的情况：

- 硬件、镜像 digest、模型、数据 cache 或 ASR/WER 路径不一致。
- 任一 required gate 失败。
- vLLM c=8 仍只有 offline diagnostic，没有 online ingress artifact。
- 新数字只刷新了报告正文，没有同步刷新 machine evidence、图表、manifest、share bundle、
  checksum、receiver smoke、final completion audit 和 release seal。

## 7. 对外可说的最短结论

- 当前 8x H20、Video-AMME ci-50、warmed c=4 严格对比中，优化后的
  SGLang-Omni Qwen3.5 在 latency/RTF 上优于优化版 vLLM，accuracy/WER 不退化。
- SGLang c=8 是当前 recipe 的吞吐峰值，c=16 是压力边界。
- 主要瓶颈在高并发 admission/queueing 和 talker AR tail，不在
  `talker_ar -> code2wav` handoff 或 `code2wav_decode`。
- vLLM c=8 prebuild w4 是优化后的 offline diagnostic，不是 online serving parity。
