# Qwen3.5-Omni 接收方路径手册

状态：2026-06-21 evidence-ready 交付稿；更新后的目标不再等待 6.21 晚上。
用途：合作高校或外部 reviewer 解包后，用这页区分生成端路径、接收方仓库路径、解包阅读路径和容器内路径。

## 1. 三个根目录

| 名称 | 变量 | 用途 | 当前默认值 |
| --- | --- | --- | --- |
| 生成端仓库根 | `HOST_REPO` | full audit、SGLang/vLLM 复跑、报告再生成、tarball 校验 | `/home/gangouyu/sglang-omni` |
| 解包阅读根 | `BUNDLE_ROOT` | 只读报告、查看随包 JSON、运行 extracted-only validation | `qwen35_omni_share_bundle_20260621` |
| 容器内仓库根 | container cwd | SGLang/vLLM 容器内 benchmark 命令 | `/myapp/sglang-omni` |

分享包里的 Markdown 仍保留 `/home/gangouyu/sglang-omni`，因为这是当前 evidence checkpoint 的生成端路径；接收方如果在不同目录复跑，不要手改报告里的历史证据路径，先设置下面变量：

```bash
export HOST_REPO="${HOST_REPO:-/home/gangouyu/sglang-omni}"
export EXTRACT_DIR="${EXTRACT_DIR:-/tmp/qwen35_omni_receiver_bundle}"
export STANDALONE_DIR="${STANDALONE_DIR:-/tmp/qwen35_omni_external_standalone_bundle_validation_path_map}"
export BUNDLE_ROOT="${BUNDLE_ROOT:-${EXTRACT_DIR}/qwen35_omni_share_bundle_20260621}"
```

## 2. 只读验收路径

只收到 tarball 时，先在仓库根执行 receiver quickcheck：

```bash
cd "$HOST_REPO"
bash benchmarks/eval/qwen35_omni_receiver_quickcheck.sh
```

只收到解包目录时，进入 `BUNDLE_ROOT` 跑 extracted-only validation：

```bash
cd "$BUNDLE_ROOT"
python3 benchmarks/eval/validate_qwen35_omni_share_package.py \
  --root "$PWD" \
  --extracted-only \
  --strict \
  --json-output "$HOST_REPO/results/qwen35_report_audit_20260619/share_package_validation_extracted.json"
```

这条路径只证明随包 Markdown、JSON、validator、SVG/CSV 和逐文件 hash 自洽。
tarball 入口的 quickcheck 会额外运行 standalone validation，证明同一件事可以在干净
`/tmp` 解包根中由随包 validator 独立完成。它们都不替代 SGLang/vLLM 性能复跑。

收包快检通过后，最终 goal completion 还要看 tarball 外侧相邻的
`results/qwen35_report_audit_20260619/final_completion_audit.json` 和
`benchmarks/reports/qwen35_omni_final_completion_audit_zh_20260621.md`。
更新后的目标不再等待 6.21 晚上；`completion_allowed_now=true` 表示可以进入最终完成裁决。

### audit summary 自引用状态

解包目录里的 `results/qwen35_report_audit_20260619/audit_run_summary.json` 可能显示
`in_progress=true`。这是 tarball 自引用哈希的生成顺序导致的：`audit_run_summary.json`
本身是包成员，但最终 summary 又要引用 share package、extracted-only validation 和 release
seal，因此包内副本只能保留审计进行态。

这不是性能证据降级，前提是 extracted-only/standalone validation 同时证明：
`direct_rerun_delta_triage` 来自包内
`results/qwen35_report_audit_20260619/rerun_delta_triage.json`，并且为
`ready=true`、`rows_total>=19`、`checks_passed>=8`、`required_failures=0`。仓库根
`HOST_REPO` 中最终完成态的 `audit_run_summary.json` 必须是 `ok=true`，并且顶层直接暴露
`rerun_delta_triage`。

快检失败分流：

| 失败点 | 先看证据 | 正确动作 |
| --- | --- | --- |
| checksum FAIL | `.tar.gz.sha256` 和 `sha256sum -c` 输出 | 回到发送方重传 tarball/checksum，不用解包结果做结论。 |
| tarball validation FAIL | `results/qwen35_report_audit_20260619/share_package_validation.json` | 在 `HOST_REPO` 修复缺失/mismatch 后重建 tarball。 |
| receiver smoke FAIL | `results/qwen35_report_audit_20260619/share_package_receiver_smoke_validation.json` | 先看安全解包和 nested extracted-only evidence，不替换 benchmark 数字。 |
| extracted-only validation FAIL | `results/qwen35_report_audit_20260619/share_package_validation_extracted.json` | 确认 `--root` 指向 `BUNDLE_ROOT`，不是仓库根或 tarball 所在目录。 |
| standalone validation FAIL | `results/qwen35_report_audit_20260619/share_package_external_standalone_validation.json` | 说明随包 validator 不能在干净 `/tmp` 独立运行；先修包再复现。 |
| quality offenders 非空 | validation JSON 里的 `report_quality_offenders` / `chart_quality_offenders` | 修 Markdown/CSV/SVG 质量后重跑 full audit、quickcheck 和 release seal。 |

## 3. 解包后优先打开的相对路径

| 目的 | 从 `BUNDLE_ROOT` 打开 |
| --- | --- |
| 分享包索引 | `benchmarks/reports/qwen35_omni_share_package_index_zh_20260621.md` |
| 可直接发给合作高校的技术正文 | `benchmarks/reports/qwen35_omni_university_technical_report_zh_20260621.md` |
| serving/capacity 运行决策矩阵 | `benchmarks/reports/qwen35_omni_serving_capacity_matrix_zh_20260621.md` |
| 分享一致性 guard | `benchmarks/reports/qwen35_omni_share_consistency_guard_zh_20260621.md` |
| 一页状态摘要 | `benchmarks/reports/qwen35_omni_final_status_summary_zh_20260621.md` |
| 一页核心数字 | `benchmarks/reports/qwen35_omni_one_page_scorecard_zh_20260621.md` |
| 外部复现 runbook | `benchmarks/reports/qwen35_omni_external_handoff_runbook_zh_20260621.md` |
| 复现命令清单 | `benchmarks/reports/qwen35_omni_reproduction_checklist_zh_20260621.md` |
| stage bottleneck 和连接判断 | `benchmarks/reports/qwen35_omni_stage_boundary_bottleneck_ledger_zh_20260621.md` |
| stage 因果关系和 raw path drilldown | `benchmarks/reports/qwen35_omni_stage_causal_graph_zh_20260621.md` |
| pressure × stage 热力总览 | `benchmarks/reports/qwen35_omni_pressure_stage_heatmap_zh_20260621.md` |
| vLLM baseline 边界 | `benchmarks/reports/qwen35_omni_vllm_optimization_lock_zh_20260621.md` |
| 优化候选裁决 | `benchmarks/reports/qwen35_omni_optimization_candidate_ledger_zh_20260621.md` |
| 机器审计摘要 | `results/qwen35_report_audit_20260619/audit_run_summary.json` |
| serving/capacity 机器 gate | `results/qwen35_report_audit_20260619/serving_capacity_matrix.json` |
| 分享一致性机器 gate | `results/qwen35_report_audit_20260619/share_consistency_guard.json` |
| pressure × stage heatmap 机器 gate | `results/qwen35_report_audit_20260619/pressure_stage_heatmap.json` |
| 机器可读复跑命令 | `results/qwen35_report_audit_20260619/repro_command_manifest.json` |
| 随包逐文件 hash | `PACKAGE_FILE_SHA256SUMS.txt` |

## 4. Tarball 相邻伴随终检证据

下面这些文件不是 `BUNDLE_ROOT` 内成员；它们描述 tarball/checksum/收包验证本身，
应与 tarball 和 `.sha256` 放在同一个发送目录，或在接收方 `HOST_REPO` 下重跑生成：

| 目的 | 在 `HOST_REPO` 或 tarball 相邻目录查看 |
| --- | --- |
| final completion audit 机器 gate | `results/qwen35_report_audit_20260619/final_completion_audit.json` |
| final completion audit 可读页 | `benchmarks/reports/qwen35_omni_final_completion_audit_zh_20260621.md` |
| share release seal 机器 gate | `results/qwen35_report_audit_20260619/share_release_seal.json` |
| share release seal 可读页 | `benchmarks/reports/qwen35_omni_share_release_seal_zh_20260621.md` |
| tarball package manifest | `results/qwen35_report_audit_20260619/share_bundle_package_manifest.json` |
| tarball/receiver/extracted/standalone validation | `results/qwen35_report_audit_20260619/share_package_validation*.json` |

如果只拿到 `BUNDLE_ROOT` 而没有这些相邻终检证据，可以先做 extracted-only validation
确认随包内容自洽；但不能把它当作 final completion audit 或 release seal 的替代。

## 5. 什么不能从解包目录直接做

- 不在 `BUNDLE_ROOT` 里重跑 benchmark；它是轻量分享包，不包含大型 raw benchmark 输出。
- 不把 `BUNDLE_ROOT` 里的 JSON 当作新复跑结果覆盖主报告；复跑需要回到 `HOST_REPO`，并按 rerun acceptance contract 重新审计。
- 不把复跑命令里的新 `--output-dir` 当作当前 manifest 证据。例如 `results/qwen35_videoamme_seedtts_smoke_c8` 是 smoke 命令将创建的输出目录，不是随包既有 artifact。
- 不把不同硬件、不同 image digest、不同模型路径或不同数据 cache 的复跑数字直接替换 headline。
- 不把 vLLM c=8 prebuild w4 写成 online serving parity；当前它仍是 optimized offline diagnostic。

## 6. 复跑时的路径口径

完整复跑在 `HOST_REPO` 下执行：

```bash
cd "$HOST_REPO"
python3 -m benchmarks.eval.run_qwen35_omni_report_audit \
  --root "$HOST_REPO" \
  --summary-output results/qwen35_report_audit_20260619/audit_run_summary.json
```

进入容器后使用容器内路径 `/myapp/sglang-omni`，不要把 host 侧 `$HOST_REPO` 直接复制到容器命令里。SGLang/vLLM 具体命令以 `benchmarks/reports/qwen35_omni_reproduction_checklist_zh_20260621.md` 和 `results/qwen35_report_audit_20260619/repro_command_manifest.json` 为准。

## 7. 判断路径是否用错

| 现象 | 可能原因 | 正确动作 |
| --- | --- | --- |
| `PACKAGE_README.txt` 找不到 | 把 `--extracted-only --root` 指到了仓库根 | 改到 `BUNDLE_ROOT` |
| tarball `.sha256` 找不到 | 在解包目录跑 tarball-mode validation | 回到 `HOST_REPO` |
| raw benchmark artifact 找不到 | 只拿到了轻量分享包 | 先按复现清单重跑，或向发送方索取 raw artifact |
| 报告里 `/home/gangouyu/sglang-omni` 不存在 | 接收方机器路径不同 | 设置 `HOST_REPO`，并把报告里的绝对路径当作生成端证据路径 |
| vLLM c=8 数字看起来改善很多 | offline diagnostic 和 online serving parity 混用 | 先读 vLLM online parity protocol，再补 online ingress + WER/ASR |

## 8. 本页的验收口径

本页是接收方路径层的说明，不新增性能主张。它应随 share bundle 一起通过：

- final readiness `ready=true`
- share package validation `17/17`
- receiver smoke validation `receiver_smoke_ready=true`
- extracted-only validation `13/13`
- final completion audit `ready=true`，required failures 为 `0`；`completion_allowed_now=true`
  表示可以进入最终完成裁决
- `report_quality_offenders=[]`
- `chart_quality_offenders=[]`
