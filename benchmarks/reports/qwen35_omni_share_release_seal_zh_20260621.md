# Qwen3.5-Omni Share Release Seal

生成时间 UTC：`2026-06-21T02:00:34.035079+00:00`。
工作目录：`/home/gangouyu/sglang-omni`。

这份 release seal 是便捷 tarball 外侧的伴随封口证据，用来在发送前一次性确认
tarball、checksum、package manifest、tarball-mode validation、receiver smoke、
extracted-only validation、standalone validation、evidence-query host/portable smoke、
final readiness、final completion audit 和必须携带 caveat。
它描述 tarball 自身，因此不是 tarball 内成员。

## 1. Seal 状态

| Gate | Value |
| --- | ---: |
| Ready | True |
| Checks | 14/14 |
| Required failures | 0 |
| Send decision | `send_tarball_with_adjacent_seal_and_caveats` |
| Tarball SHA-256 | `sha256:ec16e578682c1d431510fbd1a44e80f0f95f1bbc305abf3d84274cc69239d746` |
| Tarball bytes | 570445 |
| Package files | 122 |
| Manifest records | 196 |
| Final readiness checks | 49 |
| Repro commands | 63 |
| Adjacent artifacts | 14/16 hashed; 2 self-reference omitted |
| Goal complete | False |
| Completion allowed now | True |
| Completion blockers | `` |

## 2. 发送文件

| 文件 | 用途 |
| --- | --- |
| `results/qwen35_report_audit_20260619/qwen35_omni_share_bundle_20260621.tar.gz` | 便捷发送包 |
| `results/qwen35_report_audit_20260619/qwen35_omni_share_bundle_20260621.tar.gz.sha256` | `sha256sum -c` 校验输入 |
| `results/qwen35_report_audit_20260619/share_bundle_package_manifest.json` | tarball 文件数、source manifest、tarball SHA-256 |
| `results/qwen35_report_audit_20260619/share_package_validation.json` | tarball-mode validation 机器证据 |
| `results/qwen35_report_audit_20260619/share_package_receiver_smoke_validation.json` | receiver smoke 机器证据 |
| `results/qwen35_report_audit_20260619/share_package_validation_extracted.json` | 手动 extracted-only validation 机器证据 |
| `results/qwen35_report_audit_20260619/share_package_external_standalone_validation.json` | 干净目录 standalone validation 机器证据 |
| `results/qwen35_report_audit_20260619/evidence_query_cards_smoke_summary.json` | evidence-query host/portable 只读烟测摘要；绑定当前 tarball SHA-256 |
| `results/qwen35_report_audit_20260619/evidence_query_cards_smoke_host.summary.out` | host 仓库态 evidence-query smoke 控制台摘要 |
| `results/qwen35_report_audit_20260619/evidence_query_cards_smoke_host.query.out` | host 仓库态 evidence-query 原始查询输出 |
| `results/qwen35_report_audit_20260619/evidence_query_cards_smoke_portable.summary.out` | portable 解包态 evidence-query smoke 控制台摘要 |
| `results/qwen35_report_audit_20260619/evidence_query_cards_smoke_portable.query.out` | portable 解包态 evidence-query 原始查询输出 |
| `results/qwen35_report_audit_20260619/final_completion_audit.json` | 最终 completion audit 机器证据；更新后的目标取消 6.21 晚间等待，completion_allowed_now=true 才能标记完成 |
| `results/qwen35_report_audit_20260619/share_release_seal.json` | 本 seal 的机器可读版本 |
| `benchmarks/reports/qwen35_omni_final_completion_audit_zh_20260621.md` | 最终 completion audit 中文可读版本 |
| `benchmarks/reports/qwen35_omni_share_release_seal_zh_20260621.md` | 本 seal 的中文可读版本 |

## 3. 关键 Gate 摘要

| Gate | Evidence |
| --- | --- |
| final readiness | `ready=True`, `checks=49/49`, `required_failures=0` |
| final completion audit | `ready=True`, `completion_allowed_now=True`, `recommendation=safe_to_call_update_goal_complete` |
| share package validation | `ready=True`, `checks=17/17`, `required_failures=0` |
| receiver smoke | `ready=True`, `receiver_smoke_ready=True` |
| extracted-only validation | `ready=True`, `checks=13/13` |
| external standalone validation | `ready=True`, `checks=8/8` |
| evidence-query smoke | `ready=True`, `host_pass=21`, `portable_pass=19` |
| runtime fairness | `warmed c=4 only`, `offline_diagnostic_not_online_parity` |
| vLLM c=8 caveat | `online_parity_proven=False` |
| chart source consistency | `ready=True`, `byte_exact_files=14` |

## 4. Machine Checks

| Status | Required | Check | Evidence |
| --- | --- | --- | --- |
| PASS | yes | tarball and checksum agree | tarball, checksum, package manifest, tarball validation and receiver smoke all agree on `sha256:ec16e578682c1d431510fbd1a44e80f0f95f1bbc305abf3d84274cc69239d746`; bytes `570445`. |
| PASS | yes | share bundle package manifest ready | package ready `True`, files `122`, source records `122`. |
| PASS | yes | tarball-mode validation ready | tarball validation `17/17`, required failures `0`, tar members `124`. |
| PASS | yes | receiver smoke validation ready | receiver smoke `17/17`, receiver_smoke_ready `True`. |
| PASS | yes | extracted-only validation ready | extracted-only `13/13`, required failures `0`. |
| PASS | yes | external standalone validation ready | standalone `8/8`, required failures `0`. |
| PASS | yes | release seal remains outside tarball | forbidden tarball members `[]`; adjacent artifacts `16`. |
| PASS | yes | full audit and final readiness are green | audit ok; final readiness `49/49`, required failures `0`. |
| PASS | yes | objective is share-ready with documented caveats | objective={'share_ready_with_documented_caveats': True, 'goal_complete': False, 'rows_total': 17, 'required_failures': 0, 'boundary_items': 3, 'preflight_pending_in_full_audit': False, 'status_counts': {'PASS': 14, 'PASS_WITH_CAVEAT': 3}, 'send_decision': 'ready_to_share_with_documented_caveats'} |
| PASS | yes | completion watchlist allows evidence-based completion | checkpoint={'ready': True, 'checks_total': 24, 'checks_passed': 24, 'required_failures': 0, 'watch_items_total': 7, 'checkpoint': 'evidence-ready completion audit', 'current_time_local': '2026-06-21T10:00:33.672198+08:00', 'checkpoint_start_local': '2026-06-21T18:00:00+08:00', 'checkpoint_phase': 'completion_audit_ready', 'seconds_until_checkpoint': 0, 'share_ready_with_documented_caveats': True, 'final_completion_evidence_ready': True, 'preflight_pending_in_full_audit': False, 'final_readiness_pending_in_full_audit': False, 'goal_complete': False, 'completion_allowed_now': True, 'completion_blockers': [], 'current_decision': 'run_final_completion_audit_now'} |
| PASS | yes | core audit inventory gates are current | preflight checks `62`, manifest records `196`, repro commands `63`, evidence-query host/portable `21/19`. |
| PASS | yes | runtime and optimization contracts are green | runtime image `12/12`, comparison `9/9`, SGLang lock `26/26`, vLLM lock `22/22`; headline `warmed c=4 only`. |
| PASS | yes | vLLM c8 caveat is preserved | vLLM online protocol `18/18`; online_parity_proven `False`. |
| PASS | yes | chart source and share bundle gates are green | chart source `8/8`, byte-exact files `14`; share bundle records `122`. |

## 5. 接收方第一组命令

```bash
HOST_REPO="${HOST_REPO:-/home/gangouyu/sglang-omni}"
SMOKE_DIR="${SMOKE_DIR:-/tmp/qwen35_omni_receiver_smoke_release_seal}"
EXTRACT_DIR="${EXTRACT_DIR:-/tmp/qwen35_omni_share_bundle_release_seal}"
STANDALONE_DIR="${STANDALONE_DIR:-/tmp/qwen35_omni_external_standalone_bundle_validation_release_seal}"
cd "$HOST_REPO"
bash benchmarks/eval/qwen35_omni_receiver_quickcheck.sh
python3 -m benchmarks.eval.build_qwen35_omni_share_release_seal \
  --root "$HOST_REPO" \
  --strict \
  --output benchmarks/reports/qwen35_omni_share_release_seal_zh_20260621.md \
  --json-output results/qwen35_report_audit_20260619/share_release_seal.json
```

## 6. 必须携带的 Caveat

- c=16 is pressure-boundary evidence, not a recommended serving point.
- Official SeedTTS full-set is not staged as headline evidence in this package.
- Strict vLLM c=8 online serving parity still needs an online-ingress rerun if that claim is required.

## 7. 不可提前升级

- 不把当前 vLLM c=8 prebuild w4 写成 online serving parity。
- 不把 c=16 写成默认推荐服务点。
- 不把 official SeedTTS full-set 写成 headline benchmark。
- 不在 final completion audit 显示 `completion_allowed_now=true` 前把长线目标标记 complete。

这份 seal 证明当前 tarball 和伴随验证证据可以带 caveat 分享；是否完成由 final completion audit 的证据门决定。
