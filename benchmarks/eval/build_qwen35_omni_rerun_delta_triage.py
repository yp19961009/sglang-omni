# SPDX-License-Identifier: Apache-2.0
"""Build the external rerun-delta triage matrix for Qwen3.5-Omni."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AUDIT_DIR = Path("results/qwen35_report_audit_20260619")
DEFAULT_OUTPUT = Path(
    "benchmarks/reports/qwen35_omni_rerun_delta_triage_zh_20260621.md"
)
DEFAULT_JSON_OUTPUT = AUDIT_DIR / "rerun_delta_triage.json"


@dataclass(frozen=True)
class TriageRow:
    symptom: str
    likely_stage: str
    first_evidence: str
    decision_boundary: str
    next_action: str
    replacement_scope: str

    def to_dict(self) -> dict[str, str]:
        return {
            "symptom": self.symptom,
            "likely_stage": self.likely_stage,
            "first_evidence": self.first_evidence,
            "decision_boundary": self.decision_boundary,
            "next_action": self.next_action,
            "replacement_scope": self.replacement_scope,
        }


def _load_json_optional(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with path.open(encoding="utf-8") as fp:
            payload = json.load(fp)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)
        fp.write("\n")


def _md_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", "<br>")


def _rows() -> list[TriageRow]:
    return [
        TriageRow(
            symptom="收包或 full audit 失败",
            likely_stage="package gate / evidence integrity",
            first_evidence=(
                "share_package_validation.json, "
                "share_package_receiver_smoke_validation.json, "
                "audit_run_summary.json"
            ),
            decision_boundary="红：不要读性能结论，也不要替换 headline 数字。",
            next_action=(
                "先修 checksum、tar member、extracted-only validation、"
                "public_doc_quality_guard；通过后再讨论性能差异。"
            ),
            replacement_scope="不得替换",
        ),
        TriageRow(
            symptom="复跑机器、镜像、模型或数据 cache 与当前 checkpoint 不一致",
            likely_stage="environment / runtime image boundary",
            first_evidence=(
                "environment_snapshot.json, runtime_image_contract.json, "
                "rerun_acceptance_contract.json"
            ),
            decision_boundary="黄：只能确认趋势，不能直接替换当前 8x H20 headline。",
            next_action=(
                "在 collaborator rerun validation sheet 标注差异；"
                "同硬件同镜像重跑后再进入替换评审。"
            ),
            replacement_scope="附录趋势",
        ),
        TriageRow(
            symptom="SGLang warmed c=4 latency/RTF 不再优于 vLLM warmed c=4",
            likely_stage="strict c=4 comparison",
            first_evidence=(
                "headline_scorecard.json, claims_verification.json, "
                "runtime_comparison_contract.json"
            ),
            decision_boundary=(
                "红或黄：若环境不一致为黄；若同环境且 claims 失败为红，"
                "不能保留原 headline。"
            ),
            next_action=(
                "同时复核 SGLang c=4 profile、vLLM c=4 report、WER artifact；"
                "只在 rerun acceptance contract 通过后替换。"
            ),
            replacement_scope="需替换评审",
        ),
        TriageRow(
            symptom="c=8 不再是 SGLang 吞吐峰值",
            likely_stage="admission / queueing / running-request cap",
            first_evidence=(
                "tables_summary.json, stage_latency_budget.json, "
                "stage_interaction_summary.json"
            ),
            decision_boundary=(
                "黄：先作为复跑差异，不直接改推荐点；同环境确认两次后再改 c=8 结论。"
            ),
            next_action=(
                "检查 --thinker-max-running-requests、--talker-max-running-requests、"
                "warmup、请求 profile run_id 是否一致。"
            ),
            replacement_scope="附录趋势",
        ),
        TriageRow(
            symptom="c=16 吞吐没有下降，或 queue/admission 不明显",
            likely_stage="high-concurrency saturation boundary",
            first_evidence=(
                "stage_latency_budget.json, stage_boundary_bottleneck_ledger.json, "
                "request_profile_c16_profile_skipwer.json"
            ),
            decision_boundary=(
                "黄：说明本次 admission 形态不同；不要直接删除 c=16 saturation caveat。"
            ),
            next_action=(
                "确认并发驱动、样本数、warmup、GPU clock 和 profile 是否覆盖完整请求。"
            ),
            replacement_scope="附录趋势",
        ),
        TriageRow(
            symptom="prefill/queue 占比升高，mean/p95 latency 同时变差",
            likely_stage="preprocessor / admission queue",
            first_evidence=(
                "stage_latency_budget.json, stage_boundary_bottleneck_ledger.json, "
                "stage_causal_graph report"
            ),
            decision_boundary="黄：先定位为 admission 压力，不归因到 code2wav。",
            next_action=(
                "检查 video preprocessing cache、输入帧数、max pixels、HTTP client 并发和 "
                "profile 中 queue/wait 字段。"
            ),
            replacement_scope="附录趋势",
        ),
        TriageRow(
            symptom="talker_ar p95 明显拉长，但 code2wav decode 稳定",
            likely_stage="talker AR compute tail",
            first_evidence=(
                "stage_interaction_summary.json, stage_boundary_bottleneck_ledger.json, "
                "request_profile_*_profile*.json"
            ),
            decision_boundary=(
                "绿或黄：若 headline gate 仍过，可保留结论并把差异放在 tail analysis。"
            ),
            next_action=(
                "优先检查 talker torch compile、CUDA graph、max running requests、"
                "生成音频时长分布。"
            ),
            replacement_scope="可进入评审",
        ),
        TriageRow(
            symptom="talker_ar -> code2wav hop wait 升高",
            likely_stage="stream handoff boundary",
            first_evidence=(
                "stage_interaction_summary.json, stage_boundary_bottleneck_ledger.json, "
                "qwen35_omni_stage_causal_graph_zh_20260621.md"
            ),
            decision_boundary="红：若 handoff health 失败，不要继续声明 stage 连接健康。",
            next_action=(
                "先复核 request profile 中 collect wait、handoff wait、decode cadence；"
                "再判断是否为 talker 输出节奏或 code2wav 消费侧问题。"
            ),
            replacement_scope="需替换评审",
        ),
        TriageRow(
            symptom="code2wav_decode 成为 latency 主项",
            likely_stage="code2wav decode compute",
            first_evidence=(
                "stage_latency_budget.json, stage_boundary_bottleneck_ledger.json, "
                "sglang_optimization_lock.json"
            ),
            decision_boundary="红：当前报告的 code2wav-not-bottleneck 结论不再可直接引用。",
            next_action=(
                "检查 NO_CODE2WAV_TORCH_COMPILE、TORCHDYNAMO_DISABLE、模型权重、"
                "decode batch/cadence，并重新生成 stage ledger。"
            ),
            replacement_scope="需替换评审",
        ),
        TriageRow(
            symptom="long synthetic c=8 RTF 接近或超过 1",
            likely_stage="long-form talker AR / audio duration pressure",
            first_evidence=(
                "synthetic_short_long_speech.csv, tables_summary.json, "
                "request_profile_long_c8_profile.json"
            ),
            decision_boundary="红：不能继续说 long c=8 快于实时。",
            next_action=(
                "先确认 long prompt 是 944 chars / 139 words；再复核音频时长、"
                "max_tokens、voice 和 serving warmup。"
            ),
            replacement_scope="需替换评审",
        ),
        TriageRow(
            symptom="短/长文本输入形状不匹配",
            likely_stage="synthetic workload contract",
            first_evidence=(
                "tables_summary.json, synthetic_short_long_speech.csv, "
                "qwen35_omni_reproduction_checklist_zh_20260621.md"
            ),
            decision_boundary="红：长短文覆盖证据无效，不能替换相关表格。",
            next_action=(
                "按 reproduction checklist 重新跑 short 74 chars / 12 words、"
                "long 944 chars / 139 words 的固定输入。"
            ),
            replacement_scope="不得替换",
        ),
        TriageRow(
            symptom="复跑样本数、skip-first、warmup 或 fixed prompt 口径不一致",
            likely_stage="measurement protocol / warmup boundary",
            first_evidence=(
                "repro_command_manifest.json, rerun_acceptance_contract.json, "
                "headline_scorecard.json, tables_summary.json"
            ),
            decision_boundary=(
                "红或黄：协议口径不一致时不能替换主报告数字；最多作为附录趋势。"
            ),
            next_action=(
                "按 command ID 重新跑，保留 sample_count、skip_first、warmup、"
                "fixed short/long prompt 和 WER/ASR 口径；再重跑 full audit。"
            ),
            replacement_scope="不得替换",
        ),
        TriageRow(
            symptom="只重跑 benchmark 原始结果，没有重跑 regeneration/full audit",
            likely_stage="evidence regeneration boundary",
            first_evidence=(
                "audit_run_summary.json, final_readiness_audit.json, "
                "rerun_acceptance_contract.json"
            ),
            decision_boundary=(
                "红：raw 结果本身不足以替换 headline；必须证明生成表、图、证据索引和门禁同步。"
            ),
            next_action=(
                "运行 run_qwen35_omni_report_audit，确认 claims、acceptance、"
                "stage ledger、chart source consistency、final readiness 和 share package validation 全绿。"
            ),
            replacement_scope="不得替换",
        ),
        TriageRow(
            symptom="只修改公开 Markdown 或图表，没有同步 JSON/manifest/share bundle",
            likely_stage="silent replacement / share consistency boundary",
            first_evidence=(
                "share_consistency_guard.json, chart_source_consistency.json, "
                "share_bundle_manifest.json, share_release_seal.json"
            ),
            decision_boundary=(
                "红：这是 silent-replacement 风险；公开数字不能脱离机器证据、manifest 和 tarball seal。"
            ),
            next_action=(
                "从源 JSON 重生成报告和 chart pack，再重跑 share bundle manifest、"
                "tarball package、receiver smoke、extracted-only、standalone validation 和 receiver quickcheck。"
            ),
            replacement_scope="不得替换",
        ),
        TriageRow(
            symptom="WER/accuracy 退化或 ASR 路径改变",
            likely_stage="quality / ASR side channel",
            first_evidence=(
                "claims_verification.json, headline_scorecard.json, "
                "rerun_acceptance_contract.json"
            ),
            decision_boundary=(
                "红：accuracy/WER 不过 gate 时，不能只替换 latency/RTF headline。"
            ),
            next_action=(
                "离线单独复算 WER，确认 Whisper large-v3 权重或 ASR router 一致；"
                "避免与 serving 压测抢 GPU。"
            ),
            replacement_scope="不得替换",
        ),
        TriageRow(
            symptom="vLLM c=8 original offline 明显变好但 prebuild 证据缺失",
            likely_stage="vLLM offline prompt build/feed admission",
            first_evidence=(
                "vllm_admission_diagnosis.json, vllm_log_stage_summary.json, "
                "vllm_optimization_lock.json"
            ),
            decision_boundary=(
                "黄：仍是 offline runner 形态，不能升级为 online serving parity。"
            ),
            next_action=(
                "补跑 --prebuild-prompts --prebuild-workers 4；"
                "再按 online parity protocol 补 online ingress 与 WER/ASR。"
            ),
            replacement_scope="附录趋势",
        ),
        TriageRow(
            symptom="有人想用 vLLM c=8 prebuild w4 替换 online parity 结论",
            likely_stage="vLLM parity scope",
            first_evidence=(
                "vllm_online_parity_protocol.json, runtime_comparison_contract.json, "
                "caveat_adjudication_matrix report"
            ),
            decision_boundary="红：当前 package 明确 online_parity_proven=false。",
            next_action=(
                "必须先补 online serving artifact、同口径 WER/ASR、ingress latency 和 "
                "rerun acceptance contract。"
            ),
            replacement_scope="不得替换",
        ),
        TriageRow(
            symptom="naive preprocessing 并发看起来提升吞吐",
            likely_stage="preprocessing anti-recipe",
            first_evidence=(
                "sglang_optimization_lock.json, stage_interaction_summary.json, "
                "qwen35_omni_optimization_playbook_zh_20260621.md"
            ),
            decision_boundary=(
                "黄：先作为新候选 recipe，不覆盖当前 preproc=2 回退、preproc=4 OOM 边界。"
            ),
            next_action=(
                "至少补 c=4/c=8/c=16、WER、profile、OOM/稳定性记录，再进入 recipe 替换。"
            ),
            replacement_scope="附录趋势",
        ),
        TriageRow(
            symptom="图表或公开 Markdown 被手工改动后 validator 失败",
            likely_stage="share asset hygiene",
            first_evidence=(
                "share_package_validation.json, share_bundle_manifest.json, "
                "final_readiness_audit.json"
            ),
            decision_boundary="红：不要发送；先修报告资产，再重新打包。",
            next_action=(
                "重跑 share chart pack、share bundle manifest、tarball package、"
                "receiver smoke 和 extracted-only validation。"
            ),
            replacement_scope="不得替换",
        ),
    ]


def _summary_from_sources(root: Path) -> dict[str, Any]:
    audit_dir = root / AUDIT_DIR
    final = _load_json_optional(audit_dir / "final_readiness_audit.json")
    acceptance = _load_json_optional(audit_dir / "rerun_acceptance_contract.json")
    stage = _load_json_optional(audit_dir / "stage_interaction_summary.json")
    vllm = _load_json_optional(audit_dir / "vllm_online_parity_protocol.json")
    final_summary = final.get("summary", {})
    return {
        "final_readiness": {
            "ready": final_summary.get("ready"),
            "checks_total": final_summary.get("checks_total"),
            "checks_passed": final_summary.get("checks_passed"),
            "required_failures": final_summary.get("required_failures"),
        },
        "rerun_acceptance_contract": acceptance.get("summary", {}),
        "stage_interaction_summary": stage.get("summary", {}),
        "vllm_online_parity_protocol": vllm.get("summary", {}),
    }


def build_triage(root: Path) -> dict[str, Any]:
    root = root.resolve()
    rows = _rows()
    stage_names = {row.likely_stage for row in rows}
    replacement_scopes = {row.replacement_scope for row in rows}
    checks = {
        "rows_total": len(rows) >= 19,
        "stage_routes_present": {
            "package gate / evidence integrity",
            "environment / runtime image boundary",
            "strict c=4 comparison",
            "admission / queueing / running-request cap",
            "measurement protocol / warmup boundary",
            "evidence regeneration boundary",
            "silent replacement / share consistency boundary",
            "talker AR compute tail",
            "stream handoff boundary",
            "code2wav decode compute",
            "vLLM parity scope",
            "quality / ASR side channel",
        }.issubset(stage_names),
        "replacement_scopes_present": {
            "不得替换",
            "附录趋势",
            "需替换评审",
            "可进入评审",
        }.issubset(replacement_scopes),
        "safe_vllm_boundary_present": any(
            "online_parity_proven=false" in row.decision_boundary for row in rows
        ),
        "package_hygiene_route_present": any(
            "public_doc_quality_guard" in row.next_action
            or "validator" in row.symptom
            for row in rows
        ),
        "measurement_protocol_guard_present": any(
            "skip-first" in row.symptom
            and "sample_count" in row.next_action
            and "full audit" in row.next_action
            for row in rows
        ),
        "evidence_regeneration_guard_present": any(
            "regeneration/full audit" in row.symptom
            and "chart source consistency" in row.next_action
            and "final readiness" in row.next_action
            for row in rows
        ),
        "silent_replacement_guard_present": any(
            "silent replacement" in row.likely_stage
            and "tarball seal" in row.decision_boundary
            and "receiver quickcheck" in row.next_action
            for row in rows
        ),
    }
    required_failures = [name for name, ok in checks.items() if not ok]
    source_summaries = _summary_from_sources(root)
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "summary": {
            "ready": not required_failures,
            "rows_total": len(rows),
            "required_failures": len(required_failures),
            "checks_total": len(checks),
            "checks_passed": sum(1 for ok in checks.values() if ok),
            "stage_routes_total": len(stage_names),
            "replacement_scopes": sorted(replacement_scopes),
            "usage": (
                "Use after collaborator reruns to map metric deltas to stage evidence "
                "before changing headline numbers."
            ),
        },
        "checks": checks,
        "source_summaries": source_summaries,
        "rows": [row.to_dict() for row in rows],
        "required_failures": required_failures,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    sources = payload["source_summaries"]
    lines: list[str] = [
        "# Qwen3.5-Omni 外部复跑差异定位矩阵",
        "",
        "状态：证据门已就绪；更新后的目标不再等待 6.21 晚间，后续变更必须重跑 full audit。",
        "工作目录：`/home/gangouyu/sglang-omni`。",
        "用途：合作方复跑后，如果 latency、RTF、WER、stage 占比或 vLLM 结果和当前 checkpoint 不一致，先用这张表定位差异，再决定是否能替换 headline 数字。",
        "",
        "## 1. 使用原则",
        "",
        "1. 先跑收包快检和 full audit；package 或 audit 不绿时，不讨论性能结论。",
        "2. 再判断环境、镜像、模型、数据 cache、ASR/WER 路径是否与当前 checkpoint 一致。",
        "3. 只有同硬件、同镜像、同模型、同数据、同 ASR/WER 路径，并且 rerun acceptance contract 通过，才进入 headline 替换评审。",
        "4. vLLM c=8 prebuild w4 在当前包内仍是 optimized offline diagnostic；除非 online parity protocol 变为 proven，否则不能说成 online serving parity。",
        "5. stage 结论要从 request profile、stage latency budget、boundary ledger 和 causal graph 一起读，不用单个 latency 数字直接归因。",
        "6. 只重跑 raw benchmark 不足以替换数字；必须重跑 regeneration/full audit，并让报告、图表、JSON evidence 和 final readiness 同步全绿。",
        "7. 不能手工只改公开 Markdown 或图表；任何 headline 或 stage 数字替换都必须重新生成 manifest、share bundle、tarball seal 和 receiver quickcheck。",
        "",
        "## 2. 当前 gate 摘要",
        "",
        "| Gate | 当前摘要 |",
        "| --- | --- |",
        f"| triage ready | `{summary.get('ready')}`，rows `{summary.get('rows_total')}`，checks `{summary.get('checks_passed')}/{summary.get('checks_total')}` |",
        f"| final readiness | `{sources.get('final_readiness', {})}` |",
        f"| rerun acceptance | `{sources.get('rerun_acceptance_contract', {})}` |",
        f"| stage interaction | `{sources.get('stage_interaction_summary', {})}` |",
        f"| vLLM online parity | `{sources.get('vllm_online_parity_protocol', {})}` |",
        "",
        "## 3. 差异定位矩阵",
        "",
        "| 复跑症状 | 优先定位 stage / boundary | 第一证据 | 裁决边界 | 下一步动作 | 数字替换范围 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload["rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    _md_escape(row["symptom"]),
                    _md_escape(row["likely_stage"]),
                    _md_escape(row["first_evidence"]),
                    _md_escape(row["decision_boundary"]),
                    _md_escape(row["next_action"]),
                    _md_escape(row["replacement_scope"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 4. 快速裁决模板",
            "",
            "| 问题 | 需要填写 | 裁决提示 |",
            "| --- | --- | --- |",
            "| 复跑是否同 8x H20、同镜像、同模型、同数据 cache、同 ASR/WER 路径？ |  | 否则默认只能做附录趋势。 |",
            "| full audit、share package validation、receiver smoke、extracted-only validation 是否全绿？ |  | 任一失败时不得替换数字。 |",
            "| SGLang c=4 strict headline 是否仍优于 vLLM c=4？ |  | 失败时需要重新跑 strict pair 并更新 claims。 |",
            "| c=8 是否仍是 SGLang 主 recipe 的吞吐峰值？ |  | 失败时先查 admission、queue 和 running request cap。 |",
            "| long c=8 是否仍快于实时？ |  | 失败时不能继续引用 long c=8 RTF 结论。 |",
            "| code2wav decode 是否仍不是主瓶颈？ |  | 失败时必须更新 stage latency budget 和 boundary ledger。 |",
            "| vLLM c=8 是否已有 online serving parity artifact？ |  | 当前默认没有；prebuild w4 只能当 offline diagnostic。 |",
            "",
            "## 5. 证据入口",
            "",
            "- `results/qwen35_report_audit_20260619/rerun_delta_triage.json`：本矩阵的机器可读版本。",
            "- `benchmarks/reports/qwen35_omni_collaborator_rerun_validation_sheet_zh_20260621.md`：collaborator rerun validation sheet，合作方复跑填写表。",
            "- `benchmarks/reports/qwen35_omni_rerun_acceptance_contract_zh_20260621.md`：数字替换阈值合同。",
            "- `benchmarks/reports/qwen35_omni_stage_latency_budget_zh_20260621.md`：stage latency 占比。",
            "- `benchmarks/reports/qwen35_omni_stage_boundary_bottleneck_ledger_zh_20260621.md`：boundary-level bottleneck 账本。",
            "- `benchmarks/reports/qwen35_omni_stage_causal_graph_zh_20260621.md`：stage 因果和 manifest-backed drilldown。",
            "- `benchmarks/reports/qwen35_omni_vllm_online_parity_protocol_zh_20260621.md`：vLLM c=8 online parity 升级条件。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the external rerun-delta triage matrix."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    json_output = (
        args.json_output if args.json_output.is_absolute() else root / args.json_output
    )
    payload = build_triage(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_markdown(payload), encoding="utf-8")
    _save_json(payload, json_output)
    if args.strict and payload["summary"]["required_failures"]:
        raise SystemExit(
            "Rerun delta triage failed: "
            + ", ".join(payload.get("required_failures", []))
        )
    print(
        "rerun_delta_triage_ready="
        f"{payload['summary']['ready']} rows={payload['summary']['rows_total']} "
        f"checks={payload['summary']['checks_passed']}/{payload['summary']['checks_total']}"
    )


if __name__ == "__main__":
    main()
