# Qwen3.5-Omni 长短输入/输出 Length-Regime 覆盖矩阵

## 1. 结论

- Video-AMME ci-50 覆盖真实多模态样本：目标文本 199-484 chars，平均音频 26.0s。
- Synthetic short/long speech 覆盖固定短输入 74 chars / 12 words 和固定长输入 944 chars / 139 words，c=1/4/8 全部保留。
- long c=8 仍快于实时：RTF mean=0.4932，RTF p95=0.5001。
- Handoff/decode guard 仍健康：最大 talker->code2wav hop p95=24.0ms，最大 code2wav decode p95=24.3ms。
- ci-50/stress/synthetic 证据不能直接外推到完整线上流量；更大 Video-AMME 或真实线上流量需要同口径复跑和 gate 全绿。

## 2. 覆盖矩阵

| 覆盖项 | 并发 | 输入形状 | 输出形状 | Stage 压力 | 关键指标 | Stage 读法 | 可说结论 | 边界 |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- |
| Video-AMME ci-50 video + spoken question -> text + speech | 1/2/4/8/16 | target_chars=199-484, mean=317.3 | audio_duration=18.6s-38.5s, mean=26.0s | real multimodal preprocessing, encoder, Talker, code2wav, and admission queue | SGLang stress c=1/2/4/8/16; c=8 current throughput peak; c=16 saturation boundary | stage latency budget and stage boundary ledger carry per-stage percentages | Video-AMME ci-50 anchors the main single/high-concurrency performance shape | ci-50/stress/synthetic 证据不能直接外推到完整线上流量 |
| synthetic short text-to-speech | 1 | 74 chars / 12 words | audio_mean=4.2s | short-output thinker/talker/code2wav guard | lat_mean=0.866s, lat_p95=0.924s, RTF=0.2052, RTF_p95=0.2112, QPS=1.154 | talker_pct=98.0%, hop_p95=14.9ms, decode_p95=14.2ms | short/long speech guard remains faster than real time | synthetic evidence isolates speech output and does not replace full Video-AMME or online traffic |
| synthetic short text-to-speech | 4 | 74 chars / 12 words | audio_mean=4.3s | short-output thinker/talker/code2wav guard | lat_mean=1.768s, lat_p95=2.056s, RTF=0.4105, RTF_p95=0.4388, QPS=2.218 | talker_pct=97.4%, hop_p95=20.1ms, decode_p95=22.3ms | short/long speech guard remains faster than real time | synthetic evidence isolates speech output and does not replace full Video-AMME or online traffic |
| synthetic short text-to-speech | 8 | 74 chars / 12 words | audio_mean=4.3s | short-output thinker/talker/code2wav guard | lat_mean=2.638s, lat_p95=2.828s, RTF=0.6257, RTF_p95=0.7440, QPS=2.983 | talker_pct=83.6%, hop_p95=21.2ms, decode_p95=24.3ms | short/long speech guard remains faster than real time | synthetic evidence isolates speech output and does not replace full Video-AMME or online traffic |
| synthetic long text-to-speech | 1 | 944 chars / 139 words | audio_mean=51.9s | long-form Talker AR cadence and chunk cadence | lat_mean=9.168s, lat_p95=9.465s, RTF=0.1766, RTF_p95=0.1776, QPS=0.109 | talker_pct=99.2%, hop_p95=15.0ms, decode_p95=14.2ms | short/long speech guard remains faster than real time | synthetic evidence isolates speech output and does not replace full Video-AMME or online traffic |
| synthetic long text-to-speech | 4 | 944 chars / 139 words | audio_mean=52.6s | long-form Talker AR cadence and chunk cadence | lat_mean=17.551s, lat_p95=18.025s, RTF=0.3338, RTF_p95=0.3373, QPS=0.227 | talker_pct=99.5%, hop_p95=20.4ms, decode_p95=21.6ms | short/long speech guard remains faster than real time | synthetic evidence isolates speech output and does not replace full Video-AMME or online traffic |
| synthetic long text-to-speech | 8 | 944 chars / 139 words | audio_mean=52.3s | long-form Talker AR cadence and chunk cadence | lat_mean=25.799s, lat_p95=26.318s, RTF=0.4932, RTF_p95=0.5001, QPS=0.303 | talker_pct=99.1%, hop_p95=24.0ms, decode_p95=18.2ms | long c=8 remains faster than real time | synthetic evidence isolates speech output and does not replace full Video-AMME or online traffic |

## 3. 复现入口

- 形状证据：`jq '.summary' results/qwen35_report_audit_20260619/length_regime_coverage.json`。
- Video-AMME ci-50 分布：`jq '.target_chars, .audio_duration_s' results/qwen35_report_audit_20260619/videoamme_seedtts_meta_summary.json`。
- Synthetic short/long CSV：`column -s, -t results/qwen35_report_audit_20260619/share_charts/synthetic_short_long_speech.csv | head`。
- Stage budget：`jq '.synthetic_speech_budget' results/qwen35_report_audit_20260619/stage_latency_budget.json`。
- 复跑裁决：`jq '.rules[] | select(.id | test("synthetic_(short|long)"))' results/qwen35_report_audit_20260619/rerun_acceptance_contract.json`。

## 4. 不可越界说法

- 不说“synthetic long c=8 代表所有长文线上流量”。
- 不说“Video-AMME ci-50 已经覆盖完整线上流量”。
- 不说“短/长输入形状改变后仍可直接复用当前 headline 数字”。
- 任何替换 headline 的长短文数据都必须保持固定输入形状、同一镜像、同一硬件、同一 WER/ASR 口径，并重跑 full audit。
