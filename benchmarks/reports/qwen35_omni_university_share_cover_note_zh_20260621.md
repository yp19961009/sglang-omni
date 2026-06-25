# Qwen3.5-Omni 合作高校外发 Cover Note

状态：2026-06-21 evidence-ready 交付稿；更新后的目标不再等待 6.21 晚上。
工作目录：`/home/gangouyu/sglang-omni`。
用途：作为邮件、微信或飞书正文的最短可复制版本；技术细节以随包报告和机器证据为准。

## 1. 建议标题

Qwen3.5-Omni SGLang-Omni 性能分析分享包：stage breakdown、单/高并发、短/长文本、优化 vLLM baseline

## 2. 可直接发送正文

```text
各位老师/同学好，

这版是 Qwen3.5-Omni 在 SGLang-Omni 上的性能分析分享包，覆盖本地 8x H20 环境下的
Video-AMME ci-50 语音输出 workload、SGLang c=1/2/4/8/16 压测、短/长文本输入 + 语音输出、
vLLM optimized baseline、vLLM c=8 prebuild offline diagnostic，以及可直接放 PPT 的 SVG/CSV
图表包。

建议先打开 qwen35_omni_start_here_zh_20260621.md，用收包快检命令确认 checksum、receiver
quickcheck、full-audit summary 和 evidence-query smoke；随后按
qwen35_omni_share_package_index_zh_20260621.md 的阅读顺序看技术正文、stage breakdown、
runtime comparison contract 和复跑清单。
如果现场追问某个 stage，先打开
qwen35_omni_stage_reproduction_drilldown_zh_20260621.md 第 2 节的 5 条答辩 quick route；
对应机器入口是 `stage_reproduction_drilldown.json` 的 `quick_reproduction_map`，可以直接跳到
SGLang c=8/c=16 admission queue、长文本 c=8、vLLM original c=8 和 prebuild w4 诊断证据。

当前 high-confidence 结论是：在同硬件、同模型和 warmed c=4 严格对比下，优化后的
SGLang-Omni Qwen3.5 在 latency/RTF 上优于优化版 vLLM，并保留 accuracy/WER 验收链路。
SGLang c=8 是当前吞吐峰值；c=16 是饱和边界，不作为默认推荐。stage 连接整体健康，
主要瓶颈来自 talker AR cadence 与 high-concurrency admission/queue，而不是 code2wav decode
或未解释的 stage handoff。

边界请一起保留：vLLM baseline 不是弱基线，已锁定 Qwen3.5-capable 镜像、compile/CUDA graph、
prefix/chunked prefill、shared-memory transfer、encoder compile/batch 和 prebuild w4 证据；
vLLM c=8 prebuild w4 当前只作为 offline diagnostic，不是 online serving parity；official
SeedTTS full-set 不是本报告 headline；复跑后只有在同硬件、同镜像、同模型/cache、同 ASR/WER
链路且门禁全绿时，才允许替换 headline 数字。ci-50/stress/synthetic 证据不能直接外推到完整线上流量；
更大 Video-AMME 或真实线上流量需要同口径复跑、stage/tail/quality gate 和 caveat gate 全绿。
```

## 3. 附件或目录顺序

1. `results/qwen35_report_audit_20260619/qwen35_omni_share_bundle_20260621.tar.gz`
2. `results/qwen35_report_audit_20260619/qwen35_omni_share_bundle_20260621.tar.gz.sha256`
3. `benchmarks/reports/qwen35_omni_start_here_zh_20260621.md`
4. `benchmarks/reports/qwen35_omni_share_package_index_zh_20260621.md`
5. `benchmarks/reports/qwen35_omni_university_technical_report_zh_20260621.md`
6. `benchmarks/reports/qwen35_omni_one_page_scorecard_zh_20260621.md`
7. `benchmarks/reports/qwen35_omni_receiver_command_card_zh_20260621.md`
8. `benchmarks/reports/qwen35_omni_share_release_seal_zh_20260621.md`
9. `benchmarks/reports/qwen35_omni_final_completion_audit_zh_20260621.md`

`share_release_seal` 和 `final_completion_audit` 是 tarball 外侧伴随证据；tarball 内 README 会说明
这些文件为什么不作为 tar member。

## 4. 接收方第一条命令

```bash
export HOST_REPO="${HOST_REPO:-/home/gangouyu/sglang-omni}"
cd "$HOST_REPO"
bash benchmarks/eval/qwen35_omni_receiver_quickcheck.sh
bash benchmarks/eval/qwen35_omni_evidence_query_cards_smoke.sh --root "$HOST_REPO" --mode host
```

期望结果：checksum OK；tarball validation `17/17`；receiver smoke `17/17`；extracted-only
validation `13/13`；standalone validation `8/8`；receiver quickcheck contract `15/15`，
且包含 stage dictionary crosswalk；
evidence-query host smoke 通过。解包后可用 portable 模式复核同一套查询。
更新后的目标不再等待 6.21 晚上；`completion_allowed_now=true` 表示可以进入最终完成裁决。

## 5. 不要单独外发的说法

- 不说“vLLM c=8 online serving parity 已证明”。
- 不说“c=16 是推荐生产默认”。
- 不说“已经覆盖官方 SeedTTS full-set headline”。
- 不说“ci-50/stress/synthetic 证据已经覆盖完整线上流量”。
- 不在未完成同环境复跑和验收前改写 headline 数字。
- 不把 host `/root/.cache/whisper/large-v3.pt` 缺失写成 serving benchmark 失败；它是 WER/ASR 复跑链路的 optional warning。
