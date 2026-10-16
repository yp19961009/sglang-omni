# Qwen3.5-Omni Caveat 裁决矩阵

生成时间 UTC：`2026-06-21T02:00:21.850614+00:00`。
工作目录：`/home/gangouyu/sglang-omni`。

这页把 caveat 从“提醒”变成可执行裁决：哪些结论可以直接分享，哪些必须带边界，
哪些只有补跑新实验后才能升级成 headline，哪些说法应明确禁止。它不新增 benchmark
数字，只引用当前 audit JSON、confidence ledger、acceptance matrix 和 objective audit。

## 1. 裁决 Gate

| Gate | 当前值 | 证据 |
| --- | ---: | --- |
| final readiness contract | `ready=true, contract=true` | `final_readiness_audit.json` bootstrap contract |
| confidence ledger | `high=9, medium=3, unsupported=0` | `confidence_ledger.json` |
| objective caveats | `3 boundary items, 0 required failures` | `objective_completion_audit.json` |
| acceptance matrix | `17/17` | `acceptance_matrix.json` |
| preflight | `62 checks, 1 optional warning` | `preflight_repro.json` |
| manifest / repro | `196 records, 63 commands` | `manifest.json`; `repro_command_manifest.json` |
| SGLang best measured recipe | `26/26 checks` | `sglang_optimization_lock.json` |

## 2. Caveat 裁决表

| 主题 | 当前裁决 | 对外允许说法 | 不能说 | 升级/替换数字条件 | 证据 |
| --- | --- | --- | --- | --- | --- |
| strict c=4 SGLang vs vLLM | 可作为主 headline，带环境边界 | 可以作为主 headline：在 Video-AMME ci-50 warmed c=4 上，SGLang-Omni 至少与优化版 vLLM 相当，并在 latency/RTF 上更优。 | 把 c=4 结论外推到所有并发、所有 workload 或所有线上流量 | 硬件/image/model/data/ASR 口径变化时重跑 full audit；headline scorecard、claims、final readiness 全部通过 | high / PASS |
| SGLang c=8 峰值 / c=16 饱和 | 可分享；c=16 必须说成压力边界 | 可以说 c=8 是当前高并发 sweet spot，c=16 用来证明 admission/queueing 饱和。 | 把 c=16 当默认服务点，或把 c=8 峰值外推到新 admission 策略 | 若调整 admission、preprocessing placement 或 thinker memory fraction，必须重跑 stress+stage audit | high / PASS；c16 evidence=PASS |
| 更大数据 / 真实流量外推 | 当前 ci-50/stress/synthetic 证据支持本报告范围；不能直接外推到全量线上流量 | 可以说 ci-50/stress/synthetic 证据充分支持当前报告范围。 | 把 ci-50、stress 或 synthetic 证据表述成 full online traffic coverage | 补更大 Video-AMME/full-traffic 样本，重跑 SGLang/vLLM、stage、tail、caveat 和 final readiness 后才扩大结论范围 | medium / PASS |
| 当前 best measured recipe | 可说是当前审计环境里的最优 recipe，不能说全局数学最优 | 当前 best measured recipe 是 compiled/graph path、serial preprocessing、16GiB preprocessing cache，并以 c=4-c=8 作为服务窗口 | 说已经搜索完所有未来 kernel、placement、admission policy，或承诺任何环境下全局最优 | 任何新 recipe 只有在 c=4/c=8/c=16、WER、stage interaction、acceptance、final readiness 全部通过后才能替换当前 recipe | `sglang_optimization_lock.json` 26/26；anti-recipe rows=PASS/PASS |
| short/long text-to-speech | 可作为语音输出 guardrail | 短文本 12 words、长文本 139 words 已覆盖，long c=8 快于实时 | 用 synthetic speech 替代 Video-AMME 或 official SeedTTS headline | 若要 natural-speech headline，补 official SeedTTS/full natural speech benchmark 并更新 claims | `tables_summary.json`; `acceptance_matrix.json` |
| official SeedTTS full-set | 只能说未作为 headline；可说已有 smoke path | 可以说已提供 SeedTTS-compatible smoke meta，官方 full-set 需要预置数据后再跑。 | 把 Video-AMME spoken-reference smoke 说成官方 SeedTTS 完整集 | 预置 official SeedTTS 数据，跑完整 benchmark，补 WER/RTF/claims/coverage/final readiness | medium / PASS |
| vLLM original c=8 | 只能做 prompt-feed/admission 诊断 | original c=8 受 host prompt build/feed admission 限制 | 用 original offline c=8 wall QPS 证明 SGLang online parity 或 non-parity | 若要 c=8 横向 headline，先建立 online ingress 同口径服务路径 | `vllm_admission_diagnosis.json` |
| vLLM c=8 prebuild w4 | 当前最强 vLLM offline diagnostic，但不是 online parity | 可以把 prebuild w4 作为优化过的 offline diagnostic baseline。 | 不要宣称已经完成严格 c=8 在线 serving parity。 | 补 online ingress、同口径 WER/ASR、engine/talker boundary 复核，并更新 runtime contract | high / PASS；parity=medium / PASS |
| optional Whisper host cache warning | 不阻塞当前分享包；只影响 host 侧直接重算 WER | 当前 WER 证据可引用；host 重算 WER 需要 Whisper 权重或 ASR router | 把 optional warning 写成 required failure，或在缺 ASR 时替换 WER 数字 | 若要让外部 host 直接重算 WER，提供 cache、容器内权重或 ASR router | high / PASS；preflight warnings=1 |
| preproc=2/4 | 当前 recipe 的 anti-recipe | 可以明确把 preproc=2/4 列为当前 recipe 的 anti-recipe。 | 把 preproc=2 或 preproc=4 当作当前最优配置 | 只有 admission/placement/memory 同步重设计后才能重新评估；新结果必须通过 acceptance/stage/final readiness | high / PASS；preproc2=PASS；preproc4=PASS |
| stage handoff / code2wav | 可说连接健康、decode 非主瓶颈 | 可以说 talker->code2wav hop 和 code2wav decode 都有明确非瓶颈证据。 | 把 code2wav 或 stage 连接说成当前主瓶颈 | 若改 talker chunk policy、streaming path 或 code2wav compile 路径，重跑 stage interaction 和 tail appendix | high / PASS |
| final evidence completion gate | 更新后的目标不再等待 2026-06-21 晚间；完成由 final_completion_audit 的证据门裁决 | 可以说 final_completion_audit `ready=true` 且 `completion_allowed_now=true` 后可标记完成 | 在 final_completion_audit 有 required failure、blocker 或 `completion_allowed_now=false` 时标记完成 | 任一 evidence/report/package/caveat 变化后重跑 full audit、final completion audit、package validation 和 release seal | PASS |

## 3. 替换数字的触发器

| 触发器 | 是否可直接替换报告数字 | 必须先满足 |
| --- | --- | --- |
| 只重跑 full audit，无 benchmark 变化 | 可以更新 hash/manifest/gate 数字 | full audit `ok=true`，旧值扫描干净 |
| 硬件、Docker image、模型权重、数据 cache、ASR 口径变化 | 不可直接替换 | environment snapshot、claims、coverage、preflight、manifest、final readiness 全部重跑通过 |
| 新 SGLang stress 或 synthetic speech 结果 | 不可直接替换 | acceptance matrix 仍 17/17 或同步扩展；stage interaction、headline scorecard、confidence ledger 重新通过 |
| 新大样本或真实流量结果 | 可进入扩展范围评审，不能直接替换当前 headline | 新样本的环境、数据、ASR/WER、stage/tail、claims、confidence、caveat 和 final readiness 全部接入 audit |
| 新 vLLM c=8 结果 | 默认只进入 diagnostic | runtime comparison contract 明确 online/offline 边界；若要 headline 必须有 online ingress + WER/ASR |
| 新 official SeedTTS/full natural-speech 结果 | 可作为新增 evidence，不能自动替换 Video-AMME headline | 数据集、WER/RTF、stage breakdown、coverage、confidence ledger 全部接入 audit |
| 任一 required gate 失败 | 不可分享为最终包 | 修复失败项后重跑 full audit；final readiness 必须 `ready=true` |

## 4. Reviewer 快速口径

- 主 headline 只用 warmed c=4 strict runtime comparison。
- 高并发只在 SGLang 内部说 c=4-c=8 推荐窗口、c=8 峰值、c=16 饱和边界。
- 更大 Video-AMME 或真实流量结果只能在补齐同口径 stage/tail/quality gate 后扩大结论范围。
- vLLM c=8 只作为 offline diagnostic，prebuild w4 是优化后的 diagnostic baseline。
- Stage 连接和 code2wav decode 当前不是主瓶颈；优先讨论 admission/queue 与 talker AR。
- SeedTTS full-set、online c=8 parity、host-side WER 重算都属于升级项，不影响当前 share package 的 headline。
