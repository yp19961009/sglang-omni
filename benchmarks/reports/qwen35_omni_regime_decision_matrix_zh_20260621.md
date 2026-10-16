# Qwen3.5-Omni Regime 决策矩阵

生成时间 UTC：`2026-06-21T02:00:21.305628+00:00`。
工作目录：`/home/gangouyu/sglang-omni`。

这页给 reviewer 快速回答三个问题：哪个压力条件推荐、哪个只是边界或诊断、下一步优化该动哪里。
所有行均来自 `acceptance_matrix.json`，stage 判断交叉引用 `stage_interaction_summary.json`、
`headline_scorecard.json` 和 `vllm_admission_diagnosis.json`。

## 1. 总结 Gate

- ready：`True`
- acceptance rows：`17/17` accepted
- checks：`9/9`
- required failures：`0`
- final readiness：`True`
- stage handoff healthy：`True`
- code2wav decode not bottleneck：`True`
- vLLM diagnostic rows：`4`
- machine evidence：`results/qwen35_report_audit_20260619/regime_decision_matrix.json`

## 2. 推荐结论

- 对外 headline 使用 warmed c=4：SGLang latency/RTF 优于优化版 vLLM，accuracy/WER 不退化。
- 当前 SGLang 推荐服务窗口是 c=4 到 c=8；c=8 是吞吐峰值，c=16 是压力边界。
- 短/长文本语音输出用于守住 thinker/talker/code2wav 路径，长文本 c=8 仍快于实时。
- vLLM c=8 prebuild w4 是优化后的 offline diagnostic，不是 online serving parity。
- 不要把 preproc=2/4 当作当前优化方向；证据显示回退或失败。

## 3. Regime 决策表

| Regime | Pressure | 决策 | Stage/瓶颈判断 | 关键数字 | 推荐动作 |
| --- | --- | --- | --- | --- | --- |
| strict_runtime_comparison | Video-AMME ci-50 warmed c=4 | 严格横向 headline | 用 warmed c=4 做严格 SGLang-vLLM 对比；不要把 c=8 offline 诊断混进 headline。 | SGLang lat=1.743/3.328s, RTF=1.3536/2.4023, acc=67.4%, WER=4.12%; vLLM lat=2.093/3.525s, RTF=1.4677/3.0717, acc=63.0%, WER=7.44% | Use this as the main cross-runtime headline. |
| sglang_videoamme_stress | c=1 | 推荐服务窗口 | 低/中并发主 tail 是 talker_ar；stage handoff 和 code2wav decode 不是主瓶颈。 | acc=70.0%, WER=3.85%, lat_mean=1.316s, lat_p95=2.406s, QPS=0.760, hop_p95=15.5ms, decode_p95=17.8ms | Use for steady-state quality/latency shape. |
| sglang_videoamme_stress | c=2 | 推荐服务窗口 | 低/中并发主 tail 是 talker_ar；stage handoff 和 code2wav decode 不是主瓶颈。 | acc=70.0%, WER=3.85%, lat_mean=1.508s, lat_p95=3.124s, QPS=1.315, hop_p95=16.1ms, decode_p95=20.2ms | Use for steady-state quality/latency shape. |
| sglang_videoamme_stress | c=4 | 推荐服务窗口 | 低/中并发主 tail 是 talker_ar；stage handoff 和 code2wav decode 不是主瓶颈。 | acc=70.0%, WER=3.85%, lat_mean=1.929s, lat_p95=3.633s, QPS=2.036, hop_p95=17.8ms, decode_p95=21.8ms | Use for steady-state quality/latency shape. |
| sglang_videoamme_stress | c=8 | 推荐吞吐峰值 | 当前高并发甜点；admission/queue 开始显性化，但 talker->code2wav hop 仍健康。 | acc=70.0%, WER=3.23%, lat_mean=3.064s, lat_p95=5.853s, QPS=2.540, hop_p95=20.4ms, decode_p95=25.9ms | Use as the current high-concurrency sweet spot. |
| sglang_videoamme_stress | c=16 | 压力边界 | 吞吐回落且 queue/admission 饱和；只作为压力边界，不做默认服务点。 | acc=70.0%, WER=2.88%, lat_mean=6.066s, lat_p95=7.846s, QPS=2.407, hop_p95=19.7ms, decode_p95=23.9ms | Do not present c=16 as a recommended serving point. |
| sglang_synthetic_speech | short c=1 | 语音输出回归保护 | 短文本语音用于验证 thinker/talker/code2wav 输出路径；code2wav 边界保持小。 | text_words=12, audio=4.2s, lat_mean=0.866s, RTF=0.2052, QPS=1.154, hop_p95=14.9ms, decode_p95=14.2ms | Use as the short/long text-input and speech-output guardrail. |
| sglang_synthetic_speech | short c=4 | 语音输出回归保护 | 短文本语音用于验证 thinker/talker/code2wav 输出路径；code2wav 边界保持小。 | text_words=12, audio=4.3s, lat_mean=1.768s, RTF=0.4105, QPS=2.218, hop_p95=20.1ms, decode_p95=22.3ms | Use as the short/long text-input and speech-output guardrail. |
| sglang_synthetic_speech | short c=8 | 语音输出回归保护 | 短文本语音用于验证 thinker/talker/code2wav 输出路径；code2wav 边界保持小。 | text_words=12, audio=4.3s, lat_mean=2.638s, RTF=0.6257, QPS=2.983, hop_p95=21.2ms, decode_p95=24.3ms | Use as the short/long text-input and speech-output guardrail. |
| sglang_synthetic_speech | long c=1 | 语音输出回归保护 | 长文本/长语音主要压 talker AR 和 chunk cadence；long c=8 仍快于实时。 | text_words=139, audio=51.9s, lat_mean=9.168s, RTF=0.1766, QPS=0.109, hop_p95=15.0ms, decode_p95=14.2ms | Use as the short/long text-input and speech-output guardrail. |
| sglang_synthetic_speech | long c=4 | 语音输出回归保护 | 长文本/长语音主要压 talker AR 和 chunk cadence；long c=8 仍快于实时。 | text_words=139, audio=52.6s, lat_mean=17.551s, RTF=0.3338, QPS=0.227, hop_p95=20.4ms, decode_p95=21.6ms | Use as the short/long text-input and speech-output guardrail. |
| sglang_synthetic_speech | long c=8 | 语音输出回归保护 | 长文本/长语音主要压 talker AR 和 chunk cadence；long c=8 仍快于实时。 | text_words=139, audio=52.3s, lat_mean=25.799s, RTF=0.4932, QPS=0.303, hop_p95=24.0ms, decode_p95=18.2ms | Use as the short/long text-input and speech-output guardrail. |
| vllm_offline_diagnostic | original c=8 | 诊断证据 | 原始 vLLM c=8 主要是 host prompt build/feed admission 受限，不是 online parity。 | runner_overhead=81.8%, admission_avg=33314.0ms, engine_QPS=0.1622 | Use only as diagnosis; do not use as strict online parity evidence. |
| vllm_offline_diagnostic | prebuild c=8 workers=4 | 优化后 offline 诊断 | prebuild w4 是当前最强 vLLM offline 诊断；仍需 online ingress + WER 才能做 c=8 parity。 | prompt_wall=129.2s, runner_QPS=0.2127, engine_QPS=0.5360, admission_avg=4089.0ms | Use as optimized offline diagnostic; require online ingress plus WER before c=8 parity claims. |
| negative_optimization | PREPROCESSING_MAX_CONCURRENCY=2 at c=8 | 反例：性能回退 | preproc=2 把 admission 问题转成共享资源 contention，不能作为当前优化方向。 | baseline_QPS=2.540, preproc2_QPS=1.642, baseline_lat=3.064s, preproc2_lat=4.579s | Keep preprocessing concurrency at 1 unless placement/admission is redesigned. |
| negative_optimization | PREPROCESSING_MAX_CONCURRENCY=4 at c=8 | 反例：失败/OOM | preproc=4 有失败/OOM 风险，当前 recipe 禁用。 | failed=7, accuracy=60.0% | Do not use preproc=4 in the current recipe. |
| stage_connection_health | all audited boundaries | stage 连接防线 | 把 admission、stage-local compute、handoff 分开回答；不要把健康 handoff 说成瓶颈。 | sglang_talker_to_code2wav_healthy=True, sglang_code2wav_decode_not_bottleneck=True, vllm_original_c8_prompt_feed_limited=True, preprocessing_parallelism_regresses=True | Use this row to answer whether stage boundaries themselves are the bottleneck. |

## 4. Status 分布

| Status | Count | 读法 |
| --- | ---: | --- |
| `anti_recipe_failure` | 1 | 失败/OOM，应避免。 |
| `anti_recipe_regression` | 1 | 实测回退，应避免。 |
| `cross_stage_guardrail` | 1 | 用于回答 stage 连接是否卡住。 |
| `diagnostic_prompt_feed_limited` | 1 | 只能用于定位，不可当 serving parity。 |
| `not_recommended_saturation` | 1 | 可运行但不建议默认使用。 |
| `optimized_offline_diagnostic` | 1 | 优化后的诊断基线，仍需在线复核。 |
| `recommended_peak_throughput` | 1 | 当前 recipe 的高并发吞吐峰值。 |
| `recommended_serving_window` | 3 | 可作为当前推荐服务窗口的一部分。 |
| `recommended_strict_baseline` | 1 | 可做主 headline 的严格横向对比。 |
| `speech_generation_regression_guard` | 6 | 证明短/长文本语音输出路径没有退化。 |

## 5. 复核入口

- 主报告：`benchmarks/reports/qwen35_omni_stress_performance_plan_20260621.md`
- 压力矩阵：`benchmarks/reports/qwen35_omni_pressure_matrix_zh_20260621.md`
- stage metric dictionary：`benchmarks/reports/qwen35_omni_stage_metric_dictionary_zh_20260621.md`
- acceptance matrix：`results/qwen35_report_audit_20260619/acceptance_matrix.json`
- stage interaction summary：`results/qwen35_report_audit_20260619/stage_interaction_summary.json`
- regime decision matrix JSON：`results/qwen35_report_audit_20260619/regime_decision_matrix.json`
