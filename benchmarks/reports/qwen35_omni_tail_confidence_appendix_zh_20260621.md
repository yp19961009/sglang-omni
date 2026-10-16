# Qwen3.5-Omni Tail Confidence Appendix

生成时间 UTC：`2026-06-21T02:00:22.992897+00:00`。
工作目录：`/home/gangouyu/sglang-omni`。

这页只使用已审计 raw `per_sample` 结果，不新增 benchmark。它补充 mean/p95 以外的
p50/p90/IQR/max 视角，帮助 reviewer 判断 headline、压力窗口和长短文结论是否只是偶然尾部。

## 1. Machine Gate

| Gate | Value |
| --- | ---: |
| Ready | `True` |
| Rows | `18` |
| Bootstrap comparisons | `9` |
| Bootstrap draws | `5000` |
| Checks | `13/13` |
| Required failures | `0` |

## 2. Distribution Rows

| Group | Case | n | Success | Accuracy | WER | QPS | Lat p50/p90/p95/max | RTF p50/p90/p95/max | IQR Lat / RTF |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| strict_c4 | SGLang warmed c=4 | 46 | 46/46 | 70.0% | 4.1% | 2.175 | 1.580/2.275/3.328/3.999s | 1.304/2.213/2.402/3.035 | 0.403s / 0.453 |
| strict_c4 | vLLM warmed c=4 | 46 | 46/46 | 66.0% | 7.4% | 0.154 | 1.692/2.558/3.525/6.861s | 1.478/2.296/3.072/3.689 | 0.836s / 0.888 |
| sglang_stress | SGLang Video-AMME c=1 | 50 | 50/50 | 70.0% | 3.8% | 0.760 | 1.179/1.809/2.406/2.883s | 0.959/1.476/2.020/2.333 | 0.092s / 0.443 |
| sglang_stress | SGLang Video-AMME c=2 | 50 | 50/50 | 70.0% | 3.8% | 1.315 | 1.270/2.100/3.124/3.392s | 1.065/1.669/1.931/2.111 | 0.234s / 0.397 |
| sglang_stress | SGLang Video-AMME c=4 | 50 | 50/50 | 70.0% | 3.8% | 2.036 | 1.673/2.742/3.633/4.881s | 1.397/2.166/2.498/2.721 | 0.489s / 0.634 |
| sglang_stress | SGLang Video-AMME c=8 | 50 | 50/50 | 70.0% | 3.2% | 2.540 | 2.746/5.046/5.853/7.014s | 1.889/3.992/4.392/6.132 | 0.978s / 1.112 |
| sglang_stress | SGLang Video-AMME c=16 | 50 | 50/50 | 70.0% | 2.9% | 2.407 | 6.262/7.213/7.846/10.218s | 4.557/8.102/10.409/12.460 | 1.110s / 2.594 |
| synthetic_speech | Synthetic short c=1 | 16 | 16/16 | n/a | n/a | 1.154 | 0.857/0.912/0.924/0.925s | 0.206/0.209/0.211/0.215 | 0.055s / 0.006 |
| synthetic_speech | Synthetic short c=4 | 16 | 16/16 | n/a | n/a | 2.218 | 1.736/1.986/2.056/2.265s | 0.418/0.432/0.439/0.451 | 0.161s / 0.023 |
| synthetic_speech | Synthetic short c=8 | 16 | 16/16 | n/a | n/a | 2.983 | 2.606/2.811/2.828/2.845s | 0.607/0.722/0.744/0.789 | 0.244s / 0.131 |
| synthetic_speech | Synthetic long c=1 | 8 | 8/8 | n/a | n/a | 0.109 | 9.127/9.444/9.465/9.486s | 0.177/0.177/0.178/0.178 | 0.258s / 0.001 |
| synthetic_speech | Synthetic long c=4 | 8 | 8/8 | n/a | n/a | 0.227 | 17.637/18.016/18.025/18.034s | 0.334/0.337/0.337/0.338 | 0.701s / 0.002 |
| synthetic_speech | Synthetic long c=8 | 8 | 8/8 | n/a | n/a | 0.303 | 26.031/26.262/26.318/26.375s | 0.494/0.500/0.500/0.500 | 0.686s / 0.006 |
| vllm_diagnostic | vLLM original c=1 | 46 | 46/46 | 66.0% | n/a | 0.127 | 1.640/2.573/3.545/5.452s | 1.428/2.019/2.547/4.483 | 0.875s / 0.736 |
| vllm_diagnostic | vLLM warmed c=4 | 46 | 46/46 | 66.0% | 7.4% | 0.154 | 1.692/2.558/3.525/6.861s | 1.478/2.296/3.072/3.689 | 0.836s / 0.888 |
| vllm_diagnostic | vLLM original c=8 | 42 | 42/42 | 66.0% | n/a | 0.162 | 1.657/2.493/3.260/4.908s | 1.451/2.565/3.199/4.131 | 0.855s / 0.723 |
| vllm_diagnostic | vLLM prebuild c=8 w1 | 42 | 42/42 | 66.0% | n/a | 0.539 | 4.514/6.497/7.009/8.658s | 3.353/5.875/6.258/10.239 | 2.335s / 3.065 |
| vllm_diagnostic | vLLM prebuild c=8 w4 | 42 | 42/42 | 66.0% | n/a | 0.536 | 4.569/7.124/7.730/8.674s | 3.932/5.943/7.087/9.157 | 2.622s / 2.787 |

## 3. Bootstrap Comparisons

| Comparison | Metric | Stat | Point | 95% CI | Decision | Interpretation |
| --- | --- | --- | ---: | ---: | --- | --- |
| Strict c=4 mean latency advantage | latency_s | mean_delta | 0.350s | 0.025s/0.696s | ci95_low_gt_0 | Bootstrap 95% CI excludes zero, so the strict c=4 latency mean advantage is a strong headline support. |
| Strict c=4 p95 latency advantage | latency_s | p95_delta | 0.197s | -1.164s/3.687s | point_estimate_gt_0_with_ci_overlap | Point estimate favors SGLang; the p95 CI overlaps zero, so this is supporting tail evidence, not a standalone significance claim. |
| Strict c=4 mean RTF advantage | rtf | mean_delta | 0.114 | -0.159/0.391 | point_estimate_gt_0_with_ci_overlap | Point estimate favors SGLang, but the bootstrap CI overlaps zero; report it as directional support with a caveat. |
| Strict c=4 p95 RTF advantage | rtf | p95_delta | 0.669 | -0.496/1.469 | point_estimate_gt_0_with_ci_overlap | Point estimate favors SGLang; CI overlap keeps this as tail support rather than an inferential RTF-only claim. |
| SGLang c16 vs c8 p95 latency penalty | latency_s | p95_delta | 1.993s | 0.384s/4.289s | ci95_low_gt_0 | Bootstrap CI excludes zero; this supports c16 as a saturation boundary, not the recommended serving point. |
| SGLang c16 vs c8 p95 RTF penalty | rtf | p95_delta | 6.016 | 2.539/7.896 | ci95_low_gt_0 | Bootstrap CI excludes zero; c16 increases tail RTF even though quality remains stable. |
| Synthetic long c8 p95 RTF real-time margin | rtf | p95 | 0.500 | 0.497/0.500 | ci95_high_lt_1 | Bootstrap upper bound remains below 1.0, so the long-text synthetic guardrail stays faster than real time. |
| vLLM prebuild w4 vs original c8 p95 latency penalty | latency_s | p95_delta | 4.470s | 2.684s/6.125s | ci95_low_gt_0 | Prebuild w4 improves throughput diagnostics but exposes a larger per-request latency tail. |
| vLLM prebuild w4 vs original c8 p95 RTF penalty | rtf | p95_delta | 3.888 | 1.758/6.514 | ci95_low_gt_0 | The RTF tail penalty also excludes zero; keep prebuild w4 as offline diagnostic evidence, not online parity. |

## 4. Interpretation

- Strict warmed c=4 的 SGLang latency/RTF mean 和 p95 同时优于优化版 vLLM，quality/WER 不退化。
- Bootstrap 显示 strict c=4 mean latency 优势的 95% CI 排除 0；RTF 的点估计也优，但 CI 有重叠，所以不要单独用 RTF 做显著性表述。
- SGLang c=8 是当前 throughput peak；c=16 的 tail 更高且 QPS 回落，因此只作为 saturation boundary。
- Synthetic long c=8 的 RTF p95 仍低于 1，说明长文本/长语音 guardrail 快于实时。
- vLLM c=8 prebuild w4 提高 runner/engine QPS，但 per-request tail 更高；这正是 offline diagnostic，不升级成 online parity。

## 5. Checks

| Status | Required | Check | Evidence |
| --- | --- | --- | --- |
| PASS | yes | all required per-sample artifacts are present | missing=[] |
| PASS | yes | strict warmed c4 SGLang tail beats vLLM | lat_mean/p95 SGLang=1.743/3.328s, vLLM=2.093/3.525s; rtf_mean/p95 SGLang=1.354/2.402, vLLM=1.468/3.072 |
| PASS | yes | strict warmed c4 quality does not regress | accuracy SGLang=70.0%, vLLM=66.0%; WER SGLang=4.1%, vLLM=7.4% |
| PASS | yes | SGLang stress keeps zero per-sample failures | failures=sglang_stress_c1=0, sglang_stress_c2=0, sglang_stress_c4=0, sglang_stress_c8=0, sglang_stress_c16=0 |
| PASS | yes | SGLang c8 remains the throughput peak and c16 remains saturation | c8_qps=2.540, c16_qps=2.407; c8_p95=5.853s, c16_p95=7.846s |
| PASS | yes | long synthetic c8 tail remains faster than real time | long_c8_rtf_p95=0.500, failures=0 |
| PASS | yes | vLLM c8 prebuild w4 improves throughput but exposes tail | original_c8_p95=3.260s, w4_p95=7.730s; original_qps=0.162, w4_qps=0.536 |
| PASS | yes | strict c4 latency mean bootstrap advantage excludes zero | advantage_ci95=0.025/0.696s |
| PASS | yes | strict c4 RTF bootstrap overlap is explicitly caveated | rtf_mean_ci95=-0.159/0.391; rtf_p95_ci95=-0.496/1.469 |
| PASS | yes | strict c4 latency p95 point advantage remains positive | latency_p95_advantage=0.197s; ci95=-1.164/3.687s |
| PASS | yes | SGLang c16 tail penalty bootstrap excludes zero | latency_ci95=0.384/4.289s; rtf_ci95=2.539/7.896 |
| PASS | yes | long synthetic c8 bootstrap p95 RTF upper bound remains real-time | long_c8_rtf_p95_ci95=0.497/0.500 |
| PASS | yes | vLLM prebuild w4 tail penalty bootstrap excludes zero | latency_ci95=2.684/6.125s; rtf_ci95=1.758/6.514 |
