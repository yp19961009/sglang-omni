# Qwen3.5-Omni Stage Latency Budget

生成时间 UTC：`2026-06-21T02:00:21.943865+00:00`。
工作目录：`/home/gangouyu/sglang-omni`。

这份附录把端到端 latency 和关键 stage span 放在同一张预算表里，便于在分享现场回答：
preprocessing、talker AR、code2wav、stage handoff、vLLM offline admission 到底谁在限制性能。

> 注意：这些百分比是 non-additive pressure ratio，不是 flame graph。
> 对 streaming/repeated stage，多个 span 可能重叠或按 window 重复；本表用于瓶颈归因，
> 不要求各列相加为 100%。

## 1. Gate

| Gate | Value |
| --- | ---: |
| Ready | True |
| Checks | 12/12 |
| Required failures | 0 |
| SGLang rows | 5 |
| Synthetic rows | 6 |
| vLLM rows | 4 |

## 2. SGLang Video-AMME Stage Budget

| c | QPS | Lat mean | Preproc lifecycle | Actual preproc | Est. queue | Talker AR | Code2wav decode | Hop p95 | Diagnosis |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 0.760 | 1316.0ms | 294.8ms (22.4%) | n/a (n/a) | n/a (n/a) | 444.1ms (33.7%) | 14.3ms (1.09%) | 15.5ms | talker_ar_tail |
| 2 | 1.315 | 1508.0ms | 334.1ms (22.2%) | n/a (n/a) | n/a (n/a) | 525.8ms (34.9%) | 16.0ms (1.06%) | 16.1ms | talker_ar_tail |
| 4 | 2.036 | 1929.0ms | 486.9ms (25.2%) | 295.8ms (15.3%) | 191.1ms (9.9%) | 663.1ms (34.4%) | 16.0ms (0.83%) | 17.8ms | talker_ar_tail |
| 8 | 2.540 | 3064.0ms | 1226.6ms (40.0%) | 289.2ms (9.4%) | 937.4ms (30.6%) | 982.7ms (32.1%) | 17.2ms (0.56%) | 20.4ms | admission_queue_plus_talker_tail |
| 16 | 2.407 | 6066.0ms | 4395.1ms (72.5%) | 304.6ms (5.0%) | 4090.5ms (67.4%) | 815.6ms (13.4%) | 16.7ms (0.28%) | 19.7ms | saturation_boundary |

关键读法：c=8/c=16 的 preprocessing lifecycle 变大，但 actual preprocess 仍约 0.29-0.30s；
放大的是 queue/admission。code2wav decode 仍是十几毫秒/window，和端到端 latency 不是一个量级。

## 3. Synthetic Short/Long Speech Budget

| Scenario | c | Words | Audio mean | Lat mean | RTF | Talker AR | Code2wav decode | Hop p95 | Diagnosis |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| long | 1 | 139 | 51.92s | 9168.0ms | 0.1766 | 9091.2ms (99.2%) | 12.7ms (0.14%) | 15.0ms | faster_than_realtime |
| long | 4 | 139 | 52.58s | 17551.0ms | 0.3338 | 17463.9ms (99.5%) | 14.0ms (0.08%) | 20.4ms | faster_than_realtime |
| long | 8 | 139 | 52.33s | 25799.0ms | 0.4932 | 25572.3ms (99.1%) | 14.5ms (0.06%) | 24.0ms | long_speech_talker_ar_dominant_but_faster_than_realtime |
| short | 1 | 12 | 4.22s | 866.0ms | 0.2052 | 849.0ms (98.0%) | 12.6ms (1.45%) | 14.9ms | faster_than_realtime |
| short | 4 | 12 | 4.31s | 1768.0ms | 0.4105 | 1722.6ms (97.4%) | 16.5ms (0.93%) | 20.1ms | faster_than_realtime |
| short | 8 | 12 | 4.27s | 2638.0ms | 0.6257 | 2205.9ms (83.6%) | 15.3ms (0.58%) | 21.2ms | faster_than_realtime |

关键读法：长文本长语音 c=8 的 talker AR 自然变长，但 RTF 仍低于 1；
这支持“长短文都有覆盖，长输出仍可快于实时”的结论。

## 4. vLLM Offline Budget / Admission Diagnosis

| Case | c | Runner QPS | Engine QPS | Runner overhead | Admission span avg/p95 | Encoder p95 | Thinker->Talker p95 | Talker->C2W p95 | Scope | Diagnosis |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| vLLM-c4 | 4 | 0.1536 | 0.1536 | 76.7% | 15110.8/19135.8ms | 44.2ms | 1.0ms | 17.5ms | strict_c4_only | prompt_feed_limited |
| vLLM-c8 | 8 | 0.1622 | 0.1622 | 81.8% | 33314.0/43972.7ms | 43.8ms | 1.0ms | 16.0ms | offline_diagnostic_only | prompt_feed_limited |
| vLLM-c8-prebuild-w1 | 8 | 0.1420 | 0.5391 | 77.6% | 4439.7/5425.0ms | 41.3ms | 4.0ms | 88.7ms | offline_diagnostic_only | engine_or_workload_limited |
| vLLM-c8-prebuild-w4 | 8 | 0.2127 | 0.5360 | 65.6% | 4089.0/4891.5ms | 46.2ms | 3.9ms | 123.4ms | offline_diagnostic_only | engine_or_workload_limited |

关键读法：vLLM original c=8 的 runner overhead 和 batch admission span 表明它是 offline prompt-feed/admission limited；
prebuild w4 缓解 prompt/admission，但仍是 offline diagnostic，不是 online parity。

## 5. Machine Checks

| Status | Required | Check | Evidence |
| --- | --- | --- | --- |
| PASS | yes | SGLang Video-AMME budget rows | rows=5 |
| PASS | yes | synthetic short/long budget rows | rows=6 |
| PASS | yes | vLLM diagnostic budget rows | rows=4 |
| PASS | yes | c8 queue pressure separated from actual preprocessing | c8_queue_pct=30.594223237597905, c8_actual_pct=9.438707571801567 |
| PASS | yes | c16 saturation visible in queue budget | c16_diagnosis=saturation_boundary, c16_queue_pct=67.43293768545993 |
| PASS | yes | code2wav decode remains small | all SGLang and synthetic code2wav decode averages <=30ms/window |
| PASS | yes | talker-to-code2wav hop remains small | all SGLang and synthetic talker->code2wav hop p95 <=30ms |
| PASS | yes | long c8 remains faster than real time | long_c8_rtf_mean=0.4932 |
| PASS | yes | vLLM c8 remains prompt-feed limited | diagnosis=prompt_feed_limited, runner_overhead_pct=81.77382174832061 |
| PASS | yes | vLLM prebuild w4 remains diagnostic-only | scope=offline_diagnostic_only, diagnosis=engine_or_workload_limited |
| PASS | yes | stage interaction booleans agree with budget | stage_interactions={'total_interactions': 37, 'status_counts': {'healthy': 28, 'queue_limited': 2, 'contention_regression': 1, 'prompt_feed_limited': 2, 'bottleneck': 1, 'diagnostic_only': 2, 'watch': 1}, 'sglang_talker_to_code2wav_healthy': True, 'sglang_code2wav_decode_not_bottleneck': True, 'vllm_original_c8_prompt_feed_limited': True, 'preprocessing_parallelism_regresses': True} |
| PASS | yes | claim verifier agrees with budget | claims_passed=True, failed=0 |
