# Qwen3.5-Omni Chart Source Consistency

用途：校验分享包里的 PPT CSV/SVG 图表是否仍然是从审计 JSON 重新生成的版本，
避免 slide 数字被手工改动后继续对外分享。

## 1. Summary

| Gate | Value |
| --- | ---: |
| Ready | `True` |
| Checks | `8/8` |
| Required failures | `0` |
| CSV files checked | `7` |
| SVG files checked | `7` |
| Byte-exact files | `14` |

## 2. Checks

| Status | Required | Check | Evidence |
| --- | --- | --- | --- |
| PASS | yes | source audit JSONs exist | sources=5/5 |
| PASS | yes | chart pack manifest ready | chart_summary={'ready': True, 'csv_files': 7, 'svg_files': 7, 'generated_files': 14, 'checks': {'strict_c4_csv': True, 'pressure_csv': True, 'stage_csv': True, 'stage_budget_csv': True, 'stage_budget_svg': True, 'vllm_csv': True, 'enough_svg': True, 'enough_csv': True}} |
| PASS | yes | expected chart regeneration ready | expected_summary={'ready': True, 'csv_files': 7, 'svg_files': 7, 'generated_files': 14, 'checks': {'strict_c4_csv': True, 'pressure_csv': True, 'stage_csv': True, 'stage_budget_csv': True, 'stage_budget_svg': True, 'vllm_csv': True, 'enough_svg': True, 'enough_csv': True}} |
| PASS | yes | chart file set matches regenerated source | actual=['preprocessing_antirecipe.csv', 'sglang_handoff_decode_ms.svg', 'sglang_pressure_latency.svg', 'sglang_pressure_qps.svg', 'sglang_pressure_sweep.csv', 'sglang_stage_latency_budget.csv', 'sglang_stage_latency_budget_pct.svg', 'stage_connection_health.csv', 'strict_c4_latency_rtf.svg', 'strict_c4_runtime_comparison.csv', 'synthetic_short_long_rtf.svg', 'synthetic_short_long_speech.csv', 'vllm_admission_diagnosis.csv', 'vllm_c8_diagnostic_qps.svg']; expected=['preprocessing_antirecipe.csv', 'sglang_handoff_decode_ms.svg', 'sglang_pressure_latency.svg', 'sglang_pressure_qps.svg', 'sglang_pressure_sweep.csv', 'sglang_stage_latency_budget.csv', 'sglang_stage_latency_budget_pct.svg', 'stage_connection_health.csv', 'strict_c4_latency_rtf.svg', 'strict_c4_runtime_comparison.csv', 'synthetic_short_long_rtf.svg', 'synthetic_short_long_speech.csv', 'vllm_admission_diagnosis.csv', 'vllm_c8_diagnostic_qps.svg'] |
| PASS | yes | all chart assets byte-exact with regenerated source | checked=14, csv_exact=7, svg_exact=7 |
| PASS | yes | all chart manifest metadata matches regenerated source | metadata_matches=14/14 |
| PASS | yes | all CSV assets parse with manifest row counts | csv_parseable=7/7 |
| PASS | yes | all SVG assets have valid svg envelope | svg_parseable=7/7 |

## 3. Chart Files

| File | Kind | Byte-exact | Parseable | Size | Shape |
| --- | --- | --- | --- | ---: | --- |
| preprocessing_antirecipe.csv | csv | True | True | 210 | 9 columns / 2 rows |
| sglang_handoff_decode_ms.svg | svg | True | True | 3764 |  |
| sglang_pressure_latency.svg | svg | True | True | 3745 |  |
| sglang_pressure_qps.svg | svg | True | True | 2848 |  |
| sglang_pressure_sweep.csv | csv | True | True | 1989 | 17 columns / 5 rows |
| sglang_stage_latency_budget.csv | csv | True | True | 1031 | 17 columns / 5 rows |
| sglang_stage_latency_budget_pct.svg | svg | True | True | 5537 |  |
| stage_connection_health.csv | csv | True | True | 5270 | 11 columns / 37 rows |
| strict_c4_latency_rtf.svg | svg | True | True | 3332 |  |
| strict_c4_runtime_comparison.csv | csv | True | True | 614 | 10 columns / 2 rows |
| synthetic_short_long_rtf.svg | svg | True | True | 3078 |  |
| synthetic_short_long_speech.csv | csv | True | True | 2035 | 17 columns / 6 rows |
| vllm_admission_diagnosis.csv | csv | True | True | 1627 | 15 columns / 4 rows |
| vllm_c8_diagnostic_qps.svg | svg | True | True | 2993 |  |

## 4. 使用边界

- 这个 gate 证明图表文件与当前审计 JSON 的生成结果 byte-exact 一致。
- 如果合作方改了 PPT 数字或 CSV/SVG，必须重跑 chart pack 和本一致性检查。
- 它不替代 benchmark 复跑；benchmark 复跑仍以 `repro_command_manifest.json` 和 full audit 为准。
