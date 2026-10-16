# Qwen3.5-Omni C12 Audio Validation

生成时间：2026-06-30 21:58 CST

## 结论

- 本地稳定最优音频：12 个样本。
- 可被 Python `wave` 标准库解析：12/12。
- channels：1。
- sample_rate：24000 Hz。
- sample_width：2 bytes。
- wave duration 与 result.json `audio_duration_s` 最大差值：0.0 ms。

## 明细

| sample | valid | channels | sample_rate | duration_wave | duration_result | delta_ms |
|---|---:|---:|---:|---:|---:|---:|
| `sample_00` | True | 1 | 24000 | 9.680 | 9.680 | 0.0 |
| `sample_01` | True | 1 | 24000 | 9.040 | 9.040 | 0.0 |
| `sample_02` | True | 1 | 24000 | 8.800 | 8.800 | 0.0 |
| `sample_03` | True | 1 | 24000 | 8.880 | 8.880 | 0.0 |
| `sample_04` | True | 1 | 24000 | 9.360 | 9.360 | 0.0 |
| `sample_05` | True | 1 | 24000 | 10.320 | 10.320 | 0.0 |
| `sample_06` | True | 1 | 24000 | 10.560 | 10.560 | 0.0 |
| `sample_07` | True | 1 | 24000 | 8.400 | 8.400 | 0.0 |
| `sample_08` | True | 1 | 24000 | 9.200 | 9.200 | 0.0 |
| `sample_09` | True | 1 | 24000 | 8.320 | 8.320 | 0.0 |
| `sample_10` | True | 1 | 24000 | 8.400 | 8.400 | 0.0 |
| `sample_11` | True | 1 | 24000 | 6.640 | 6.640 | 0.0 |

CSV：`/Users/gangouyu/Documents/gangouyu-ecs-ssh/reports/sglang_omni_qwen35_c12_audio_validation_20260630.csv`
