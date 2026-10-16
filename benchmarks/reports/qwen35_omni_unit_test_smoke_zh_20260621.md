# Qwen3.5-Omni 单元测试 Smoke 回归证据

状态：附加回归证据，ready=`true`。
用途：证明本轮性能报告相关的 profiler、Qwen3.5-Omni helper、talker/subtalker、code2wav 单元路径没有明显回归。它不是性能 benchmark，也不替代 full audit。

## 1. 结论

| 项目 | 结果 |
| --- | --- |
| 运行环境 | `docker --gpus all frankleeeee/sglang-omni:dev` |
| 测试范围 | profiler views；Qwen3.5-Omni bootstrap/preflight/request builders/subtalker/talker；Qwen3-Omni code2wav |
| pytest 结果 | `235 passed, 2 warnings in 26.35s` |
| required failures | `0` |
| 机器证据 | `results/qwen35_report_audit_20260619/unit_test_smoke.json` |

## 2. 运行命令

```bash
docker run --gpus all --rm --entrypoint /bin/bash \
  -v /home/gangouyu/sglang-omni:/workspace \
  -w /workspace frankleeeee/sglang-omni:dev \
  -lc 'python3 -m pytest -q \
    tests/unit_test/profiler/test_views.py \
    tests/unit_test/qwen3_5_omni/test_bootstrap.py \
    tests/unit_test/qwen3_5_omni/test_preflight.py \
    tests/unit_test/qwen3_5_omni/test_request_builders.py \
    tests/unit_test/qwen3_5_omni/test_subtalker.py \
    tests/unit_test/qwen3_5_omni/test_talker.py \
    tests/unit_test/qwen3_omni/test_code2wav.py'
```

## 3. 覆盖边界

- 覆盖的是报告相关的轻量单元路径：profile view、启动/preflight、request builder、subtalker/talker helper、code2wav。
- 不覆盖真实性能 benchmark；性能结论仍以 full audit、raw benchmark artifacts、stage drilldown、metric provenance 和 share release seal 为准。
- 宿主 Python 当前没有安装 pytest 和核心项目依赖，因此宿主侧 unit test 不是可执行入口。
- 未预配依赖的全量 `tests/unit_test/qwen3_5_omni` collection 会被 `typer`、`msgpack`、`jiwer`、`av` 等 optional/import 依赖打断；这不作为性能报告 gate。
- 第一次未挂 GPU 的容器 collection 会触发 `sgl_kernel` 找不到 CUDA runtime；有效 smoke 使用 `docker --gpus all`。

## 4. 对报告可信度的作用

这份 smoke 只增强“当前性能报告工具链没有明显单元级回归”的证据：

- profiler 视图仍能解析和展示 stage 统计。
- Qwen3.5-Omni 的启动、preflight、request builder 和 talker/subtalker helper 单元路径通过。
- Qwen3-Omni code2wav 调度相关单元路径通过。

它不改变 headline：SGLang-vLLM 严格对比仍只引用 warmed c=4；c=8 仍是 SGLang 当前吞吐峰值；vLLM c=8 prebuild w4 仍是 offline diagnostic，不是 online serving parity。
