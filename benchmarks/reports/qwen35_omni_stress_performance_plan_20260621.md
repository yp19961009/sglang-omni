# Qwen3.5-Omni SGLang-Omni Stress Performance Report

Status: evidence-ready share report; the updated objective no longer waits for the 2026-06-21 evening checkpoint.
Owner: SGLang-Omni Qwen3.5 bring-up/performance investigation.
Model: `qwen3_5_omni_23b_final_multilingual_all_voice_bf16_0315`.

This document is intended to become the shareable Qwen3.5-Omni performance
report for external university collaborators. It records the current optimized
SGLang-Omni recipe, vLLM comparison points, stress results across concurrency
levels, stage-level bottlenecks, inter-stage effects, anti-recipes, and exact
reproduction commands.

## 1. Executive Summary

As of the 2026-06-20 checkpoint, the optimized SGLang-Omni Qwen3.5 stack is
performance-aligned with the optimized vLLM baseline on the local Video-AMME
ci-50 speech-output workload. On the warmed steady-state c=4 slice, SGLang-Omni
is faster than vLLM on latency mean, latency p95, RTF mean, and RTF p95, while
also preserving comparable or better text accuracy and speech consistency.

| Runtime | Scope | n | Accuracy | Latency Mean | Latency P95 | RTF Mean | RTF P95 | WER Corpus |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vLLM optimized | Video-AMME ci-50, skip first 4, c=4 | 46 | 63.0% | 2.093s | 3.525s | 1.4677 | 3.0717 | 7.44% |
| SGLang-Omni optimized | Video-AMME ci-50, skip first 4, c=4 | 46 | 67.4% | 1.743s | 3.328s | 1.3536 | 2.4023 | 4.12% |

Relative to vLLM on this warmed c=4 comparison, SGLang-Omni is 16.7% lower on
mean latency, 5.6% lower on p95 latency, 7.8% lower on mean RTF, and 21.8%
lower on p95 RTF. This remains the strongest apples-to-apples cross-runtime
comparison point.

Additional vLLM c=1 and c=8 checkpoints were added on 2026-06-19 with the same
optimized vLLM image and `max_num_seqs=8`. They confirm two useful facts:
SGLang-Omni is clearly ahead of vLLM at single concurrency on request latency
and RTF, and vLLM can run c=8 without OOM. The vLLM offline runner builds
prompts by locally decoding/sampling videos before `omni.add_request()`, so its
per-request latency is engine-side after prompt construction while wall QPS
includes local prompt construction. This is why vLLM c=8 warmed request latency
looks healthy but wall throughput stays low.

A 2026-06-20 c=8 vLLM prebuilt-prompt diagnostic separates runner wall time,
prompt-build time, and engine wall time. The single-worker prebuild run completed
50/50 requests with 66.0% accuracy and reduced warmed batch-admission span from
33.3s avg / 44.0s p95 to 4.44s avg / 5.43s p95. A follow-up
`--prebuild-workers 4` run also completed 50/50 with the same 66.0% accuracy and
cut prompt-build wall from 249.3s to 129.2s, runner wall from 352.1s to 235.1s,
and runner QPS from 0.1420 to 0.2127. That confirms the original c=8 limiter was
serial local prompt build/feed admission and that parallel prompt prebuild is the
right vLLM-side offline optimization. It still does not create a strict c=8
throughput win: engine QPS remains about 0.536 versus SGLang c=8 at 2.540 QPS,
and warmed request latency is 4.714s mean / 7.563s p95. Treat this as a
diagnostic that moves the vLLM limiter from prompt feed to
engine/workload/talker-side tail; WER was not computed for the prebuild
artifacts.

The SGLang-only stress sweep with `max_running_requests=8` validates the current
operational shape:

- c=1/c=2/c=4: talker autoregressive generation is the dominant tail stage.
- c=8: throughput peaks on this ci-50 workload at 2.540 QPS and 5.372 generated
  audio seconds per wall second.
- c=16: throughput regresses and RTF rises sharply; this is beyond the current
  sweet spot with the measured recipe.
- Offline Whisper large-v3 WER on the same c=1/2/4/8/16 generated audio stays
  stable at 2.88%-3.85% corpus WER for the warmed sweep rows, so the throughput
  peak is not coming from degraded speech/text consistency.
- Tail-request analysis confirms the transition: c1/c4 tails are talker
  dominated, c8 is talker plus preprocessing admission queue, and c16 is often
  queue dominated even for short-output requests.
- Synthetic long-text speech output remains faster than real time even at c=8:
  a 944-character / 139-word input generates about 52.3s audio in 25.8s mean
  latency, RTF 0.4932.
- `code2wav_decode` is not the compute bottleneck. It remains roughly
  13-17ms/window across the measured regimes.
- Stage-to-stage connection from `talker_ar` to `code2wav` remains healthy:
  stream-hop p95 is about 15-24ms across the measured runs.
- Blindly increasing preprocessing concurrency is not a viable fix with the
  current memory/admission layout. `PREPROCESSING_MAX_CONCURRENCY=2` completed
  but reduced c=8 QPS by 35.4%; `PREPROCESSING_MAX_CONCURRENCY=4` produced OOMs
  and severe tail latency.

The current recommended serving window is c=4 to c=8 for warmed short-answer
Video-AMME-like workloads on 8x H20. For production-facing performance claims,
the service must be explicitly warmed before measurement because the first
compile/capture requests are not representative.

### 1.1 Handoff Readiness

The current evidence package is ready for collaborator review, with two scoped
caveats: official SeedTTS full-set data is not staged locally, and strict vLLM
c=8 serving-throughput parity still needs online ingress or a WER/ASR-scored
prebuild serving-path rerun. The audited claims in this report do not depend on
either missing item.

Current audit status:

| Gate | Result | Evidence |
| --- | --- | --- |
| One-command audit | PASS | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/audit_run_summary.json`: `ok=true` |
| Claim verifier | PASS | 17/17 checks pass, 0 failed |
| Requirement coverage | PASS | 34/34 requirements pass, 0 missing |
| Reproduction preflight | PASS | 62 checks, 0 required failures; only optional host-side Whisper cache warning |
| Reproduction command manifest | PASS | 63 commands, 7 phases, all required command IDs present |
| Metric provenance index | PASS | headline, stress, synthetic, vLLM diagnostic, acceptance, and stage metrics link to raw artifacts and rerun commands |
| Claim metric crosswalk | PASS | 10 external defense claims link to concrete metric rows, raw artifacts, and rerun commands |
| Objective requirement crosswalk | PASS | 11 original requirement rows link to objective rows, defense claims, metric rows, raw artifacts, and rerun commands |
| SGLang optimization lock | PASS | 26/26 checks; image, compiled/graph recipe, c=8 peak, stage handoff, and anti-recipes locked |
| vLLM optimization lock | PASS | 22/22 checks; image, compile, CUDA graph, cache, and prebuild evidence locked |
| vLLM online parity protocol | PASS | 18/18 checks; c=8 online parity upgrade gates declared, current package safe, online parity not claimed |
| Runtime Image Contract | PASS | 12/12 checks; SGLang/vLLM image digests, GPU contract, optimization switches, and claim scopes locked |
| Rerun Acceptance Contract | PASS | 17/17 checks; 18 rerun threshold and headline replacement rules locked; 34 required return-evidence files; 27 command return evidence rows; matrix complete |
| Final Checkpoint Watchlist | PASS | 24/24 checks; 7 watch items; final_completion_evidence_ready=true; checkpoint_phase=completion_audit_ready and completion_allowed_now=true |
| Stage Latency Budget | PASS | 12/12 checks; SGLang, synthetic speech, and vLLM offline stage pressure ratios generated |
| Stage Boundary Bottleneck Ledger | PASS | 12/12 checks; 37 audited stage boundaries plus 11 pressure-transition rows mapped to evidence, decision, and claim scope |
| Stage Reproduction Drilldown | PASS | 52 stage rows with jq queries, metric provenance rows, raw artifacts, and rerun command IDs |
| Stage Route Decision Matrix | PASS | 11 route-level decisions aggregate stage bottlenecks, safe talking points, evidence, and rerun commands |
| Original objective completion audit | PASS | 17 objective rows, 0 required failures, share-ready with documented caveats |
| Final readiness audit | PASS | ready=true, 0 required failures, share/send decision recorded |
| Share bundle manifest | PASS | ready=true, share reports, machine evidence, and chart assets hashed |
| Chart source consistency | PASS | 14 CSV/SVG chart assets are byte-exact regenerations from audited JSON sources |
| Share release seal | PASS | adjacent tarball seal confirms checksum, validations, final readiness, and caveats |
| Evidence manifest | PASS | current 196 records, minimum 180 records, 0 missing |

For handoff, start from this report and these machine-readable evidence files:

| Artifact | Path |
| --- | --- |
| Full report | `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_stress_performance_plan_20260621.md` |
| Chinese university technical report | `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_university_technical_report_zh_20260621.md` |
| Chinese university technical report JSON | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/university_technical_report.json` |
| Chinese final status summary | `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_final_status_summary_zh_20260621.md` |
| Chinese final share delivery note | `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_final_share_delivery_note_zh_20260621.md` |
| Chinese one-page scorecard | `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_one_page_scorecard_zh_20260621.md` |
| Chinese regime decision matrix | `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_regime_decision_matrix_zh_20260621.md` |
| Regime decision matrix JSON | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/regime_decision_matrix.json` |
| Chinese runtime comparison contract | `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_runtime_comparison_contract_zh_20260621.md` |
| Chinese SGLang Optimization Lock | `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_sglang_optimization_lock_zh_20260621.md` |
| Chinese vLLM Optimization Lock | `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_vllm_optimization_lock_zh_20260621.md` |
| Chinese vLLM Online Parity Protocol | `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_vllm_online_parity_protocol_zh_20260621.md` |
| Chinese Final Checkpoint Watchlist | `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_final_checkpoint_watchlist_zh_20260621.md` |
| Chinese Stage Latency Budget | `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_stage_latency_budget_zh_20260621.md` |
| Chinese Stage Boundary Bottleneck Ledger | `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_stage_boundary_bottleneck_ledger_zh_20260621.md` |
| Chinese stage causal graph | `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_stage_causal_graph_zh_20260621.md` |
| Chinese Stage Reproduction Drilldown | `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_stage_reproduction_drilldown_zh_20260621.md` |
| Chinese Stage Route Decision Matrix | `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_stage_route_decision_matrix_zh_20260621.md` |
| Chinese caveat adjudication matrix | `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_caveat_adjudication_matrix_zh_20260621.md` |
| Chinese share package index | `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_share_package_index_zh_20260621.md` |
| Chinese collaborator brief | `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_collaboration_brief_zh_20260621.md` |
| Chinese share deck outline | `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_share_deck_outline_zh_20260621.md` |
| Chinese requirement evidence map | `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_requirement_evidence_map_zh_20260621.md` |
| Chinese pressure matrix | `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_pressure_matrix_zh_20260621.md` |
| Chinese metric source map | `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_metric_source_map_zh_20260621.md` |
| Chinese stage metric dictionary | `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_stage_metric_dictionary_zh_20260621.md` |
| Chinese stage causal graph | `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_stage_causal_graph_zh_20260621.md` |
| Chinese caveat adjudication matrix | `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_caveat_adjudication_matrix_zh_20260621.md` |
| Chinese defense Q&A | `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_defense_qna_zh_20260621.md` |
| Chinese optimization playbook | `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_optimization_playbook_zh_20260621.md` |
| Chinese reproduction checklist | `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_reproduction_checklist_zh_20260621.md` |
| Chinese external handoff runbook | `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_external_handoff_runbook_zh_20260621.md` |
| Chinese collaborator rerun validation sheet | `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_collaborator_rerun_validation_sheet_zh_20260621.md` |
| Audit summary | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/audit_run_summary.json` |
| Evidence manifest | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/manifest.json` |
| Environment snapshot | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/environment_snapshot.json` |
| Claims verifier | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/claims_verification.json` |
| Requirement coverage | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/coverage_matrix.json` |
| Headline scorecard | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/headline_scorecard.json` |
| Metric provenance index | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/metric_provenance_index.json` |
| Share chart pack manifest | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/share_charts/chart_pack_manifest.json` |
| Share chart SVG/CSV directory | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/share_charts/` |
| Acceptance matrix | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/acceptance_matrix.json` |
| Confidence ledger | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/confidence_ledger.json` |
| Original objective completion audit | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/objective_completion_audit.json` |
| Objective requirement crosswalk | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/objective_requirement_crosswalk.json` |
| Reproduction command manifest | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/repro_command_manifest.json` |
| Defense claim matrix | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/defense_claim_matrix.json` |
| Claim metric crosswalk | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/claim_metric_crosswalk.json` |
| Final readiness audit | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/final_readiness_audit.json` |
| Share bundle manifest | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/share_bundle_manifest.json` |
| Table summary | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/tables_summary.json` |
| Stage interaction summary | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/stage_interaction_summary.json` |
| Stage drilldown index | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/stage_drilldown_index.json` |
| vLLM admission diagnosis | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/vllm_admission_diagnosis.json` |
| SGLang optimization lock JSON | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/sglang_optimization_lock.json` |
| vLLM optimization lock JSON | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/vllm_optimization_lock.json` |
| vLLM online parity protocol JSON | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/vllm_online_parity_protocol.json` |
| vLLM log stages | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/vllm_log_stage_summary.json` |

The fastest local verification command is:

```bash
cd /home/gangouyu/sglang-omni

python3 -m benchmarks.eval.run_qwen35_omni_report_audit \
  --root /home/gangouyu/sglang-omni \
  --summary-output results/qwen35_report_audit_20260619/audit_run_summary.json
```

The environment snapshot JSON records the GPU inventory, Docker image IDs,
git state, model/data paths, and current audit summaries for handoff.

## 2. Methodology And Controls

Measured workloads:

| Workload | Purpose | Coverage |
| --- | --- | --- |
| Video-AMME ci-50, video + spoken question, text + speech answer | Main SGLang/vLLM aligned multimodal workload | vLLM c=1/4/8; SGLang c=1/2/4/8/16 |
| Synthetic text-to-speech short prompt | Isolate thinker/talker/code2wav without video/audio encoders | SGLang c=1/4/8 |
| Synthetic text-to-speech long prompt | Stress long talker/code2wav streaming path | SGLang c=1/4/8 |
| Video-AMME spoken-reference SeedTTS-compatible smoke | Reuse local spoken-question wav files as reference audio when official SeedTTS cache is absent | Meta generation only; optional Qwen3.5-Omni voice-clone/TTS smoke |
| Preprocessing concurrency negative test | Check whether parallel preprocessing removes c=8/c=16 queueing | SGLang c=8 with preproc=2 and preproc=4 |

Controls and definitions:

- Latency is end-to-end client-observed request latency from the benchmark
  runner.
- RTF is latency divided by generated audio duration; lower is better.
- Audio throughput is generated audio seconds per wall-clock second; higher is
  better.
- SGLang stress sweeps used `--skip-wer` during serving to avoid ASR contention.
  WER was then computed offline from saved wav files with
  `benchmarks.eval.compute_audio_consistency_from_results`. The current run used
  local OpenAI Whisper `large-v3` from `/root/.cache/whisper/large-v3.pt`; the
  same tool can also target an OpenAI-compatible ASR router.
- `stage_input_received->stage_complete` includes stage-local queue/admission
  time. Internal spans such as `preprocess_start->preprocess_end` and
  `code2wav_decode_start->code2wav_decode_end` are used to separate actual
  compute from waiting/backpressure.
- Warmed comparisons are explicit. For the vLLM/SGLang c=4 comparison, the
  first four requests are excluded because they include cold compile/capture
  behavior.
- The vLLM offline runner starts per-request latency timing after `_build_prompt`
  completes. `_build_prompt` includes local video/audio prompt construction
  before `omni.add_request()`. Therefore vLLM latency tables describe engine
  request time, while vLLM wall QPS includes local prompt construction and is
  the better throughput measure for that runner.

## 3. Environment

| Item | Value |
| --- | --- |
| Host workspace | `/home/gangouyu/sglang-omni` |
| Git HEAD | `41cb2ae40cd2ece3dad079d6fb7107a1d1779076` |
| SGLang container | `b5f665f3d883` / `frankleeeee/sglang-omni:dev` |
| SGLang image digest | `sha256:be7e72126f525c3767008a73de16f400f974a09db431ded3c52bd48370941a84` |
| SGLang image created | `2026-02-05T09:30:27.304070315Z` |
| Container Python | `3.12.3` |
| Container Torch | `2.9.1+cu129` |
| Container SGLang | `0.0.0.dev1+gdce8b0606` |
| Container Triton | `3.5.1` |
| GPU | 8x NVIDIA H20, 97871 MiB each |
| Model path | `/myapp/models/qwen3_5_omni_23b_final_multilingual_all_voice_bf16_0315` |
| Video-AMME cache | `/myapp/data/videoamme` |

The vLLM baseline ran in the Qwen3.5-capable image:
`tongyi-duanwu-registry-vpc.cn-beijing.cr.aliyuncs.com/dashscope/dashllm:cuda129_cp312_test_vl_13589`.
The run log reports `torch 2.8.0a0+nv25.06`; the initial `triton 3.6.0`
environment failed a minimal CUDA `torch.compile` smoke test with
`ImportError: cannot import name 'triton_key'`. Pinning `triton==3.3.1` fixed
the compile smoke test and was used for the optimized vLLM run.

Reference architecture context:

- Qwen3.5-Omni uses a Thinker/Talker-style multimodal architecture with
  streaming speech generation; see the Qwen3.5-Omni technical report:
  https://arxiv.org/html/2604.15804v2
- vLLM-Omni documents Qwen3-Omni online serving, stage deployment, and
  stage-specific overrides here:
  https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/examples/online_serving/qwen3_omni/

## 4. Current Best SGLang Serving Recipe

Recommended stress recipe for the current SGLang c=1/2/4/8/16 sweep:

```bash
cd /myapp/sglang-omni

NO_CODE2WAV_TORCH_COMPILE=0 \
TORCHDYNAMO_DISABLE=0 \
SGLANG_OMNI_VIDEO_PREPROCESS_CACHE_MAX_BYTES=17179869184 \
SGLANG_OMNI_VIDEO_PREPROCESS_CACHE_MAX_ENTRIES=64 \
EXTRA_ARGS="--thinker-cuda-graph on --talker-cuda-graph on --talker-torch-compile on --thinker-max-running-requests 8 --talker-max-running-requests 8" \
bash examples/launch_qwen35_omni_speech_server_container.sh
```

Important serving details:

- Keep `--thinker-cuda-graph on`, `--talker-cuda-graph on`, and
  `--talker-torch-compile on` for warmed steady-state performance.
- Keep `NO_CODE2WAV_TORCH_COMPILE=0`; compiled code2wav is stable and actual
  decode compute is not the bottleneck.
- `max_running_requests=8` is the current stress recipe. It captures enough
  shapes for the c=8 sweep and exposes the c=16 saturation point.
- The earlier head-to-head vLLM comparison artifact used a c=4-oriented SGLang
  recipe. Use that artifact for strict vLLM comparison; use the recipe above
  for the broader SGLang stress sweep.
- Keep `PREPROCESSING_MAX_CONCURRENCY=1` for the current recipe.
  `PREPROCESSING_MAX_CONCURRENCY=2` completed but reduced c=8 throughput and
  inflated encoder/thinker spans; `PREPROCESSING_MAX_CONCURRENCY=4` caused OOMs.

### 4.1 Optimization Evidence Ledger

This ledger ties each important optimization switch to the evidence that keeps
it in the recommended recipe. It also records settings that look attractive but
are not part of the current best configuration.

| Runtime | Setting | Role In The Optimized Recipe | Evidence / Guardrail |
| --- | --- | --- | --- |
| SGLang | `--thinker-cuda-graph on`, `--talker-cuda-graph on`, `--talker-torch-compile on` | Keep Thinker/Talker request execution on the warmed compiled/graph path | Warmed c=4 beats optimized vLLM on latency and RTF; cold compile/capture is reported separately in section 10 |
| SGLang | `NO_CODE2WAV_TORCH_COMPILE=0`, `TORCHDYNAMO_DISABLE=0` | Keep code2wav compiled while preserving torch dynamo coverage | `code2wav_decode` stays about 13-17ms/window and is not the bottleneck |
| SGLang | `--thinker-max-running-requests 8`, `--talker-max-running-requests 8` | Expose the current c4-c8 operating window and the c16 saturation point | c=8 is the throughput peak at 2.540 QPS; c=16 regresses to 2.407 QPS |
| SGLang | `SGLANG_OMNI_VIDEO_PREPROCESS_CACHE_MAX_BYTES=17179869184`, `SGLANG_OMNI_VIDEO_PREPROCESS_CACHE_MAX_ENTRIES=64` | Stabilize repeated Video-AMME preprocessing during the stress sweep | Preprocessing actual compute stays near 0.27-0.32s even as lifecycle queueing grows at c8/c16 |
| SGLang | `PREPROCESSING_MAX_CONCURRENCY=1` | Current safe admission point for H20 96GB memory layout | preproc=2 lowers c=8 QPS by 35.4%; preproc=4 produces OOM/failures |
| vLLM | Qwen3.5-capable image plus `triton==3.3.1` | Avoid the local `torch.compile` smoke-test failure seen with the initial Triton stack | Optimized vLLM c=4 artifact runs with compile enabled and provides the strict comparison baseline |
| vLLM | `enforce_eager=False`, compile mode, `FULL_AND_PIECEWISE` CUDA graph, talker code predictor CUDA graph | Keep vLLM on its optimized graph/compile path rather than a conservative eager baseline | Warmed c=4 comparison is against this optimized vLLM path, not an intentionally weak baseline |
| vLLM | code2wav `torch.compile`, prefix caching, chunked prefill, shared-memory inter-stage transfer | Enable the published vLLM-Omni performance features for Qwen3.5-style online serving paths | Log-derived engine-side encoder/thinker/talker/code2wav boundaries are small in original c1/c4/c8 |
| vLLM | `max_num_seqs=4` for strict c=4, `max_num_seqs=8` for c1/c8 diagnostics | Match the comparison point while allowing c8 diagnostic pressure | Original c8 is prompt-feed limited; prebuild w4 is used as the strongest offline c8 diagnostic |
| vLLM | `--prebuild-prompts --prebuild-workers 4` | Remove serial local prompt build/feed from the offline runner's c8 admission path | Prompt-build wall drops from 249.3s to 129.2s and runner QPS rises from 0.1420 to 0.2127; still not an online serving parity claim |

### 4.2 SGLang Optimization Lock

The reviewer-facing SGLang Optimization Lock is:
`/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_sglang_optimization_lock_zh_20260621.md`.
Its machine JSON is
`/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/sglang_optimization_lock.json`.
It verifies 26/26 required evidence needles for the SGLang image, launch script,
compiled/graph recipe, c=8 throughput peak, c=16 saturation boundary,
code2wav decode health, talker-to-code2wav stream handoff, synthetic short/long
speech coverage, and preprocessing anti-recipes. Use it when answering whether
the SGLang side is a deliberate best-known recipe rather than an accidental
single run.

### 4.3 vLLM Optimization Lock

The reviewer-facing vLLM Optimization Lock is:
`/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_vllm_optimization_lock_zh_20260621.md`.
Its machine JSON is
`/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/vllm_optimization_lock.json`.
It verifies 22/22 required evidence needles for the vLLM image, compile path,
`FULL_AND_PIECEWISE` CUDA graph, prefix/chunked prefill, shared-memory transfer,
encoder compile/batch, and the `--prebuild-prompts --prebuild-workers 4`
offline diagnostic. Use it when answering whether the vLLM baseline was a
real optimized baseline rather than an intentionally weak comparison point.

### 4.4 vLLM Online Parity Protocol

The reviewer-facing vLLM Online Parity Protocol is:
`/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_vllm_online_parity_protocol_zh_20260621.md`.
Its machine JSON is
`/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/vllm_online_parity_protocol.json`.
It verifies 18/18 protocol checks and records `online_parity_proven=false`.
Use it when a collaborator wants to upgrade the current c=8 vLLM evidence from
offline diagnostic to strict online serving parity: it lists the required
online ingress artifact, WER/ASR artifact, stage profile, and replacement gates.

## 5. vLLM Baseline Notes

The validated vLLM baseline used:

- `enforce_eager=False`
- vLLM compile mode enabled
- `FULL_AND_PIECEWISE` CUDA graph
- talker code predictor CUDA graph
- code2wav `torch.compile`
- prefix caching and chunked prefill enabled
- `max_num_seqs=4` for the strict c=4 comparison artifact; the later c=1/c=8
  vLLM checkpoints used `max_num_seqs=8`
- thinker `gpu_memory_utilization=0.9`, talker `gpu_memory_utilization=0.4`

The rerun wrapper also sets the following vLLM-Omni optimization switches, so
the comparison is against an optimized baseline rather than a conservative
out-of-box run:

| Area | Enabled Switches |
| --- | --- |
| Engine/runtime | `VLLM_USE_V1=1`, multiprocessing/spawn workers, `VLLM_ENABLE_TORCH_COMPILE=True` |
| Attention/graph | upstream FlashAttention disabled for the local stack, compile mode with CUDA graph, encoder torch compile |
| Multimodal preprocessing | GPU preprocessing, multi-thread preprocessing, encoder batch, shared-memory MM processor cache |
| Inter-stage transfer | hidden-buffer shared memory and fast transfer, talker external embedding, thinker-preprocess reuse |
| Prompt/update path | skip redundant text encode, prompt post-update, realtime MM metadata disabled for this offline workload |

The vLLM run log captures the full engine args at:
`/home/gangouyu/sglang-omni/results/qwen35_vllm_videoamme_ci50_official_talker_compile_c4_20260618_202106/run.log`.

The full 50-request vLLM report is intentionally not used as the main
steady-state number because early compile/capture requests contaminate the
tail:

| Runtime | Scope | n | Accuracy | Latency Mean | Latency P95 | RTF Mean | RTF P95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vLLM optimized | full ci-50, c=4 | 50 | 66.0% | 4.108s | 21.530s | 3.5165 | 15.1461 |
| vLLM optimized | skip first 4, c=4 | 46 | 63.0% | 2.093s | 3.525s | 1.4677 | 3.0717 |

For fair steady-state comparison, use the warmed skip-first-4 row.

Additional vLLM c=1 and c=8 checkpoints:

| Runtime | Scope | n | Accuracy | Latency Mean | Latency P95 | RTF Mean | RTF P95 | Wall QPS | Audio Throughput |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vLLM optimized | full ci-50, c=1, `max_num_seqs=8` | 50 | 66.0% | 2.713s | 4.301s | 1.8882 | 3.0283 | 0.127 | 0.311 |
| vLLM optimized | skip first 4, c=1, `max_num_seqs=8` | 46 | 63.0% | 2.076s | 3.545s | 1.4499 | 2.5473 | 0.1465 | 0.3606 |
| vLLM optimized | full ci-50, c=8, `max_num_seqs=8` | 50 | 66.0% | 4.409s | 21.356s | 3.0649 | 13.8083 | 0.162 | 0.382 |
| vLLM optimized | skip first batch, c=8, `max_num_seqs=8` | 42 | 69.0% | 1.991s | 3.260s | 1.5141 | 3.1987 | 0.1752 | 0.3605 |

Prebuilt-prompt c=8 diagnostic, same ci-50 set and same optimized vLLM runner:

| Runtime | Scope | n | Accuracy | Latency Mean | Latency P95 | RTF Mean | RTF P95 | Runner QPS | Engine QPS | Engine Audio Throughput | Note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| vLLM optimized prebuilt prompt w1 | full ci-50, c=8, `max_num_seqs=8`, `--prebuild-prompts --prebuild-workers 1` | 50 | 66.0% | 9.349s | 34.083s | 6.3461 | 27.3093 | 0.1420 | 0.5391 | 1.2774 | prompt prebuild wall 249.3s |
| vLLM optimized prebuilt prompt w1 | skip first batch, c=8, workers=1 | 42 | 69.0% | 4.689s | 7.009s | 3.5781 | 6.2581 | n/a | n/a | n/a | diagnostic row; WER not computed |
| vLLM optimized prebuilt prompt w4 | full ci-50, c=8, `max_num_seqs=8`, `--prebuild-prompts --prebuild-workers 4` | 50 | 66.0% | 9.393s | 34.141s | 6.1411 | 22.0333 | 0.2127 | 0.5360 | 1.2538 | prompt prebuild wall 129.2s |
| vLLM optimized prebuilt prompt w4 | skip first batch, c=8, workers=4 | 42 | 69.0% | 4.714s | 7.563s | 3.8252 | 6.1605 | n/a | n/a | n/a | best runner wall in this offline family; WER not computed |

Interpretation:

- c=1: SGLang-Omni's c=1 stress row is faster than warmed vLLM c=1 on latency
  mean, latency p95, RTF mean, and RTF p95.
- c=8: vLLM's warmed engine request latency is good after the first batch, but
  the offline runner's wall throughput remains far below SGLang c=8 because
  local video prompt construction feeds the engine gradually. This is a runner
  bottleneck and should not be confused with code2wav compute.
- c=8 prebuild: single-worker prompt prebuild reduces engine admission span by
  about 7.5x versus the original c8 run. Four-worker prebuild then cuts prompt
  build wall by 48.2% and runner wall by 33.2% versus the single-worker artifact.
  The bottleneck moves from prompt feed to engine/workload/talker-side tail, so
  this is the best vLLM offline diagnostic artifact here, not a strict
  serving-throughput win.
- The c=8 first batch dominates full-run p95/p99: the top three request
  latencies are 37.8s, 27.9s, and 23.1s, all from sample group 001 in the first
  batch. Dropping the first batch brings p95 latency down to 3.260s.

The runner-overhead estimate below is computed by summing the maximum observed
request latency in each fixed-size batch and comparing it with the recorded wall
time. It is not a replacement for a true online vLLM server benchmark, but it
quantifies how much of the current offline wall time is outside the measured
request latency window:

| Label | c | Requests | Runner Wall s | Engine Wall s | Prompt Build s | Batch-max Sum s | Runner Overhead | Engine Overhead | Runner QPS | Engine QPS | Batch-max QPS | Runner Audio Thr | Engine Audio Thr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vLLM-c4 | 4 | 50 | 325.5 | 325.5 | 0.0 | 76.0 | 249.6s (76.7%) | 249.6s (76.7%) | 0.1536 | 0.1536 | 0.6582 | 0.3888 | 0.3888 |
| vLLM-c8 | 8 | 50 | 308.2 | 308.2 | 0.0 | 56.2 | 252.0s (81.8%) | 252.0s (81.8%) | 0.1622 | 0.1622 | 0.8900 | 0.3823 | 0.3823 |
| vLLM-c8-prebuild-w1 | 8 | 50 | 352.1 | 92.8 | 249.3 | 79.0 | 273.2s (77.6%) | 13.8s (14.9%) | 0.1420 | 0.5391 | 0.6333 | 0.3365 | 1.2774 |
| vLLM-c8-prebuild-w4 | 8 | 50 | 235.1 | 93.3 | 129.2 | 80.8 | 154.3s (65.6%) | 12.5s (13.4%) | 0.2127 | 0.5360 | 0.6186 | 0.4975 | 1.2538 |

Log-derived vLLM stage signals show that the engine-side stage boundaries are
not hiding a large code2wav or inter-stage bottleneck. The table below drops
the same warmup requests as the corresponding latency rows (`skip=4` for c1/c4,
first batch `skip=8` for c8). `Processor Total` is the model processor's
reported actual multimodal preprocessing time, while `Input Preproc` is the
larger vLLM input-preprocessor lifecycle log cost. Small negative raw deltas
from asynchronous logging are clamped to zero for boundary-latency fields.

| Run | Skip | Request IDs | Processor Total Avg/P95 | Input Preproc Avg/P95 | Encoder Avg/P95 | Thinker->Talker Avg/P95 | Feed->Codec Avg/P95 | Codec Gap Avg/P95 | Talker->C2W Drain Avg/P95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vLLM-c1 | 4 | 46/50 | 54.0/166.7ms | 324.9/873.4ms | 35.6/47.8ms | 0.2/1.0ms | 69.0/77.5ms | 13.5/16.1ms | 14.3/16.0ms |
| vLLM-c4 | 4 | 46/50 | 47.0/137.7ms | 322.1/847.3ms | 34.3/44.2ms | 0.2/1.0ms | 70.5/77.7ms | 13.5/16.1ms | 15.2/17.5ms |
| vLLM-c8 | 8 | 42/50 | 45.6/137.9ms | 323.7/851.2ms | 34.8/43.8ms | 0.2/1.0ms | 76.5/70.0ms | 13.5/15.7ms | 14.0/16.0ms |
| vLLM-c8-prebuild-w1 | 8 | 42/50 | 52.3/161.9ms | 367.9/975.4ms | 36.1/41.3ms | 1.1/4.0ms | 189.9/549.0ms | 20.0/28.3ms | 55.6/88.7ms |
| vLLM-c8-prebuild-w4 | 8 | 42/50 | 53.5/170.9ms | 332.2/886.9ms | 38.8/46.2ms | 1.3/3.9ms | 203.3/509.7ms | 19.4/26.7ms | 51.0/123.4ms |

The same log parser also quantifies batch admission lag: time from the offline
runner's `Running batch` log to the first/last engine-visible request in that
batch. This is the clearest evidence that the current vLLM c8 wall throughput
is limited by serial prompt build/feed before engine admission. For the
prebuilt-prompt row, first/last engine lag still includes the prompt prebuild
that happens after `Running batch`; the admission span itself is the engine-side
spread between the first and last visible request in the batch.

| Run | Included Batches | Req/Batch Avg/P95 | First Engine Lag Avg/P95 | Last Engine Lag Avg/P95 | Batch Admission Span Avg/P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| vLLM-c1 | 46/50 | 1.0/1.0 | 4746.7/5927.0ms | 4746.7/5927.0ms | 0.0/0.0ms |
| vLLM-c4 | 12/13 | 3.8/4.0 | 4641.5/5583.4ms | 19752.3/24188.7ms | 15110.8/19135.8ms |
| vLLM-c8 | 6/7 | 7.0/8.0 | 4691.5/5300.8ms | 38005.5/46881.0ms | 33314.0/43972.7ms |
| vLLM-c8-prebuild-w1 | 6/7 | 7.0/8.0 | 33576.7/42385.0ms | 38016.3/47342.0ms | 4439.7/5425.0ms |
| vLLM-c8-prebuild-w4 | 6/7 | 7.0/8.0 | 16508.2/20283.0ms | 20597.2/24887.3ms | 4089.0/4891.5ms |

For reruns, use the diagnosis table below as the quick gate. A result marked
`prompt_feed_limited` should not be used as a final c8 serving-throughput
comparison until prompt construction is prebuilt, parallelized, or moved behind
an online server ingress path. The measured `--prebuild-prompts --prebuild-workers 4`
row is the strongest vLLM offline artifact here, but it is still not a headline
serving-throughput comparison because prompt construction remains runner work
and WER was not computed for that artifact.

| Label | c | Runner QPS | Engine QPS | Batch-max QPS | Runner Overhead | Engine Overhead | Admission Span Avg/P95 | Last Engine Lag Avg/P95 | Boundary P95 Encoder/C2W | Diagnosis |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| vLLM-c4 | 4 | 0.1536 | 0.1536 | 0.6582 | 249.6s (76.7%) | 249.6s (76.7%) | 15110.8/19135.8ms | 19752.3/24188.7ms | 44.2/17.5ms | prompt_feed_limited |
| vLLM-c8 | 8 | 0.1622 | 0.1622 | 0.8900 | 252.0s (81.8%) | 252.0s (81.8%) | 33314.0/43972.7ms | 38005.5/46881.0ms | 43.8/16.0ms | prompt_feed_limited |
| vLLM-c8-prebuild-w1 | 8 | 0.1420 | 0.5391 | 0.6333 | 273.2s (77.6%) | 13.8s (14.9%) | 4439.7/5425.0ms | 38016.3/47342.0ms | 41.3/88.7ms | engine_or_workload_limited |
| vLLM-c8-prebuild-w4 | 8 | 0.2127 | 0.5360 | 0.6186 | 154.3s (65.6%) | 12.5s (13.4%) | 4089.0/4891.5ms | 20597.2/24887.3ms | 46.2/123.4ms | engine_or_workload_limited |

Interpretation: vLLM's multimodal encoder p95 remains below 50ms in all warmed
slices. In the original c4/c8 rows, talker-to-code2wav drain p95 also remains
below 18ms, so the c8 throughput caveat is dominated by offline prompt
construction, batch feed/admission span, and first-batch cold behavior. In the
prebuilt-prompt rows, admission span improves materially. With w4, feed-to-codec
and talker-to-code2wav drain p95 are 509.7ms and 123.4ms respectively. The
limiter has moved to engine/workload/talker-side tail rather than being solved.

## 6. Validated Artifacts

### 6.1 Requirement Coverage Matrix

The table below maps the original handoff requirements to the strongest local
evidence and reproduction entry points. The machine-readable form is
`/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/coverage_matrix.json`.

| Status | Requirement | Evidence | Reproducibility |
| --- | --- | --- | --- |
| PASS | SGLang vs vLLM optimized baseline, at least comparable | Warmed c=4 SGLang beats vLLM on latency mean/p95 and RTF mean/p95 while preserving accuracy/WER. | claims verifier; sections 1/5 |
| PASS | Optimization switch ledger and base/optimized boundary | SGLang and vLLM optimization switches, anti-recipes, and comparison boundaries are explicitly documented. | section 4.1; coverage matrix |
| PASS | Share-package index and defense route | Chinese package index tells collaborators what to read first, which evidence to inspect, and which claims are safe. | `qwen35_omni_share_package_index_zh_20260621.md`; section 1.1 |
| PASS | Collaborator-facing share deck outline | Chinese deck outline maps the headline, stage evidence, caveats, and reproduction gates into a 15-25 minute sharing flow. | `qwen35_omni_share_deck_outline_zh_20260621.md`; share index |
| PASS | Original-objective evidence map | Chinese evidence map ties each original user requirement to the strongest local evidence, confidence boundary, and reproduction entry point. | `qwen35_omni_requirement_evidence_map_zh_20260621.md`; share index |
| PASS | Collaborator-facing defense Q&A | Chinese defense Q&A gives ready-to-say answers, unsafe-wording boundaries, and evidence links for common collaborator questions. | `qwen35_omni_defense_qna_zh_20260621.md`; share index |
| PASS | Optimization playbook and safe tuning protocol | Chinese optimization playbook maps measured bottlenecks to safe knobs, experiment order, acceptance gates, and rollback rules. | `qwen35_omni_optimization_playbook_zh_20260621.md`; section 12/14 |
| PASS | Single-concurrency and high-concurrency SGLang stress | Video-AMME stress covers c=1/2/4/8/16. | `tables_summary.json`; section 7 |
| PASS | Short/long text-input and speech-output workloads | Synthetic text-to-speech covers a 74-character / 12-word short input and a 944-character / 139-word long input at c=1/4/8; long output remains faster than real time. | `tables_summary.json`; section 8 |
| PASS | Spoken-reference/SeedTTS-style smoke path | Local Video-AMME spoken-question audio is exported as a SeedTTS-compatible `meta.lst`; official SeedTTS remains an optional follow-up. | `videoamme_seedtts_meta.lst`; section 8.3/11.6 |
| PASS | Stage-level breakdown and stage connections | SGLang Video-AMME, synthetic speech, vLLM log stages, and talker/code2wav boundary checks are present. | stage tables; claims verifier |
| PASS | Machine-readable stage interaction summary | Stage boundary health, queue/admission effects, vLLM prompt-feed limits, and preprocessing contention are summarized in JSON. | `stage_interaction_summary.json`; section 12.1 |
| PASS | Machine-readable headline scorecard | Headline c=4 comparison, c=8 peak, long speech, vLLM c8 diagnostic, and stage flags are summarized in JSON. | `headline_scorecard.json`; section 1.1 |
| PASS | Machine-readable per-regime acceptance matrix | Single/high concurrency, short/long text-to-speech, vLLM diagnostics, and anti-recipes have per-regime pass/fail evidence. | `acceptance_matrix.json`; section 1.1 |
| PASS | Machine-readable confidence ledger | Safe high-confidence claims, medium-confidence boundaries, and unsupported-claim guardrails are summarized in JSON. | `confidence_ledger.json`; section 1.1/14 |
| PASS | Bottleneck and optimization guidance | c=8 peak, c=16 queueing, preproc=2 regression, preproc=4 OOM, and vLLM prompt-feed caveat are all explicitly diagnosed. | sections 9/12; admission diagnosis |
| PASS | Human-readable pressure-condition matrix | Chinese pressure matrix summarizes every measured pressure condition, recommendation status, bottleneck, and evidence path in one human-readable table. | `qwen35_omni_pressure_matrix_zh_20260621.md`; `acceptance_matrix.json` |
| PASS | Human-readable metric source map | Chinese metric source map ties headline, pressure, stage, vLLM diagnostic, and anti-recipe numbers to machine evidence and regeneration commands. | `qwen35_omni_metric_source_map_zh_20260621.md`; `headline_scorecard.json`; `tables_summary.json` |
| PASS | Human-readable stage metric dictionary | Chinese stage metric dictionary defines lifecycle, compute, handoff, collect-wait, and vLLM admission metrics so stage breakdowns are interpreted correctly. | `qwen35_omni_stage_metric_dictionary_zh_20260621.md`; `stage_interaction_summary.json`; `tables_summary.json` |
| PASS | Human-readable stage causal graph | Chinese stage causal graph explains admission/queue, talker cadence, stream hop, code2wav collect/decode, and vLLM offline admission as connected causal edges. | `qwen35_omni_stage_causal_graph_zh_20260621.md`; `stage_causal_graph.json`; `stage_interaction_summary.json` |
| PASS | Human-readable caveat adjudication matrix | Chinese caveat adjudication matrix classifies shareable boundaries, forbidden wording, rerun-triggered upgrades, and report-number replacement triggers. | `qwen35_omni_caveat_adjudication_matrix_zh_20260621.md`; `confidence_ledger.json`; `objective_completion_audit.json` |
| PASS | External handoff runbook for reviewers | Chinese external handoff runbook gives reviewers the shortest audited reproduction path, rerun order, stage-reading rules, and replacement gates. | `qwen35_omni_external_handoff_runbook_zh_20260621.md`; `run_qwen35_omni_report_audit.py` |
| PASS | Collaborator rerun validation sheet | Chinese collaborator rerun validation sheet gives reviewers a structured pass/fail worksheet for environment deltas, SGLang/vLLM reruns, stage flags, replacement rules, and return artifacts. | `qwen35_omni_collaborator_rerun_validation_sheet_zh_20260621.md`; `audit_run_summary.json` |
| PASS | Final share delivery note | Chinese final share delivery note tells the sender which files and machine evidence to include, how to introduce the package, which claims are safe, and which caveats must travel with the share. | `qwen35_omni_final_share_delivery_note_zh_20260621.md`; share package index |
| PASS | One-page core-number scorecard | Chinese one-page scorecard condenses the cross-runtime headline, SGLang pressure sweep, short/long text-to-speech guardrails, stage health, vLLM c8 diagnostic, and current gates for quick sharing. | `qwen35_omni_one_page_scorecard_zh_20260621.md`; `headline_scorecard.json`; `tables_summary.json` |
| PASS | Slide-ready chart and CSV pack | Audited SVG/CSV assets provide slide-ready charts and spreadsheet rows derived from report JSONs, with byte-exact source consistency checks to prevent hand-edited numbers. | `share_charts/chart_pack_manifest.json`; `chart_source_consistency.json`; `build_qwen35_omni_share_charts.py` |
| PASS | Reproduction and handoff readiness | One-command audit, Chinese final share delivery note, Chinese one-page scorecard, Chinese stage causal graph, Chinese caveat adjudication matrix, Chinese external handoff runbook, Chinese collaborator rerun validation sheet, Chinese reproduction checklist, preflight, manifest, and per-table regeneration commands are present. | section 11.8 |

| Item | Path |
| --- | --- |
| vLLM optimized ci-50 root | `/home/gangouyu/sglang-omni/results/qwen35_vllm_videoamme_ci50_official_talker_compile_c4_20260618_202106` |
| vLLM result JSON | `/home/gangouyu/sglang-omni/results/qwen35_vllm_videoamme_ci50_official_talker_compile_c4_20260618_202106/benchmark_audio_50_c4_offline_compile/videoamme_results.json` |
| vLLM WER JSON | `/home/gangouyu/sglang-omni/results/qwen35_vllm_videoamme_ci50_official_talker_compile_c4_20260618_202106/benchmark_audio_50_c4_offline_compile/whisper_large_v3_wer.json` |
| vLLM c=1 `max_num_seqs=8` root | `/home/gangouyu/sglang-omni/results/qwen35_vllm_videoamme_ci50_offline_compile_c1_mns8_20260619_20260619_220617` |
| vLLM c=1 `max_num_seqs=8` result JSON | `/home/gangouyu/sglang-omni/results/qwen35_vllm_videoamme_ci50_offline_compile_c1_mns8_20260619_20260619_220617/benchmark_audio_50_c1_offline_compile/videoamme_results.json` |
| vLLM c=8 `max_num_seqs=8` root | `/home/gangouyu/sglang-omni/results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_20260619_20260619_222434` |
| vLLM c=8 `max_num_seqs=8` result JSON | `/home/gangouyu/sglang-omni/results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_20260619_20260619_222434/benchmark_audio_50_c8_offline_compile/videoamme_results.json` |
| vLLM c=8 prebuilt-prompt w1 root | `/home/gangouyu/sglang-omni/results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuild_20260620_002020` |
| vLLM c=8 prebuilt-prompt w1 result JSON | `/home/gangouyu/sglang-omni/results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuild_20260620_002020/benchmark_audio_50_c8_offline_compile/videoamme_results.json` |
| vLLM c=8 prebuilt-prompt w1 report | `/home/gangouyu/sglang-omni/results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuild_20260620_002020/benchmark_audio_50_c8_offline_compile/vllm_videoamme_report.md` |
| vLLM c=8 prebuilt-prompt w4 root | `/home/gangouyu/sglang-omni/results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuildw4_20260620_005346` |
| vLLM c=8 prebuilt-prompt w4 result JSON | `/home/gangouyu/sglang-omni/results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuildw4_20260620_005346/benchmark_audio_50_c8_offline_compile/videoamme_results.json` |
| vLLM c=8 prebuilt-prompt w4 report | `/home/gangouyu/sglang-omni/results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuildw4_20260620_005346/benchmark_audio_50_c8_offline_compile/vllm_videoamme_report.md` |
| vLLM c=1 run log | `/home/gangouyu/sglang-omni/results/qwen35_vllm_videoamme_ci50_offline_compile_c1_mns8_20260619_20260619_220617/run.log` |
| vLLM c=4 run log | `/home/gangouyu/sglang-omni/results/qwen35_vllm_videoamme_ci50_official_talker_compile_c4_20260618_202106/run.log` |
| vLLM c=8 run log | `/home/gangouyu/sglang-omni/results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_20260619_20260619_222434/run.log` |
| vLLM c=8 prebuilt-prompt w1 run log | `/home/gangouyu/sglang-omni/results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuild_20260620_002020/run.log` |
| vLLM c=8 prebuilt-prompt w4 run log | `/home/gangouyu/sglang-omni/results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuildw4_20260620_005346/run.log` |
| vLLM offline rerun shell wrapper | `/home/gangouyu/sglang-omni/results/qwen35_vllm_videoamme_ci50_opt_20260618_162319/run_vllm_videoamme_ci5_offline_compile.sh` |
| vLLM offline Video-AMME runner | `/home/gangouyu/sglang-omni/results/qwen35_vllm_videoamme_ci50_opt_20260618_162319/vllm_videoamme_runner.py` |
| SGLang c=4 comparison root | `/home/gangouyu/sglang-omni/results/qwen35_sglang_subtalker_seedfix_compile_mr4_ci50_c4_20260618_181046` |
| SGLang c=4 comparison result JSON | `/home/gangouyu/sglang-omni/results/qwen35_sglang_subtalker_seedfix_compile_mr4_ci50_c4_20260618_181046/benchmark_audio_50_c4_warm_profile_no_wer/videoamme_results.json` |
| SGLang c=4 comparison WER JSON | `/home/gangouyu/sglang-omni/results/qwen35_sglang_subtalker_seedfix_compile_mr4_ci50_c4_20260618_181046/benchmark_audio_50_c4_warm_profile_no_wer/whisper_large_v3_wer.json` |
| SGLang stress sweep root | `/home/gangouyu/sglang-omni/results/qwen35_sglang_mr8_stress_20260619` |
| SGLang stress sweep local Whisper WER JSONs | `/home/gangouyu/sglang-omni/results/qwen35_sglang_mr8_stress_20260619/benchmark_audio_50_c{1,2,4,8,16}_*/whisper_large_v3_local_wer.json` |
| SGLang synthetic speech root | `/home/gangouyu/sglang-omni/results/qwen35_synthetic_speech_20260619` |
| Preprocessing concurrency preproc=2 root | `/home/gangouyu/sglang-omni/results/qwen35_sglang_preproc2_mr8_c8_20260619` |
| Preprocessing concurrency preproc=2 result JSON | `/home/gangouyu/sglang-omni/results/qwen35_sglang_preproc2_mr8_c8_20260619/benchmark_audio_50_c8_preproc2_profile_skipwer/videoamme_results.json` |
| Preprocessing concurrency preproc=2 profile JSON | `/home/gangouyu/sglang-omni/results/qwen35_sglang_preproc2_mr8_c8_20260619/request_profile_c8_preproc2_profile_skipwer.json` |
| Preprocessing concurrency negative root | `/home/gangouyu/sglang-omni/results/qwen35_sglang_mr8_preproc4_stress_20260619` |
| Short checked-in c=4 report | `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_videoamme_perf_report_20260619.md` |
| Chinese share package index | `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_share_package_index_zh_20260621.md` |
| Chinese final share delivery note | `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_final_share_delivery_note_zh_20260621.md` |
| Chinese one-page scorecard | `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_one_page_scorecard_zh_20260621.md` |
| Chinese share deck outline | `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_share_deck_outline_zh_20260621.md` |
| Chinese requirement evidence map | `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_requirement_evidence_map_zh_20260621.md` |
| Chinese pressure matrix | `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_pressure_matrix_zh_20260621.md` |
| Chinese metric source map | `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_metric_source_map_zh_20260621.md` |
| Chinese stage metric dictionary | `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_stage_metric_dictionary_zh_20260621.md` |
| Chinese defense Q&A | `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_defense_qna_zh_20260621.md` |
| Chinese optimization playbook | `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_optimization_playbook_zh_20260621.md` |
| Chinese reproduction checklist | `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_reproduction_checklist_zh_20260621.md` |
| Chinese external handoff runbook | `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_external_handoff_runbook_zh_20260621.md` |
| Chinese collaborator rerun validation sheet | `/home/gangouyu/sglang-omni/benchmarks/reports/qwen35_omni_collaborator_rerun_validation_sheet_zh_20260621.md` |
| Offline WER postprocessor | `/home/gangouyu/sglang-omni/benchmarks/eval/compute_audio_consistency_from_results.py` |
| vLLM offline runner overhead summarizer | `/home/gangouyu/sglang-omni/benchmarks/eval/summarize_vllm_offline_runner_overhead.py` |
| vLLM log-stage summarizer | `/home/gangouyu/sglang-omni/benchmarks/eval/summarize_vllm_omni_log_stages.py` |
| vLLM admission diagnosis helper | `/home/gangouyu/sglang-omni/benchmarks/eval/diagnose_vllm_offline_admission.py` |
| Stage interaction summarizer | `/home/gangouyu/sglang-omni/benchmarks/eval/summarize_qwen35_stage_interactions.py` |
| Headline scorecard builder | `/home/gangouyu/sglang-omni/benchmarks/eval/build_qwen35_omni_headline_scorecard.py` |
| Metric provenance index builder | `/home/gangouyu/sglang-omni/benchmarks/eval/build_qwen35_omni_metric_provenance_index.py` |
| Claim metric crosswalk builder | `/home/gangouyu/sglang-omni/benchmarks/eval/build_qwen35_omni_claim_metric_crosswalk.py` |
| Objective requirement crosswalk builder | `/home/gangouyu/sglang-omni/benchmarks/eval/build_qwen35_omni_objective_requirement_crosswalk.py` |
| Share chart pack builder | `/home/gangouyu/sglang-omni/benchmarks/eval/build_qwen35_omni_share_charts.py` |
| Acceptance matrix builder | `/home/gangouyu/sglang-omni/benchmarks/eval/build_qwen35_omni_acceptance_matrix.py` |
| Confidence ledger builder | `/home/gangouyu/sglang-omni/benchmarks/eval/build_qwen35_omni_confidence_ledger.py` |
| Original objective completion audit builder | `/home/gangouyu/sglang-omni/benchmarks/eval/build_qwen35_omni_objective_completion_audit.py` |
| Reproduction command manifest builder | `/home/gangouyu/sglang-omni/benchmarks/eval/build_qwen35_omni_repro_command_manifest.py` |
| Final readiness audit builder | `/home/gangouyu/sglang-omni/benchmarks/eval/build_qwen35_omni_final_readiness.py` |
| Final status summary builder | `/home/gangouyu/sglang-omni/benchmarks/eval/build_qwen35_omni_final_status_summary.py` |
| Chinese university technical report builder | `/home/gangouyu/sglang-omni/benchmarks/eval/build_qwen35_omni_university_technical_report.py` |
| Regime decision matrix builder | `/home/gangouyu/sglang-omni/benchmarks/eval/build_qwen35_omni_regime_decision_matrix.py` |
| Runtime comparison contract builder | `/home/gangouyu/sglang-omni/benchmarks/eval/build_qwen35_omni_runtime_comparison_contract.py` |
| Rerun acceptance contract builder | `/home/gangouyu/sglang-omni/benchmarks/eval/build_qwen35_omni_rerun_acceptance_contract.py` |
| Stage causal graph builder | `/home/gangouyu/sglang-omni/benchmarks/eval/build_qwen35_omni_stage_causal_graph.py` |
| Stage reproduction drilldown builder | `/home/gangouyu/sglang-omni/benchmarks/eval/build_qwen35_omni_stage_reproduction_drilldown.py` |
| Stage route decision matrix builder | `/home/gangouyu/sglang-omni/benchmarks/eval/build_qwen35_omni_stage_route_decision_matrix.py` |
| Caveat adjudication matrix builder | `/home/gangouyu/sglang-omni/benchmarks/eval/build_qwen35_omni_caveat_adjudication_matrix.py` |
| Share bundle manifest builder | `/home/gangouyu/sglang-omni/benchmarks/eval/build_qwen35_omni_share_bundle_manifest.py` |
| Video-AMME SeedTTS-compatible meta builder | `/home/gangouyu/sglang-omni/benchmarks/eval/build_videoamme_seedtts_meta.py` |
| Reproduction preflight helper | `/home/gangouyu/sglang-omni/benchmarks/eval/preflight_qwen35_omni_repro.py` |
| Requirement coverage helper | `/home/gangouyu/sglang-omni/benchmarks/eval/summarize_qwen35_report_coverage.py` |
| Report artifact/table summarizer | `/home/gangouyu/sglang-omni/benchmarks/eval/summarize_qwen35_omni_report_artifacts.py` |
| Report claim verifier | `/home/gangouyu/sglang-omni/benchmarks/eval/verify_qwen35_omni_report_claims.py` |
| Report evidence manifest builder | `/home/gangouyu/sglang-omni/benchmarks/eval/build_qwen35_omni_report_manifest.py` |
| Full report audit runner | `/home/gangouyu/sglang-omni/benchmarks/eval/run_qwen35_omni_report_audit.py` |
| Table-summary audit JSON | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/tables_summary.json` |
| Claim-verification audit JSON | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/claims_verification.json` |
| vLLM log-stage audit JSON | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/vllm_log_stage_summary.json` |
| vLLM admission diagnosis JSON | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/vllm_admission_diagnosis.json` |
| Stage interaction summary JSON | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/stage_interaction_summary.json` |
| Headline scorecard JSON | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/headline_scorecard.json` |
| Metric provenance index JSON | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/metric_provenance_index.json` |
| Claim metric crosswalk JSON | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/claim_metric_crosswalk.json` |
| Objective requirement crosswalk JSON | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/objective_requirement_crosswalk.json` |
| Share chart pack manifest JSON | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/share_charts/chart_pack_manifest.json` |
| Chart source consistency JSON | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/chart_source_consistency.json` |
| Share chart SVG/CSV directory | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/share_charts/` |
| Acceptance matrix JSON | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/acceptance_matrix.json` |
| Confidence ledger JSON | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/confidence_ledger.json` |
| Original objective completion audit JSON | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/objective_completion_audit.json` |
| Reproduction command manifest JSON | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/repro_command_manifest.json` |
| Rerun acceptance contract JSON | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/rerun_acceptance_contract.json` |
| Final readiness audit JSON | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/final_readiness_audit.json` |
| Share bundle manifest JSON | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/share_bundle_manifest.json` |
| Reproduction preflight JSON | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/preflight_repro.json` |
| Requirement coverage JSON | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/coverage_matrix.json` |
| Video-AMME SeedTTS-compatible meta | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/videoamme_seedtts_meta.lst` |
| Video-AMME SeedTTS-compatible meta summary | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/videoamme_seedtts_meta_summary.json` |
| Evidence manifest JSON | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/manifest.json` |
| Environment snapshot JSON | `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/environment_snapshot.json` |

## 7. SGLang Video-AMME Stress Sweep

All rows below use Video-AMME ci-50, text+audio output, `--skip-wer` during
serving, offline Whisper large-v3 WER on saved wav files, `m02` voice,
`max_tokens=256`, `temperature=0.0`, `video_fps=2`, `video_max_frames=128`, and
`video_max_pixels=401408`.

| Concurrency | n | Accuracy | Latency Mean | Latency P95 | RTF Mean | RTF P95 | QPS | Audio Throughput | WER Corpus |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 50 | 70.0% | 1.316s | 2.406s | 1.0490 | 2.0198 | 0.760 | 1.487 | 3.85% |
| 2 | 50 | 70.0% | 1.508s | 3.124s | 1.0816 | 1.9309 | 1.315 | 2.745 | 3.85% |
| 4 | 50 | 70.0% | 1.929s | 3.633s | 1.4015 | 2.4983 | 2.036 | 4.079 | 3.85% |
| 8 | 50 | 70.0% | 3.064s | 5.853s | 2.2141 | 4.3925 | 2.540 | 5.372 | 3.23% |
| 16 | 50 | 70.0% | 6.066s | 7.846s | 4.8489 | 10.4087 | 2.407 | 4.759 | 2.88% |

Interpretation:

- Accuracy is stable at 70.0% across concurrency levels because the same
  deterministic prompt/model settings are used.
- Offline WER is also stable across concurrency. The c=8 throughput peak and
  c=16 saturation are performance effects, not evidence of speech/text quality
  collapse.
- Throughput improves from c=1 to c=8, then falls at c=16. The c=16 run has
  more queueing than useful batching for this workload.
- c=8 is the current throughput peak for Video-AMME ci-50 under this recipe.
- c=4 remains a good balanced latency/throughput operating point.

Offline WER detail:

| Concurrency | WER Corpus | WER Mean | WER P95 | >50% WER Samples | Artifact |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 3.85% | 6.74% | 50.0% | 0/50 | `results/qwen35_sglang_mr8_stress_20260619/benchmark_audio_50_c1_warm_profile_skipwer/whisper_large_v3_local_wer.json` |
| 2 | 3.85% | 8.74% | 50.0% | 1/50 | `results/qwen35_sglang_mr8_stress_20260619/benchmark_audio_50_c2_warm_profile_skipwer/whisper_large_v3_local_wer.json` |
| 4 | 3.85% | 7.08% | 50.0% | 0/50 | `results/qwen35_sglang_mr8_stress_20260619/benchmark_audio_50_c4_profile_skipwer/whisper_large_v3_local_wer.json` |
| 8 | 3.23% | 6.08% | 50.0% | 0/50 | `results/qwen35_sglang_mr8_stress_20260619/benchmark_audio_50_c8_profile_skipwer/whisper_large_v3_local_wer.json` |
| 16 | 2.88% | 4.74% | 50.0% | 0/50 | `results/qwen35_sglang_mr8_stress_20260619/benchmark_audio_50_c16_profile_skipwer/whisper_large_v3_local_wer.json` |

### 7.1 Video-AMME Stage Breakdown

| Concurrency | Top Stage Avg/P95 | Preproc Stage Avg/P95 | Talker Avg/P95 | Code2wav Stage Avg/P95 | Decode Avg/P95 | Window Collect Avg/P95 | Talker->Code2wav Hop Avg/P95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| c1 | talker_ar 444/1516ms | 295/306ms | 444/1516ms | 7/10ms | 14/18ms | 26/28ms | 8.4/15.5ms |
| c2 | talker_ar 526/2024ms | 334/542ms | 526/2024ms | 8/18ms | 16/20ms | 32/46ms | 7.7/16.1ms |
| c4 | talker_ar 663/2628ms | 487/956ms | 663/2628ms | 13/31ms | 16/22ms | 46/135ms | 6.3/17.8ms |
| c8 | preprocessing 1227/2164ms | 1227/2164ms | 983/4418ms | 28/99ms | 17/26ms | 68/164ms | 5.7/20.4ms |
| c16 | preprocessing 4395/5884ms | 4395/5884ms | 816/3636ms | 23/66ms | 17/24ms | 61/153ms | 5.7/19.7ms |

Key findings:

- At c=1/c=2/c=4, `talker_ar` is the dominant tail stage.
- At c=8/c=16, the top stage by lifecycle time becomes `preprocessing`, but
  this is primarily admission/queueing. Actual internal preprocessing compute
  stays close to 0.29-0.30s:

| Concurrency | Preproc Stage Avg/P95 | `preprocess_start->end` Avg/P95 | HF Processor Avg/P95 |
| ---: | ---: | ---: | ---: |
| 4 | 487/956ms | 296/353ms | 278/342ms |
| 8 | 1227/2164ms | 289/336ms | 272/325ms |
| 16 | 4395/5884ms | 305/341ms | 291/331ms |

- The `talker_ar -> code2wav` stream hop stays small and stable. There is no
  evidence that this connection is the c=8/c=16 bottleneck.
- `code2wav_decode` remains about 14-17ms/window on average. The vocoder is not
  the current compute bottleneck.
- `code2wav_window_collect` grows with concurrency because code2wav waits for
  enough codec chunks from talker; this should not be interpreted as vocoder
  compute.

## 8. Synthetic Text-Length And Speech Stress Sweep

The synthetic benchmark uses local fixed prompts and calls the OpenAI-compatible
chat completions endpoint with `modalities=["text", "audio"]`. It removes
Video-AMME video preprocessing and spoken-question audio encoding, so it is a
cleaner stress test for thinker/talker/code2wav output generation.

The short case is a 74-character / 12-word input. The long case is a
944-character / 139-word input. This is the report's audited short/long text
guardrail; the output side also becomes short/long audio because the input is
read out loud.

| Scenario | c | n | Target Chars | Target Words | Audio Mean | Latency Mean | Latency P95 | RTF Mean | RTF P95 | QPS | Audio Throughput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| short | 1 | 16 | 74 | 12 | 4.22s | 0.866s | 0.924s | 0.2052 | 0.2112 | 1.154 | 4.877 |
| short | 4 | 16 | 74 | 12 | 4.31s | 1.768s | 2.056s | 0.4105 | 0.4388 | 2.218 | 9.561 |
| short | 8 | 16 | 74 | 12 | 4.27s | 2.638s | 2.828s | 0.6257 | 0.7440 | 2.983 | 12.738 |
| long | 1 | 8 | 944 | 139 | 51.92s | 9.168s | 9.465s | 0.1766 | 0.1776 | 0.109 | 5.663 |
| long | 4 | 8 | 944 | 139 | 52.58s | 17.551s | 18.025s | 0.3338 | 0.3373 | 0.227 | 11.923 |
| long | 8 | 8 | 944 | 139 | 52.33s | 25.799s | 26.318s | 0.4932 | 0.5001 | 0.303 | 15.870 |

### 8.1 Synthetic Speech Stage Breakdown

| Scenario | c | Talker Avg/P95 | Code2wav Stage Avg/P95 | Decode Avg/P95 | Window Collect Avg/P95 | Talker->Code2wav Hop Avg/P95 |
| --- | ---: | --- | --- | --- | --- | --- |
| short | 1 | 849/904ms | 7/10ms | 13/14ms | 27/29ms | 8.6/14.9ms |
| short | 4 | 1723/2029ms | 27/64ms | 16/22ms | 58/126ms | 5.1/20.1ms |
| short | 8 | 2206/2502ms | 414/548ms | 15/24ms | 73/166ms | 4.6/21.2ms |
| long | 1 | 9091/9384ms | 9/14ms | 13/14ms | 28/29ms | 8.9/15.0ms |
| long | 4 | 17464/17947ms | 17/32ms | 14/22ms | 56/82ms | 4.6/20.4ms |
| long | 8 | 25572/26199ms | 155/242ms | 14/18ms | 82/139ms | 3.2/24.0ms |

Key findings:

- Long-text output c=1 generates about 52s of audio in 9.17s mean latency. This
  is a strong faster-than-real-time result.
- Long-text output c=8 remains faster than real time at RTF 0.4932, but latency
  increases as talker GPU time is shared across requests.
- `talker_ar` scales roughly with output length and concurrency and is the
  primary performance lever for long speech.
- `code2wav_decode` remains stable and small across both short and long speech.
- `code2wav stage` can look large in the short c=8 case, but the actual decode
  span is still only 15/24ms avg/p95. The additional lifecycle time is waiting
  and scheduling around chunk collection.

### 8.3 SeedTTS-Compatible Spoken-Reference Smoke Path

The host does not currently have the official SeedTTS cache or a local
SeedTTS `meta.lst`, so no SeedTTS full-set number is included in the headline
tables. To avoid leaving the reference-audio/TTS path unreproducible, the audit
pipeline now exports the local Video-AMME spoken-question wav files into a
SeedTTS-compatible `meta.lst`:

- Source: local Video-AMME CI snapshot under
  `/home/gangouyu/data/videoamme/hub/datasets--zhaochenyang20--Video_AMME_ci`.
- Output:
  `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/videoamme_seedtts_meta.lst`.
- Summary:
  `/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/videoamme_seedtts_meta_summary.json`.
- Scope: this is a smoke/reproducibility path for Qwen3.5-Omni voice-clone or
  plain TTS benchmarking with local reference audio. It is not a replacement for
  a SeedTTS natural-speech full-set benchmark.

Run it directly with:

```bash
cd /home/gangouyu/sglang-omni

python3 -m benchmarks.eval.build_videoamme_seedtts_meta \
  --output results/qwen35_report_audit_20260619/videoamme_seedtts_meta.lst \
  --summary-output results/qwen35_report_audit_20260619/videoamme_seedtts_meta_summary.json \
  --max-samples 50 \
  --target-mode audio_text
```

Then, with the Qwen3.5-Omni speech server running, the generated meta can drive
the existing benchmark path:

```bash
cd /myapp/sglang-omni

python -m benchmarks.eval.benchmark_omni_seedtts \
  --generate-only \
  --meta results/qwen35_report_audit_20260619/videoamme_seedtts_meta.lst \
  --output-dir results/qwen35_videoamme_seedtts_smoke_c8 \
  --model qwen3_5-omni \
  --port 8161 \
  --lang en \
  --voice-clone \
  --max-samples 50 \
  --max-concurrency 8 \
  --max-new-tokens 1024 \
  --temperature 0.0 \
  --disable-tqdm
```

## 9. Preprocessing Concurrency Negative Results

The current c=8/c=16 stage table shows large preprocessing lifecycle time, but
the bottleneck is not fixed by simply allowing more preprocessing requests to
run at once.

### 9.1 Preproc=2

```bash
PREPROCESSING_MAX_CONCURRENCY=2 \
NO_CODE2WAV_TORCH_COMPILE=0 \
TORCHDYNAMO_DISABLE=0 \
SGLANG_OMNI_VIDEO_PREPROCESS_CACHE_MAX_BYTES=17179869184 \
SGLANG_OMNI_VIDEO_PREPROCESS_CACHE_MAX_ENTRIES=64 \
EXTRA_ARGS="--thinker-cuda-graph on --talker-cuda-graph on --talker-torch-compile on --thinker-max-running-requests 8 --talker-max-running-requests 8" \
bash examples/launch_qwen35_omni_speech_server_container.sh
```

Result on Video-AMME ci-50 c=8, after an 8-request warmup:

| Setting | Completed | Failed | Accuracy | Latency Mean | Latency P95 | RTF Mean | RTF P95 | QPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| preproc=1, mr=8 baseline | 50 | 0 | 70.0% | 3.064s | 5.853s | 2.2141 | 4.3925 | 2.540 |
| preproc=2, mr=8 | 50 | 0 | 70.0% | 4.579s | 6.313s | 3.3123 | 6.9368 | 1.642 |

Relative to the current preproc=1 c=8 baseline, preproc=2 completed safely but
increased mean latency by 49.4%, increased mean RTF by 49.6%, and reduced QPS
by 35.4%.

Profiler evidence:

| Stage / Interval | preproc=1 Avg/P95 | preproc=2 Avg/P95 | Interpretation |
| --- | ---: | ---: | --- |
| preprocessing lifecycle | 1227/2164ms | 1171/2352ms | queue is not materially fixed |
| actual preprocessing compute | 289/336ms | 679/1213ms | media/preprocess work becomes slower |
| media load | 16/38ms | 357/877ms | concurrent video decode/cache pressure |
| image encoder lifecycle | 75/91ms | 858/1825ms | GPU0 encoder path is starved/contended |
| image encoder compute | 0.18/0.34ms | 412/1287ms | actual encoder dispatch is delayed |
| audio encoder lifecycle | 52/55ms | 328/684ms | audio path also regresses |
| audio encoder compute | 0.24/0.50ms | 228/481ms | GPU0 contention shows up in encoder span |
| thinker lifecycle | 125/239ms | 1004/1648ms | thinker prefill/emit path is dragged down |
| talker lifecycle | 983/4418ms | 1818/3579ms | talker is still a major tail component |
| code2wav decode | 17/26ms | 18/26ms | vocoder compute remains stable |

Tail attribution for preproc=2 also shows the same shape: the top request
(`017-1`) had 9.405s client latency, with 5.048s talker lifecycle, 3.137s
thinker lifecycle, and 1.183s preprocessing queue. The issue is system
contention, not the `talker_ar -> code2wav` connection; that stream-hop p95
remained 19.9ms.

### 9.2 Preproc=4

The more aggressive run used the same recipe but with
`PREPROCESSING_MAX_CONCURRENCY=4`.

| Setting | Completed | Failed | Accuracy | Latency Mean | Latency P95 | RTF Mean | RTF P95 | QPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| preproc=4, mr=8 | 43 | 7 | 60.0% | 12.708s | 50.789s | 10.6729 | 43.0363 | 0.569 |

Server logs showed `torch.OutOfMemoryError` on GPU0 while thinker was accepting
multiple video-heavy requests after preprocessing admission was widened.

Conclusion:

- The c=8/c=16 preprocessing lifecycle bottleneck cannot be fixed by blindly
  raising preprocessing parallelism.
- preproc=2 converts part of the queue problem into video/media load and GPU0
  encoder/thinker contention; preproc=4 exceeds safe memory/admission limits.
- The current memory layout requires either tighter thinker admission, lower
  thinker static memory fraction, smaller video batches/frames, a separate
  preprocessing/encoder placement, or a more deliberate preprocessed-request
  scheduler.
- This setting is an anti-recipe for H20 96GB under the current Qwen3.5 serving
  configuration.

## 10. Cold Start And Warmup

Cold compile/capture costs are large and must be separated from warmed
steady-state performance.

Observed SGLang cold/warmup example:

| Run | n | c | Latency Mean | Latency P95 | RTF Mean | RTF P95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| warmup before stress sweep | 8 | 4 | 27.431s | 52.491s | 17.7266 | 46.1566 |

Operational guidance:

- Always run an explicit warmup before measuring steady-state performance.
- For shareable benchmarks, either report cold-start separately or drop the
  first N compile/capture requests with a clear rule.
- The vLLM comparison already uses a warmed skip-first-4 slice for this reason.

Observed vLLM c=8 cold/init costs on 2026-06-19:

| Component | Observed Cost |
| --- | ---: |
| Thinker torch.compile | 67.83s |
| Thinker profile/create KV/cache/graph warmup | 147.88s |
| Talker torch.compile | 64.47s |
| Talker profile/create KV/cache/graph warmup | 203.65s |
| Code2wav warmup batch 1 | 28.94s |
| Code2wav warmup batch 2 | 38.08s |
| Code2wav warmup batch 3-16 | 47-72ms each |

The c=1 vLLM run shows the same shape: thinker compile about 68s, talker compile
about 63s, and code2wav warmup only becomes millisecond-level after the first
two compiled batches. These costs are outside request latency but matter for
deployment warmup and benchmark protocol.

## 11. Reproduction Commands

### 11.1 Launch SGLang

Inside the SGLang container:

```bash
cd /myapp/sglang-omni

NO_CODE2WAV_TORCH_COMPILE=0 \
TORCHDYNAMO_DISABLE=0 \
SGLANG_OMNI_VIDEO_PREPROCESS_CACHE_MAX_BYTES=17179869184 \
SGLANG_OMNI_VIDEO_PREPROCESS_CACHE_MAX_ENTRIES=64 \
EXTRA_ARGS="--thinker-cuda-graph on --talker-cuda-graph on --talker-torch-compile on --thinker-max-running-requests 8 --talker-max-running-requests 8" \
bash examples/launch_qwen35_omni_speech_server_container.sh
```

### 11.2 Run Video-AMME Stress Cases

```bash
cd /myapp/sglang-omni

for C in 1 2 4 8 16; do
  RUN_ID="c${C}_profile_skipwer"
  RUN_ROOT="results/qwen35_sglang_mr8_stress_20260619"
  OUT_DIR="${RUN_ROOT}/benchmark_audio_50_${RUN_ID}"

  curl -s http://127.0.0.1:8161/start_request_profile \
    -H "Content-Type: application/json" \
    -d "{\"run_id\":\"${RUN_ID}\",\"event_dir\":\"/myapp/sglang-omni/${RUN_ROOT}/events\"}"

  HF_HOME=/myapp/data/videoamme \
  HF_DATASETS_CACHE=/myapp/data/videoamme/datasets \
  HF_HUB_OFFLINE=1 \
  python -m benchmarks.eval.benchmark_omni_videoamme \
    --model qwen3_5-omni --port 8161 \
    --repo-id zhaochenyang20/Video_AMME_ci \
    --output-dir "${OUT_DIR}" \
    --max-samples 50 --max-concurrency "${C}" \
    --max-tokens 256 --temperature 0.0 \
    --video-fps 2 --video-max-frames 128 --video-max-pixels 401408 \
    --enable-audio --audio-voice m02 --skip-wer --disable-tqdm

  curl -s http://127.0.0.1:8161/stop_request_profile \
    -H "Content-Type: application/json" \
    -d "{\"run_id\":\"${RUN_ID}\"}"

  python -m sglang_omni.profiler \
    "/myapp/sglang-omni/${RUN_ROOT}/events" \
    --format json \
    --out "/myapp/sglang-omni/${RUN_ROOT}/request_profile_${RUN_ID}.json"
done
```

For the exact c=1 and c=2 artifact names in this checkpoint, use
`c1_warm_profile_skipwer` and `c2_warm_profile_skipwer`.

### 11.3 Regenerate Tail Appendix Tables

```bash
cd /myapp/sglang-omni

RUN_ROOT="results/qwen35_sglang_mr8_stress_20260619"

python -m benchmarks.eval.summarize_omni_tail_profiles \
  --label c8 \
  --result-json "${RUN_ROOT}/benchmark_audio_50_c8_profile_skipwer/videoamme_results.json" \
  --profile-json "${RUN_ROOT}/request_profile_c8_profile_skipwer.json" \
  --top-k 5

python -m benchmarks.eval.summarize_omni_tail_profiles \
  --label c16 \
  --result-json "${RUN_ROOT}/benchmark_audio_50_c16_profile_skipwer/videoamme_results.json" \
  --profile-json "${RUN_ROOT}/request_profile_c16_profile_skipwer.json" \
  --top-k 5
```

Use the c1/c4 paths analogously for the complete appendix. The tool prints the
rank-match quality and the Markdown tail table to stdout.

### 11.4 Run Synthetic Speech Cases

```bash
cd /myapp/sglang-omni

for SCENARIO in short long; do
  for C in 1 4 8; do
    RUN_ID="${SCENARIO}_c${C}_profile"
    RUN_ROOT="results/qwen35_synthetic_speech_20260619"
    OUT_DIR="${RUN_ROOT}/${SCENARIO}_c${C}"
    SAMPLES=16
    if [ "${SCENARIO}" = "long" ]; then SAMPLES=8; fi

    curl -s http://127.0.0.1:8161/start_request_profile \
      -H "Content-Type: application/json" \
      -d "{\"run_id\":\"${RUN_ID}\",\"event_dir\":\"/myapp/sglang-omni/${RUN_ROOT}/events\"}"

    python -m benchmarks.eval.benchmark_qwen35_speech_synthetic \
      --model qwen3_5-omni --port 8161 \
      --scenario "${SCENARIO}" --samples-per-scenario "${SAMPLES}" \
      --output-dir "${OUT_DIR}" \
      --max-concurrency "${C}" \
      --voice m02 --max-tokens 1024 --temperature 0.0 \
      --disable-tqdm

    curl -s http://127.0.0.1:8161/stop_request_profile \
      -H "Content-Type: application/json" \
      -d "{\"run_id\":\"${RUN_ID}\"}"

    python -m sglang_omni.profiler \
      "/myapp/sglang-omni/${RUN_ROOT}/events" \
      --format json \
      --out "/myapp/sglang-omni/${RUN_ROOT}/request_profile_${RUN_ID}.json"
  done
done
```

### 11.5 Offline WER Phase

Run WER after the serving benchmark is complete, so ASR does not contend with
the Qwen3.5 serving stack. If OpenAI Whisper weights are cached inside the
container, the simplest route is local Whisper:

```bash
cd /myapp/sglang-omni

python -m benchmarks.eval.compute_audio_consistency_from_results \
  results/qwen35_sglang_mr8_stress_20260619/benchmark_audio_50_c1_warm_profile_skipwer/videoamme_results.json \
  results/qwen35_sglang_mr8_stress_20260619/benchmark_audio_50_c2_warm_profile_skipwer/videoamme_results.json \
  results/qwen35_sglang_mr8_stress_20260619/benchmark_audio_50_c4_profile_skipwer/videoamme_results.json \
  results/qwen35_sglang_mr8_stress_20260619/benchmark_audio_50_c8_profile_skipwer/videoamme_results.json \
  results/qwen35_sglang_mr8_stress_20260619/benchmark_audio_50_c16_profile_skipwer/videoamme_results.json \
  --path-root /myapp/sglang-omni \
  --local-whisper-model large-v3 \
  --asr-device cuda:1 \
  --output-name whisper_large_v3_local_wer.json \
  --lang en
```

This writes `whisper_large_v3_local_wer.json` next to each input result JSON.
The 2026-06-19 run used `/root/.cache/whisper/large-v3.pt` in
`sglang-omni-dev`.

If an OpenAI-compatible ASR router is preferred, start it first:

```bash
cd /myapp/sglang-omni

python -m sglang_omni.cli serve \
  --model-path openai/whisper-large-v3 \
  --model-name openai/whisper-large-v3 \
  --port 8171
```

Then replace the local Whisper flags with router flags:

```bash
python -m benchmarks.eval.compute_audio_consistency_from_results \
  <result-json> \
  --asr-router-port 8171 \
  --asr-model-path openai/whisper-large-v3 \
  --asr-concurrency 1 \
  --lang en
```

### 11.6 SeedTTS-Compatible Local Smoke

The official SeedTTS cache is not required for the main SGLang-vLLM
Video-AMME comparison. For a local reference-audio TTS smoke using already
cached Video-AMME spoken questions, first generate the compatible `meta.lst`:

```bash
cd /home/gangouyu/sglang-omni

python3 -m benchmarks.eval.build_videoamme_seedtts_meta \
  --output results/qwen35_report_audit_20260619/videoamme_seedtts_meta.lst \
  --summary-output results/qwen35_report_audit_20260619/videoamme_seedtts_meta_summary.json \
  --max-samples 50 \
  --target-mode audio_text
```

Then run the Qwen3.5-Omni SeedTTS-style benchmark inside the same serving
environment:

```bash
cd /myapp/sglang-omni

python -m benchmarks.eval.benchmark_omni_seedtts \
  --generate-only \
  --meta results/qwen35_report_audit_20260619/videoamme_seedtts_meta.lst \
  --output-dir results/qwen35_videoamme_seedtts_smoke_c8 \
  --model qwen3_5-omni \
  --port 8161 \
  --lang en \
  --voice-clone \
  --max-samples 50 \
  --max-concurrency 8 \
  --max-new-tokens 1024 \
  --temperature 0.0 \
  --disable-tqdm
```

This path is intentionally labeled smoke: it exercises the reference-audio
Qwen3.5-Omni speech path with local data, while official SeedTTS full-set
numbers remain a separate follow-up once the dataset is staged.

### 11.7 vLLM Reproduction Notes

Use the local vLLM artifact as the exact baseline record:

```bash
cd /home/gangouyu/sglang-omni
less results/qwen35_vllm_videoamme_ci50_official_talker_compile_c4_20260618_202106/run.log
less results/qwen35_vllm_videoamme_ci50_official_talker_compile_c4_20260618_202106/benchmark_audio_50_c4_offline_compile/vllm_videoamme_report.md
```

For a clean rerun, keep the following properties aligned with the artifact:

- Qwen3.5-capable vLLM image:
  `tongyi-duanwu-registry-vpc.cn-beijing.cr.aliyuncs.com/dashscope/dashllm:cuda129_cp312_test_vl_13589`
- `triton==3.3.1` after container start.
- Thinker/talker `enforce_eager=False`.
- vLLM compile mode and `FULL_AND_PIECEWISE` CUDA graph.
- Prefix caching and chunked prefill enabled.
- `max_num_seqs=4` for the c=4 comparison.
- Same Video-AMME ci-50 dataset and same audio-output scoring path.

The c=1 and c=8 vLLM checkpoints were run from the same local runner with:

```bash
cd /home/gangouyu/sglang-omni

ls -lh \
  results/qwen35_vllm_videoamme_ci50_opt_20260618_162319/run_vllm_videoamme_ci5_offline_compile.sh \
  results/qwen35_vllm_videoamme_ci50_opt_20260618_162319/vllm_videoamme_runner.py

MAX_SAMPLES=50 MAX_CONCURRENCY=1 MAX_NUM_SEQS=8 \
RUN_TAG=ci50_offline_compile_c1_mns8_20260619 \
bash results/qwen35_vllm_videoamme_ci50_opt_20260618_162319/run_vllm_videoamme_ci5_offline_compile.sh

MAX_SAMPLES=50 MAX_CONCURRENCY=8 MAX_NUM_SEQS=8 \
RUN_TAG=ci50_offline_compile_c8_mns8_20260619 \
bash results/qwen35_vllm_videoamme_ci50_opt_20260618_162319/run_vllm_videoamme_ci5_offline_compile.sh
```

For a stricter c=8 engine-side diagnostic rerun that removes serial local prompt
construction from the timed engine window and parallelizes prompt prebuild
across host workers, use the same wrapper with `EXTRA_ARGS`. This does not
overwrite the historical c=8 artifact. The strongest local diagnostic artifact
from 2026-06-20 is
`/home/gangouyu/sglang-omni/results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuildw4_20260620_005346`:

```bash
cd /home/gangouyu/sglang-omni

RUN_ROOT="/home/gangouyu/sglang-omni/results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuildw4_20260620_$(date +%H%M%S)" \
MAX_SAMPLES=50 MAX_CONCURRENCY=8 MAX_NUM_SEQS=8 \
RUN_TAG=ci50_offline_compile_c8_mns8_prebuildw4_20260620 \
EXTRA_ARGS="--prebuild-prompts --prebuild-workers 4" \
bash results/qwen35_vllm_videoamme_ci50_opt_20260618_162319/run_vllm_videoamme_ci5_offline_compile.sh
```

The resulting `vllm_videoamme_report.md` and `videoamme_results.json` include
`prompt_build_wall_s`, `engine_wall_clock_s`, `runner_wall_clock_s`, and
`prebuild_prompts`/`prebuild_workers` in the config. The local w4 artifact
completed 50/50 requests with 66.0% accuracy. Use it to reproduce the prebuild
diagnostic, not as the final strict c=8 serving-throughput win unless it is
rerun with WER/ASR and an online serving ingress path.

To reproduce the offline-runner overhead table:

```bash
cd /home/gangouyu/sglang-omni

python3 -m benchmarks.eval.summarize_vllm_offline_runner_overhead \
  results/qwen35_vllm_videoamme_ci50_official_talker_compile_c4_20260618_202106/benchmark_audio_50_c4_offline_compile/videoamme_results.json \
  results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_20260619_20260619_222434/benchmark_audio_50_c8_offline_compile/videoamme_results.json \
  results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuild_20260620_002020/benchmark_audio_50_c8_offline_compile/videoamme_results.json \
  results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuildw4_20260620_005346/benchmark_audio_50_c8_offline_compile/videoamme_results.json \
  --labels vLLM-c4 vLLM-c8 vLLM-c8-prebuild-w1 vLLM-c8-prebuild-w4
```

To reproduce the log-derived vLLM stage table:

```bash
cd /home/gangouyu/sglang-omni

python3 -m benchmarks.eval.summarize_vllm_omni_log_stages \
  results/qwen35_vllm_videoamme_ci50_offline_compile_c1_mns8_20260619_20260619_220617/run.log \
  results/qwen35_vllm_videoamme_ci50_official_talker_compile_c4_20260618_202106/run.log \
  results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_20260619_20260619_222434/run.log \
  results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuild_20260620_002020/run.log \
  results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuildw4_20260620_005346/run.log \
  --labels vLLM-c1 vLLM-c4 vLLM-c8 vLLM-c8-prebuild-w1 vLLM-c8-prebuild-w4 \
  --skip-first-requests 4 4 8 8 8 \
  --json-output results/qwen35_report_audit_20260619/vllm_log_stage_summary.json
```

To reproduce the vLLM offline admission diagnosis:

```bash
cd /home/gangouyu/sglang-omni

python3 -m benchmarks.eval.diagnose_vllm_offline_admission \
  --case vLLM-c4 \
  results/qwen35_vllm_videoamme_ci50_official_talker_compile_c4_20260618_202106/benchmark_audio_50_c4_offline_compile/videoamme_results.json \
  results/qwen35_vllm_videoamme_ci50_official_talker_compile_c4_20260618_202106/run.log 4 \
  --case vLLM-c8 \
  results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_20260619_20260619_222434/benchmark_audio_50_c8_offline_compile/videoamme_results.json \
  results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_20260619_20260619_222434/run.log 8 \
  --case vLLM-c8-prebuild-w1 \
  results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuild_20260620_002020/benchmark_audio_50_c8_offline_compile/videoamme_results.json \
  results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuild_20260620_002020/run.log 8 \
  --case vLLM-c8-prebuild-w4 \
  results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuildw4_20260620_005346/benchmark_audio_50_c8_offline_compile/videoamme_results.json \
  results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuildw4_20260620_005346/run.log 8 \
  --json-output results/qwen35_report_audit_20260619/vllm_admission_diagnosis.json
```

### 11.8 Regenerate Key Report Tables

For the normal handoff path, run the complete audit pipeline first. It refreshes
the Video-AMME SeedTTS-compatible meta, environment snapshot, vLLM log-stage
summaries, vLLM admission diagnosis, artifact/table JSON, claim-verification
JSON, stage-interaction JSON, headline scorecard JSON, slide-ready SVG/CSV
chart pack, objective requirement crosswalk, stage reproduction drilldown,
stage route decision matrix,
reproduction command manifest, final readiness audit, share bundle
manifest, preflight JSON, requirement coverage matrix, and the final evidence
manifest in the right order:

```bash
cd /home/gangouyu/sglang-omni

python3 -m benchmarks.eval.run_qwen35_omni_report_audit \
  --root /home/gangouyu/sglang-omni \
  --summary-output results/qwen35_report_audit_20260619/audit_run_summary.json
```

`audit_run_summary.json` is a convenience run summary; the hash-stable handoff
inventory remains `manifest.json`.

The following helper checks the expected local artifacts and regenerates the
main Markdown tables for SGLang stress+WER, SGLang stage breakdown, synthetic
speech, synthetic stage breakdown, preproc=2 comparison, vLLM offline runner
overhead, vLLM log-derived stage signals, and vLLM offline admission diagnosis:

```bash
cd /home/gangouyu/sglang-omni

python3 -m benchmarks.eval.summarize_qwen35_omni_report_artifacts \
  --root /home/gangouyu/sglang-omni \
  --json-output results/qwen35_report_audit_20260619/tables_summary.json
```

The current 2026-06-20 audit reports all 45/45 expected artifacts present. Use
`--check-only` when you only want an artifact-presence gate. The JSON output is
the machine-readable form of the regenerated tables.

To verify the main report claims, run:

```bash
cd /home/gangouyu/sglang-omni

python3 -m benchmarks.eval.verify_qwen35_omni_report_claims \
  --root /home/gangouyu/sglang-omni \
  --json-output results/qwen35_report_audit_20260619/claims_verification.json
```

The current 2026-06-20 verifier passes 17/17 checked claims: warmed c=4
SGLang-vLLM latency/RTF/WER, c=8 stress throughput peak, stable stress WER,
stage transition from talker to preprocessing queue, SGLang and vLLM
code2wav/inter-stage non-bottlenecks, vLLM offline prompt-feed/admission
limitation, vLLM c=8 prebuilt-prompt clock separation and admission-span
reduction, vLLM c=8 four-worker prebuild runner-wall improvement, long
synthetic faster-than-real-time output, and the
preproc=2/preproc=4 negative-result conclusions.

To regenerate only the requirement coverage matrix:

```bash
cd /home/gangouyu/sglang-omni

python3 -m benchmarks.eval.summarize_qwen35_report_coverage \
  --root /home/gangouyu/sglang-omni \
  --strict \
  --json-output results/qwen35_report_audit_20260619/coverage_matrix.json
```

To regenerate only the machine-readable stage-interaction summary:

```bash
cd /home/gangouyu/sglang-omni

python3 -m benchmarks.eval.summarize_qwen35_stage_interactions \
  --root /home/gangouyu/sglang-omni \
  --json-output results/qwen35_report_audit_20260619/stage_interaction_summary.json
```

To regenerate only the headline scorecard used for slide/PPT numbers:

```bash
cd /home/gangouyu/sglang-omni

python3 -m benchmarks.eval.build_qwen35_omni_headline_scorecard \
  --root /home/gangouyu/sglang-omni \
  --json-output results/qwen35_report_audit_20260619/headline_scorecard.json
```

To regenerate only the SVG/CSV chart pack for slides and spreadsheets:

```bash
cd /home/gangouyu/sglang-omni

python3 -m benchmarks.eval.build_qwen35_omni_share_charts \
  --root /home/gangouyu/sglang-omni \
  --output-dir results/qwen35_report_audit_20260619/share_charts \
  --manifest-output results/qwen35_report_audit_20260619/share_charts/chart_pack_manifest.json

python3 -m benchmarks.eval.build_qwen35_omni_chart_source_consistency \
  --root /home/gangouyu/sglang-omni \
  --strict \
  --output benchmarks/reports/qwen35_omni_chart_source_consistency_zh_20260621.md \
  --json-output results/qwen35_report_audit_20260619/chart_source_consistency.json
```

To regenerate only the per-regime acceptance matrix:

```bash
cd /home/gangouyu/sglang-omni

python3 -m benchmarks.eval.build_qwen35_omni_acceptance_matrix \
  --root /home/gangouyu/sglang-omni \
  --json-output results/qwen35_report_audit_20260619/acceptance_matrix.json
```

To regenerate only the confidence ledger for safe external wording:

```bash
cd /home/gangouyu/sglang-omni

python3 -m benchmarks.eval.build_qwen35_omni_confidence_ledger \
  --root /home/gangouyu/sglang-omni \
  --json-output results/qwen35_report_audit_20260619/confidence_ledger.json
```

To regenerate only the original-objective completion audit:

```bash
cd /home/gangouyu/sglang-omni

python3 -m benchmarks.eval.build_qwen35_omni_objective_completion_audit \
  --root /home/gangouyu/sglang-omni \
  --strict \
  --json-output results/qwen35_report_audit_20260619/objective_completion_audit.json
```

To regenerate only the original objective requirement crosswalk:

```bash
cd /home/gangouyu/sglang-omni

python3 -m benchmarks.eval.build_qwen35_omni_objective_requirement_crosswalk \
  --root /home/gangouyu/sglang-omni \
  --strict \
  --json-output results/qwen35_report_audit_20260619/objective_requirement_crosswalk.json
```

To regenerate only the machine-readable reproduction command manifest:

```bash
cd /home/gangouyu/sglang-omni

python3 -m benchmarks.eval.build_qwen35_omni_repro_command_manifest \
  --root /home/gangouyu/sglang-omni \
  --strict \
  --json-output results/qwen35_report_audit_20260619/repro_command_manifest.json
```

To regenerate only the final share-readiness audit:

```bash
cd /home/gangouyu/sglang-omni

python3 -m benchmarks.eval.build_qwen35_omni_final_readiness \
  --root /home/gangouyu/sglang-omni \
  --strict \
  --json-output results/qwen35_report_audit_20260619/final_readiness_audit.json
```

To regenerate only the one-page final status summary:

```bash
cd /home/gangouyu/sglang-omni

python3 -m benchmarks.eval.build_qwen35_omni_final_status_summary \
  --root /home/gangouyu/sglang-omni \
  --output benchmarks/reports/qwen35_omni_final_status_summary_zh_20260621.md
```

To regenerate only the reviewer-facing regime decision matrix:

```bash
cd /home/gangouyu/sglang-omni

python3 -m benchmarks.eval.build_qwen35_omni_regime_decision_matrix \
  --root /home/gangouyu/sglang-omni \
  --strict \
  --output benchmarks/reports/qwen35_omni_regime_decision_matrix_zh_20260621.md \
  --json-output results/qwen35_report_audit_20260619/regime_decision_matrix.json
```

To regenerate only the runtime comparison contract:

```bash
cd /home/gangouyu/sglang-omni

python3 -m benchmarks.eval.build_qwen35_omni_runtime_comparison_contract \
  --root /home/gangouyu/sglang-omni \
  --output benchmarks/reports/qwen35_omni_runtime_comparison_contract_zh_20260621.md
```

To regenerate only the SGLang Optimization Lock:

```bash
cd /home/gangouyu/sglang-omni

python3 -m benchmarks.eval.build_qwen35_omni_sglang_optimization_lock \
  --root /home/gangouyu/sglang-omni \
  --strict \
  --output benchmarks/reports/qwen35_omni_sglang_optimization_lock_zh_20260621.md \
  --json-output results/qwen35_report_audit_20260619/sglang_optimization_lock.json
```

To regenerate only the vLLM Optimization Lock:

```bash
cd /home/gangouyu/sglang-omni

python3 -m benchmarks.eval.build_qwen35_omni_vllm_optimization_lock \
  --root /home/gangouyu/sglang-omni \
  --strict \
  --output benchmarks/reports/qwen35_omni_vllm_optimization_lock_zh_20260621.md \
  --json-output results/qwen35_report_audit_20260619/vllm_optimization_lock.json
```

To regenerate only the vLLM Online Parity Protocol:

```bash
cd /home/gangouyu/sglang-omni

python3 -m benchmarks.eval.build_qwen35_omni_vllm_online_parity_protocol \
  --root /home/gangouyu/sglang-omni \
  --strict \
  --output benchmarks/reports/qwen35_omni_vllm_online_parity_protocol_zh_20260621.md \
  --json-output results/qwen35_report_audit_20260619/vllm_online_parity_protocol.json
```

To regenerate only the Runtime Image Contract:

```bash
cd /home/gangouyu/sglang-omni

python3 -m benchmarks.eval.build_qwen35_omni_runtime_image_contract \
  --root /home/gangouyu/sglang-omni \
  --strict \
  --output benchmarks/reports/qwen35_omni_runtime_image_contract_zh_20260621.md \
  --json-output results/qwen35_report_audit_20260619/runtime_image_contract.json
```

To regenerate only the Rerun Acceptance Contract:

```bash
cd /home/gangouyu/sglang-omni

python3 -m benchmarks.eval.build_qwen35_omni_rerun_acceptance_contract \
  --root /home/gangouyu/sglang-omni \
  --strict \
  --output benchmarks/reports/qwen35_omni_rerun_acceptance_contract_zh_20260621.md \
  --json-output results/qwen35_report_audit_20260619/rerun_acceptance_contract.json
```

To regenerate only the stage causal graph:

```bash
cd /home/gangouyu/sglang-omni

python3 -m benchmarks.eval.build_qwen35_omni_stage_causal_graph \
  --root /home/gangouyu/sglang-omni \
  --output benchmarks/reports/qwen35_omni_stage_causal_graph_zh_20260621.md \
  --json-output results/qwen35_report_audit_20260619/stage_causal_graph.json \
  --strict
```

To regenerate only the Stage Reproduction Drilldown:

```bash
cd /home/gangouyu/sglang-omni

python3 -m benchmarks.eval.build_qwen35_omni_stage_reproduction_drilldown \
  --root /home/gangouyu/sglang-omni \
  --strict \
  --output benchmarks/reports/qwen35_omni_stage_reproduction_drilldown_zh_20260621.md \
  --json-output results/qwen35_report_audit_20260619/stage_reproduction_drilldown.json
```

To regenerate only the Stage Route Decision Matrix:

```bash
cd /home/gangouyu/sglang-omni

python3 -m benchmarks.eval.build_qwen35_omni_stage_route_decision_matrix \
  --root /home/gangouyu/sglang-omni \
  --strict \
  --output benchmarks/reports/qwen35_omni_stage_route_decision_matrix_zh_20260621.md \
  --json-output results/qwen35_report_audit_20260619/stage_route_decision_matrix.json
```

To regenerate only the caveat adjudication matrix:

```bash
cd /home/gangouyu/sglang-omni

python3 -m benchmarks.eval.build_qwen35_omni_caveat_adjudication_matrix \
  --root /home/gangouyu/sglang-omni \
  --output benchmarks/reports/qwen35_omni_caveat_adjudication_matrix_zh_20260621.md
```

To regenerate only the external share-bundle manifest:

```bash
cd /home/gangouyu/sglang-omni

python3 -m benchmarks.eval.build_qwen35_omni_share_bundle_manifest \
  --root /home/gangouyu/sglang-omni \
  --strict \
  --json-output results/qwen35_report_audit_20260619/share_bundle_manifest.json
```

To regenerate only the deterministic convenience share tarball:

```bash
cd /home/gangouyu/sglang-omni

python3 -m benchmarks.eval.build_qwen35_omni_share_bundle_package \
  --root /home/gangouyu/sglang-omni \
  --strict \
  --source-manifest results/qwen35_report_audit_20260619/share_bundle_manifest.json \
  --output results/qwen35_report_audit_20260619/qwen35_omni_share_bundle_20260621.tar.gz \
  --json-output results/qwen35_report_audit_20260619/share_bundle_package_manifest.json

sha256sum -c results/qwen35_report_audit_20260619/qwen35_omni_share_bundle_20260621.tar.gz.sha256
```

To regenerate only the environment snapshot:

```bash
cd /home/gangouyu/sglang-omni

python3 -m benchmarks.eval.build_qwen35_omni_environment_snapshot \
  --root /home/gangouyu/sglang-omni \
  --json-output results/qwen35_report_audit_20260619/environment_snapshot.json
```

Before handing the package to another machine or rerunning the full benchmarks,
run the local preflight:

```bash
cd /home/gangouyu/sglang-omni

python3 -m benchmarks.eval.preflight_qwen35_omni_repro \
  --root /home/gangouyu/sglang-omni \
  --json-output results/qwen35_report_audit_20260619/preflight_repro.json
```

The current preflight reports all required reproduction checks passing:
workspace, Qwen3.5 model path, Video-AMME cache, SGLang/vLLM runner scripts,
vLLM `--prebuild-prompts`/`EXTRA_ARGS` support, artifact audit JSONs,
vLLM prebuilt-prompt w1/w4 artifacts, stage-interaction summary, headline
scorecard, acceptance matrix, confidence ledger, reproduction command manifest,
final readiness audit, share bundle manifest, environment snapshot, Chinese share deck outline,
Chinese final status summary, Chinese regime decision matrix and JSON, Chinese runtime comparison contract,
Chinese stage causal graph, Chinese stage latency budget, Chinese caveat adjudication matrix, Chinese final checkpoint watchlist, Chinese final share delivery note, Chinese one-page scorecard,
Chinese requirement evidence map, Chinese pressure matrix, Chinese metric source map,
Chinese stage metric dictionary, Chinese defense Q&A, Chinese optimization playbook,
Chinese reproduction checklist, Chinese external handoff runbook,
Chinese collaborator rerun validation sheet, Video-AMME
SeedTTS-compatible meta files, Docker images, and 8x H20 inventory.
The only current warning is optional:
`/root/.cache/whisper/large-v3.pt` is not present on the host side, so offline
WER needs either the cached Whisper weights inside the serving container or an
ASR router.

To generate a full evidence manifest with file sizes, SHA-256 hashes, git HEAD,
and dirty-worktree status for every report-critical artifact, run:

```bash
cd /home/gangouyu/sglang-omni

python3 -m benchmarks.eval.build_qwen35_omni_report_manifest \
  --root /home/gangouyu/sglang-omni \
  --output results/qwen35_report_audit_20260619/manifest.json
```

The current 2026-06-20 manifest contains 183 records, 181 files, 2 directories, and
0 missing artifacts; the reproducibility gate requires at least 180 records. Treat this file as the handoff inventory for reproducing
or auditing the report evidence set.

## 12. Current Bottleneck Map

| Regime | Primary Limiter | Evidence | Optimization Direction |
| --- | --- | --- | --- |
| Video-AMME c1/c2/c4 | `talker_ar` tail | Talker is top stage; request time correlates with generated speech length | Talker decode efficiency, graph coverage, fewer per-step overheads |
| Video-AMME c8 | Preprocessing admission + talker tail | Preproc stage lifecycle 1.23s, actual compute 0.29s; talker p95 4.42s | Smarter admission, keep c<=8, avoid overfilling thinker |
| Video-AMME c16 | Queueing/saturation | QPS falls from 2.540 to 2.407; RTF mean rises to 4.8489 | Do not use c16 with current recipe; lower admission or shard |
| Synthetic long output | Talker AR compute | Long c8 talker 25.6s avg, code2wav decode 14ms/window | Talker AR optimization, batching policy, chunk cadence |
| Code2wav | Not compute-bound | Decode avg 13-17ms/window; hop p95 <=24ms | Maintain compile path; monitor lifecycle vs actual decode |
| Preprocessing parallelism | Contention/memory-bound if widened naively | preproc=2 reduced QPS 35.4% through media/encoder/thinker contention; preproc=4 caused 7/50 failures and OOM | Keep preproc=1 unless admission/memory placement changes |
| vLLM offline c8 | Original path: prompt construction/feed; prebuild path: engine/workload/talker-side tail | Original c8 overhead is 81.8% of wall time and warmed admission span is 33.3s avg / 44.0s p95; w1 prebuild reduces admission span to 4.44s avg / 5.43s p95; w4 prebuild cuts prompt wall to 129.2s and runner QPS rises to 0.2127, but engine QPS remains 0.5360 and warmed latency is 4.714s mean / 7.563s p95 | Keep original c8 out of strict serving-throughput claims; use w4 as optimized offline diagnostic, then run online ingress plus WER/ASR before claiming strict c8 parity |

### 12.1 Stage Interaction Matrix

This matrix separates stage-local compute from handoff or admission effects. The
main conclusion is that the slowdowns at high concurrency are caused by admission
and shared-resource contention before useful model work, plus talker AR tail
after useful model work. The measured stage boundaries themselves are not the
primary bottleneck in the current SGLang recipe. The machine-readable companion
is
`/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/stage_interaction_summary.json`.

| Boundary | Healthy Evidence | Saturation / Interaction Evidence | Operational Read |
| --- | --- | --- | --- |
| request admission -> preprocessing | c1/c2/c4 preprocessing lifecycle stays below 0.49s avg; actual preprocessing compute stays near 0.27-0.32s avg | c8 lifecycle rises to 1.23s avg while actual compute remains 0.29s; c16 lifecycle rises to 4.40s avg while actual compute remains about 0.30s | This is admission/queue pressure, not raw video preprocessing compute |
| preprocessing -> encoder/thinker | Baseline c8 image/audio encoder compute remains sub-millisecond in profiler spans, and thinker lifecycle is not the top stage | preproc=2 turns queue pressure into contention: media load 357/877ms, image encoder lifecycle 858/1825ms, thinker lifecycle 1004/1648ms | Widening preprocessing without changing placement/admission starves GPU0 paths |
| thinker -> talker | vLLM log-derived thinker-to-talker feed p95 is 1.0ms at c1/c4/c8; SGLang c1/c4 tails are talker work after handoff | Once prompt-feed is removed in vLLM prebuild, remaining c8 tail moves to engine/workload/talker-side behavior | Handoff is not the limiting boundary; optimize talker AR and batching |
| talker -> code2wav | SGLang stream-hop p95 stays about 15-24ms across c1/c2/c4/c8/c16; vLLM original c1/c4/c8 talker-to-code2wav drain p95 stays 16.0-17.5ms | vLLM prebuild w4 exposes a later tail: feed-to-codec p95 509.7ms and talker-to-code2wav drain p95 123.4ms after prompt admission is fixed | SGLang connection is healthy; vLLM prebuild needs engine/talker boundary follow-up before c8 parity claims |
| code2wav collect -> decode | SGLang decode avg stays about 14-17ms/window; vLLM encoder/code2wav boundary p95 remains small in original c1/c4/c8 | `code2wav_window_collect` grows with concurrency because it waits for enough codec chunks from talker | Do not optimize vocoder first; monitor chunk cadence and talker output rhythm |
| offline runner -> vLLM engine admission | c1/c4 engine-side stage slices remain small once admitted | original vLLM c8 warmed admission span is 33.3s avg / 44.0s p95; prebuild w4 cuts it to 4.09s avg / 4.89s p95 but engine QPS remains 0.5360 | Original vLLM c8 is host prompt-feed limited; prebuild w4 is the correct offline diagnostic, not a serving parity result |
| cold compile/capture -> steady state | Warmed SGLang c4 beats warmed vLLM c4 on latency and RTF while preserving WER | vLLM c8 cold path includes about 68s thinker compile, 64s talker compile, and large graph/cache warmup; first batch dominates full-run p95 | Always separate cold-start from warmed steady-state before drawing performance conclusions |

The optimization priority follows this order: keep the current compiled
code2wav path, avoid naive preprocessing parallelism, preserve the c4-c8 serving
window, then spend engineering effort on talker AR efficiency and smarter
preprocessed-request admission.

## 13. Tail Request Appendix

The profiler events do not currently persist the benchmark `sample_id`, so the
tail attribution below uses rank-matching between sorted client-observed
latency and sorted request-profiler total latency. This is safe for bottleneck
classification because the sorted latency distributions match tightly:

| Run | Max Delta | P95 Delta | Mean Delta |
| --- | ---: | ---: | ---: |
| c1 | 11.06ms | 9.40ms | 3.39ms |
| c4 | 10.92ms | 9.43ms | 3.98ms |
| c8 | 15.97ms | 12.62ms | 4.76ms |
| c16 | 18.81ms | 13.89ms | 5.95ms |

`preproc_queue_ms` is computed as preprocessing
`stage_input_received->stage_complete` minus `preprocess_start->preprocess_end`.
`c2w_window_sum_ms` is chunk collection time; `c2w_decode_sum_ms` is the actual
vocoder decode time.

### 13.1 Video-AMME c1 Top Tail Requests

| Rank | Sample | Client Latency | Profiler Total | Audio | RTF | Preproc Life | Preproc Compute | Preproc Queue | Talker | Thinker | C2W Window Sum | C2W Decode Sum | T->C Stream P95 | Dominant |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 014-2 | 2.8834s | 2.8723s | 11.12s | 0.2593 | 284.5ms | 267.1ms | 17.3ms | 1994.7ms | 216.1ms | 966.8ms | 472.2ms | 15.1ms | talker |
| 2 | 002-1 | 2.5356s | 2.5264s | 8.88s | 0.2855 | 299.1ms | 280.7ms | 18.4ms | 1643.0ms | 207.5ms | 775.5ms | 372.0ms | 15.1ms | talker |
| 3 | 002-2 | 2.4671s | 2.4576s | 8.64s | 0.2855 | 294.2ms | 275.7ms | 18.5ms | 1587.8ms | 194.1ms | 746.4ms | 373.3ms | 14.9ms | talker |
| 4 | 001-2 | 2.3310s | 2.3203s | 7.76s | 0.3004 | 309.0ms | 289.7ms | 19.3ms | 1429.3ms | 175.0ms | 674.7ms | 354.1ms | 15.6ms | talker |
| 5 | 017-1 | 1.9380s | 1.9319s | 5.44s | 0.3562 | 297.7ms | 282.5ms | 15.2ms | 1069.3ms | 163.0ms | 468.3ms | 241.1ms | 15.0ms | talker |

### 13.2 Video-AMME c4 Top Tail Requests

| Rank | Sample | Client Latency | Profiler Total | Audio | RTF | Preproc Life | Preproc Compute | Preproc Queue | Talker | Thinker | C2W Window Sum | C2W Decode Sum | T->C Stream P95 | Dominant |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 002-1 | 4.8810s | 4.8702s | 8.80s | 0.5547 | 1136.6ms | 318.2ms | 818.4ms | 3047.4ms | 216.8ms | 1666.9ms | 404.1ms | 19.4ms | talker |
| 2 | 014-2 | 4.3943s | 4.3834s | 10.56s | 0.4161 | 690.4ms | 321.4ms | 369.0ms | 2962.3ms | 284.4ms | 1511.6ms | 521.3ms | 18.2ms | talker |
| 3 | 001-2 | 3.6884s | 3.6781s | 8.08s | 0.4565 | 306.4ms | 291.1ms | 15.3ms | 2603.2ms | 233.3ms | 1567.7ms | 399.9ms | 16.5ms | talker |
| 4 | 002-2 | 3.5649s | 3.5565s | 7.92s | 0.4501 | 289.1ms | 271.5ms | 17.6ms | 2648.0ms | 197.4ms | 1484.7ms | 413.2ms | 17.8ms | talker |
| 5 | 011-1 | 2.8466s | 2.8419s | 3.20s | 0.8896 | 1039.5ms | 299.8ms | 739.7ms | 1062.6ms | 118.4ms | 675.1ms | 151.9ms | 15.3ms | talker |

### 13.3 Video-AMME c8 Top Tail Requests

| Rank | Sample | Client Latency | Profiler Total | Audio | RTF | Preproc Life | Preproc Compute | Preproc Queue | Talker | Thinker | C2W Window Sum | C2W Decode Sum | T->C Stream P95 | Dominant |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 014-2 | 7.0142s | 6.9994s | 10.32s | 0.6797 | 1739.4ms | 294.8ms | 1444.6ms | 4480.4ms | 308.6ms | 2384.9ms | 563.7ms | 19.0ms | talker |
| 2 | 002-2 | 6.8090s | 6.7974s | 8.96s | 0.7599 | 1507.2ms | 275.5ms | 1231.7ms | 4572.9ms | 229.5ms | 1529.1ms | 483.7ms | 24.1ms | talker |
| 3 | 001-2 | 6.1889s | 6.1754s | 8.56s | 0.7230 | 1120.7ms | 320.7ms | 800.1ms | 4360.5ms | 210.1ms | 2617.5ms | 504.1ms | 22.8ms | talker |
| 4 | 002-1 | 5.4418s | 5.4258s | 9.76s | 0.5576 | 256.2ms | 242.7ms | 13.5ms | 4465.1ms | 290.5ms | 2197.8ms | 540.5ms | 11.5ms | talker |
| 5 | 011-1 | 5.2669s | 5.2582s | 6.88s | 0.7655 | 1849.8ms | 253.6ms | 1596.2ms | 2713.6ms | 236.5ms | 1646.7ms | 361.3ms | 18.8ms | talker |

### 13.4 Video-AMME c16 Top Tail Requests

| Rank | Sample | Client Latency | Profiler Total | Audio | RTF | Preproc Life | Preproc Compute | Preproc Queue | Talker | Thinker | C2W Window Sum | C2W Decode Sum | T->C Stream P95 | Dominant |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 014-2 | 10.2184s | 10.2047s | 10.00s | 1.0218 | 5370.6ms | 299.4ms | 5071.3ms | 4036.6ms | 236.3ms | 2077.9ms | 528.4ms | 10.4ms | preproc_queue |
| 2 | 017-1 | 8.7452s | 8.7332s | 5.84s | 1.4975 | 4947.6ms | 351.6ms | 4596.1ms | 2862.8ms | 183.5ms | 1573.1ms | 268.8ms | 20.5ms | preproc_queue |
| 3 | 001-1 | 8.0097s | 7.9977s | 1.12s | 7.1515 | 6872.2ms | 287.2ms | 6585.0ms | 360.3ms | 78.2ms | 173.1ms | 62.7ms | 16.0ms | preproc_queue |
| 4 | 003-1 | 7.6454s | 7.6338s | 1.68s | 4.5509 | 6423.2ms | 298.9ms | 6124.3ms | 505.9ms | 128.4ms | 201.4ms | 83.6ms | 16.1ms | preproc_queue |
| 5 | 002-2 | 7.3902s | 7.3717s | 8.48s | 0.8715 | 2875.1ms | 353.5ms | 2521.7ms | 3717.8ms | 214.1ms | 2189.6ms | 491.0ms | 20.2ms | talker |

Tail conclusion:

- c1/c4 tails are talker dominated. Longer generated audio has higher absolute
  latency, but RTF remains comfortably below 1.0 for the longest examples.
- c8 is still mostly talker dominated, with 0.8-1.6s preprocessing admission
  queue on several top tails. This is the point where queueing starts to matter
  but does not yet dominate the whole run.
- c16 is not a good operating point for this recipe. Several top tails are
  short-output requests whose latency is dominated by preprocessing queue, not
  by useful talker work.
- The `talker_ar -> code2wav` stream hop stays small even inside the tail set
  (roughly 10-24ms p95). The connection between stages is not the bottleneck.
- `code2wav_decode` is materially smaller than `code2wav_window_collect`,
  confirming that the apparent code2wav tail is mostly waiting for enough
  codec chunks, not vocoder compute.

## 14. Remaining Work Before Final 2026-06-21 Version

High-priority:

1. Add official SeedTTS or another natural-speech full-set benchmark to
   complement synthetic long-form prompts. A local Video-AMME spoken-reference
   `meta.lst` smoke path is now generated by
   `benchmarks.eval.build_videoamme_seedtts_meta`, but this container currently
   has no official SeedTTS cache, and HuggingFace access failed with network
   unreachable on 2026-06-19. Pre-stage the dataset before treating SeedTTS as
   a headline benchmark.
2. If pursuing preprocessing fixes, test `PREPROCESSING_MAX_CONCURRENCY=2` only
   together with lower thinker admission, lower thinker memory fraction, or
   separated preprocessing/encoder placement. Plain preproc=2 is already a
   negative result.
3. If time allows, repeat the tail appendix after any revised preprocessing
   placement/admission run to confirm whether c8 queueing can be reduced without
   media/encoder regressions.
4. If strict c=8 cross-runtime throughput is required, keep the measured vLLM
   `--prebuild-prompts --prebuild-workers 4` artifact as the optimized offline
   baseline, then add an online ingress run with WER/ASR and inspect the
   engine/talker boundary tail before making a serving-throughput claim.

Confidence level:

The machine-readable companion is
`/home/gangouyu/sglang-omni/results/qwen35_report_audit_20260619/confidence_ledger.json`.
It currently contains 9 high-confidence claims, 3 medium-confidence boundary
statements, and 0 unsupported claims. Use it to keep external wording aligned
with the measured evidence.

- High: warmed c=4 SGLang-vLLM comparison on Video-AMME ci-50.
- High: SGLang c=1/2/4/8/16 stress shape and c=8 throughput peak under the
  measured recipe.
- High: SGLang stress sweep audio/text consistency; offline Whisper large-v3
  WER remains stable across c=1/2/4/8/16.
- High: code2wav compute is not the current bottleneck.
- High: talker AR dominates long speech and short-answer tails.
- High: naive preprocessing parallelism is unsafe or slower as configured:
  preproc=2 regresses throughput and preproc=4 OOMs.
- High: vLLM c=8 offline prompt-feed diagnosis and four-worker prebuild
  runner-wall improvement.
- Medium: strict vLLM c=8 cross-runtime serving-throughput remains medium until
  online ingress is rerun with WER/ASR.
- Medium: extrapolation to larger Video-AMME/real user traffic until more varied
  long-form and natural-speech workloads are added.
