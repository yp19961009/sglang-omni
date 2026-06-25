# SGLang Omni Benchmarks

Benchmark suite for SGLang Omni, covering performance (latency, throughput, RTF)
and accuracy (WER, MMSU, MMMU, Video-MME, Video-AMME) across supported modality
combinations.

## Directory Structure

```
benchmarks/
├── tasks/          # Per-task logic (tts, audio_understanding, visual_understand, video_understanding)
├── metrics/        # Metric computation (performance, accuracy)
├── dataset/        # Dataset loaders + download helpers
├── benchmarker/    # Framework: runner, data structures, utilities
├── eval/           # Entry-point scripts (one per task × model)
├── reports/        # Checked-in engineering reports for selected validation runs
├── cache/          # (gitignored) dataset caches
└── results/        # (gitignored) evaluation outputs
```

## Quick Start

```bash
# 0. Prepare dataset (once)
python -m benchmarks.dataset.prepare --dataset seedtts

# 1. Start a server on port 8000 (pick one matching the benchmark below)

# S2-Pro — for sections 2a/2b/2c
python -m sglang_omni.cli serve \
    --model-path fishaudio/s2-pro \
    --config examples/configs/s2pro_tts.yaml --port 8000

# Voxtral-4B-TTS — for section 2d (plain TTS, no voice cloning)
python -m sglang_omni.cli serve \
    --model-path mistralai/Voxtral-4B-TTS-2603 --port 8000

# Higgs TTS — for section 2e (voice cloning via references[])
python -m sglang_omni.cli serve \
    --model-path boson-sglang/higgs-audio-v3-tts-4b-base \
    --port 8000

# MOSS-TTS — for section 2f (voice cloning via references[], duration via token_count)
python -m sglang_omni.cli serve \
    --model-path OpenMOSS-Team/MOSS-TTS-v1.5 \
    --config examples/configs/moss_tts.yaml --port 8000

# Qwen3-Omni, speech mode — for section 3 (SeedTTS; multi-GPU)
python -m sglang_omni.cli serve \
    --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct --port 8000

# Qwen3-Omni, text-only mode — for sections 4 (MMSU) and 5 (MMMU)
python -m sglang_omni.cli serve \
    --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct --text-only --port 8000

# 2a. S2-Pro — full pipeline: generate + WER (server needed for phase 1 only)
python -m benchmarks.eval.benchmark_tts_seedtts \
    --meta zhaochenyang20/seed-tts-eval-arrow \
    --model fishaudio/s2-pro --port 8000 \
    --output-dir results/s2pro_en --lang en --max-samples 50 --concurrency 8

# 2b. S2-Pro — generate only (speed metrics, no transcription)
python -m benchmarks.eval.benchmark_tts_seedtts \
    --generate-only --stream \
    --meta zhaochenyang20/seed-tts-eval-arrow \
    --model fishaudio/s2-pro --port 8000 --max-samples 50 --concurrency 8

# 2c. S2-Pro — transcribe only (reuses audio from a prior generate run; no server)
python -m benchmarks.eval.benchmark_tts_seedtts \
    --transcribe-only \
    --meta zhaochenyang20/seed-tts-eval-arrow \
    --model fishaudio/s2-pro \
    --output-dir results/s2pro_en --lang en --device cuda:0

# 2d. Voxtral — full pipeline without voice cloning
python -m benchmarks.eval.benchmark_tts_seedtts \
    --meta zhaochenyang20/seed-tts-eval-arrow \
    --model mistralai/Voxtral-4B-TTS-2603 --port 8000 \
    --max-concurrency 16 \
    --output-dir results/voxtral_en --lang en --max-samples 50 \
    --no-ref-audio --voice cheerful_female

# 2e. Higgs TTS — full pipeline with SeedTTS voice-cloning references
python -m benchmarks.eval.benchmark_tts_seedtts \
    --meta zhaochenyang20/seed-tts-eval-arrow \
    --model boson-sglang/higgs-audio-v3-tts-4b-base --port 8000 \
    --ref-format references \
    --max-concurrency 16 \
    --output-dir results/higgs_tts_en --lang en --max-samples 50

# 2f. MOSS-TTS — full pipeline with SeedTTS voice-cloning references
python -m benchmarks.eval.benchmark_tts_seedtts \
    --meta zhaochenyang20/seed-tts-eval-arrow \
    --model OpenMOSS-Team/MOSS-TTS-v1.5 --port 8000 \
    --ref-format references --token-count auto \
    --max-concurrency 8 \
    --output-dir results/moss_tts_en --lang en --max-samples 50

# 3a. Qwen3-Omni — full pipeline (generate + transcribe)
python -m benchmarks.eval.benchmark_omni_seedtts \
    --meta zhaochenyang20/seed-tts-eval-arrow \
    --output-dir results/qwen3_omni_en \
    --max-concurrency 16 \
    --model qwen3-omni --port 8000 --max-samples 50

# 3b. Qwen3-Omni — generate only (server required; use in CI to split phases)
python -m benchmarks.eval.benchmark_omni_seedtts \
    --generate-only \
    --meta zhaochenyang20/seed-tts-eval-arrow \
    --output-dir results/qwen3_omni_en \
    --max-concurrency 16 \
    --model qwen3-omni --port 8000 --max-samples 50

# 3c. Qwen3-Omni — transcribe only (reuses audio; ASR server on --port)
python -m benchmarks.eval.benchmark_omni_seedtts \
    --transcribe-only \
    --meta zhaochenyang20/seed-tts-eval-arrow \
    --output-dir results/qwen3_omni_en \
    --model qwen3-omni --lang en --port 8000

# 3d. Qwen3.5-Omni — synthetic speech stress (fixed local prompts)
python -m benchmarks.eval.benchmark_qwen35_speech_synthetic \
    --model qwen3_5-omni --port 8000 \
    --scenario long --samples-per-scenario 8 \
    --max-concurrency 4 --voice m02 --max-tokens 1024

# 3e. Offline audio consistency WER from saved result JSON
python -m benchmarks.eval.compute_audio_consistency_from_results \
    results/qwen35_run/videoamme_results.json \
    --asr-router-port 8171 --asr-model-path openai/whisper-large-v3

# Or reuse a local openai-whisper cache, e.g. /root/.cache/whisper/large-v3.pt
python -m benchmarks.eval.compute_audio_consistency_from_results \
    results/qwen35_run/videoamme_results.json \
    --local-whisper-model large-v3 --asr-device cuda:1 \
    --output-name whisper_large_v3_local_wer.json

# 3f. Tail-stage attribution from saved result JSON + request profile JSON
python -m benchmarks.eval.summarize_omni_tail_profiles \
    --label c8 \
    --result-json results/qwen35_run/benchmark_audio_50_c8/videoamme_results.json \
    --profile-json results/qwen35_run/request_profile_c8.json \
    --top-k 5

# 3g. Build a local Video-AMME SeedTTS-compatible reference-audio meta.lst
python3 -m benchmarks.eval.build_videoamme_seedtts_meta \
    --output results/qwen35_report_audit_20260619/videoamme_seedtts_meta.lst \
    --summary-output results/qwen35_report_audit_20260619/videoamme_seedtts_meta_summary.json \
    --max-samples 50 --target-mode audio_text

# 3h. Qwen3.5-Omni report artifact check + key table regeneration
python3 -m benchmarks.eval.summarize_qwen35_omni_report_artifacts \
    --root /home/gangouyu/sglang-omni \
    --json-output results/qwen35_report_audit_20260619/tables_summary.json

# 3i. Qwen3.5-Omni report claim verifier
python3 -m benchmarks.eval.verify_qwen35_omni_report_claims \
    --root /home/gangouyu/sglang-omni \
    --json-output results/qwen35_report_audit_20260619/claims_verification.json

# 3j. vLLM Qwen3.5-Omni log-derived stage summary
python3 -m benchmarks.eval.summarize_vllm_omni_log_stages \
    results/qwen35_vllm_videoamme_ci50_offline_compile_c1_mns8_20260619_20260619_220617/run.log \
    results/qwen35_vllm_videoamme_ci50_official_talker_compile_c4_20260618_202106/run.log \
    results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_20260619_20260619_222434/run.log \
    results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuild_20260620_002020/run.log \
    results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuildw4_20260620_005346/run.log \
    --labels vLLM-c1 vLLM-c4 vLLM-c8 vLLM-c8-prebuild-w1 vLLM-c8-prebuild-w4 \
    --skip-first-requests 4 4 8 8 8 \
    --json-output results/qwen35_report_audit_20260619/vllm_log_stage_summary.json

# 3k. vLLM Qwen3.5-Omni offline admission diagnosis
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

# 3l. vLLM Qwen3.5-Omni c=8 prebuilt-prompt rerun
RUN_ROOT="/home/gangouyu/sglang-omni/results/qwen35_vllm_videoamme_ci50_offline_compile_c8_mns8_prebuildw4_20260620_$(date +%H%M%S)" \
MAX_SAMPLES=50 MAX_CONCURRENCY=8 MAX_NUM_SEQS=8 \
RUN_TAG=ci50_offline_compile_c8_mns8_prebuildw4_20260620 \
EXTRA_ARGS="--prebuild-prompts --prebuild-workers 4" \
bash results/qwen35_vllm_videoamme_ci50_opt_20260618_162319/run_vllm_videoamme_ci5_offline_compile.sh

# 3m. Qwen3.5-Omni stage interaction summary
python3 -m benchmarks.eval.summarize_qwen35_stage_interactions \
    --root /home/gangouyu/sglang-omni \
    --json-output results/qwen35_report_audit_20260619/stage_interaction_summary.json

# 3n. Qwen3.5-Omni headline scorecard
python3 -m benchmarks.eval.build_qwen35_omni_headline_scorecard \
    --root /home/gangouyu/sglang-omni \
    --json-output results/qwen35_report_audit_20260619/headline_scorecard.json

# 3o. Qwen3.5-Omni slide-ready SVG/CSV chart pack
python3 -m benchmarks.eval.build_qwen35_omni_share_charts \
    --root /home/gangouyu/sglang-omni \
    --output-dir results/qwen35_report_audit_20260619/share_charts \
    --manifest-output results/qwen35_report_audit_20260619/share_charts/chart_pack_manifest.json

# 3p. Qwen3.5-Omni per-regime acceptance matrix
python3 -m benchmarks.eval.build_qwen35_omni_acceptance_matrix \
    --root /home/gangouyu/sglang-omni \
    --json-output results/qwen35_report_audit_20260619/acceptance_matrix.json

# 3q. Qwen3.5-Omni confidence ledger
python3 -m benchmarks.eval.build_qwen35_omni_confidence_ledger \
    --root /home/gangouyu/sglang-omni \
    --json-output results/qwen35_report_audit_20260619/confidence_ledger.json

# 3r. Qwen3.5-Omni original-objective completion audit
python3 -m benchmarks.eval.build_qwen35_omni_objective_completion_audit \
    --root /home/gangouyu/sglang-omni \
    --strict \
    --json-output results/qwen35_report_audit_20260619/objective_completion_audit.json

# 3s. Qwen3.5-Omni machine-readable reproduction command manifest
python3 -m benchmarks.eval.build_qwen35_omni_repro_command_manifest \
    --root /home/gangouyu/sglang-omni \
    --strict \
    --json-output results/qwen35_report_audit_20260619/repro_command_manifest.json

# 3t. Qwen3.5-Omni final share-readiness audit
python3 -m benchmarks.eval.build_qwen35_omni_final_readiness \
    --root /home/gangouyu/sglang-omni \
    --strict \
    --json-output results/qwen35_report_audit_20260619/final_readiness_audit.json

# 3u. Qwen3.5-Omni final status summary
python3 -m benchmarks.eval.build_qwen35_omni_final_status_summary \
    --root /home/gangouyu/sglang-omni \
    --output benchmarks/reports/qwen35_omni_final_status_summary_zh_20260621.md

# 3v. Qwen3.5-Omni reviewer-facing regime decision matrix
python3 -m benchmarks.eval.build_qwen35_omni_regime_decision_matrix \
    --root /home/gangouyu/sglang-omni \
    --output benchmarks/reports/qwen35_omni_regime_decision_matrix_zh_20260621.md

# 3w. Qwen3.5-Omni runtime comparison contract
python3 -m benchmarks.eval.build_qwen35_omni_runtime_comparison_contract \
    --root /home/gangouyu/sglang-omni \
    --output benchmarks/reports/qwen35_omni_runtime_comparison_contract_zh_20260621.md

# 3x. Qwen3.5-Omni SGLang optimization lock
python3 -m benchmarks.eval.build_qwen35_omni_sglang_optimization_lock \
    --root /home/gangouyu/sglang-omni \
    --strict \
    --output benchmarks/reports/qwen35_omni_sglang_optimization_lock_zh_20260621.md \
    --json-output results/qwen35_report_audit_20260619/sglang_optimization_lock.json

# 3y. Qwen3.5-Omni vLLM optimization lock
python3 -m benchmarks.eval.build_qwen35_omni_vllm_optimization_lock \
    --root /home/gangouyu/sglang-omni \
    --strict \
    --output benchmarks/reports/qwen35_omni_vllm_optimization_lock_zh_20260621.md \
    --json-output results/qwen35_report_audit_20260619/vllm_optimization_lock.json

# 3z. Qwen3.5-Omni vLLM online parity protocol
python3 -m benchmarks.eval.build_qwen35_omni_vllm_online_parity_protocol \
    --root /home/gangouyu/sglang-omni \
    --strict \
    --output benchmarks/reports/qwen35_omni_vllm_online_parity_protocol_zh_20260621.md \
    --json-output results/qwen35_report_audit_20260619/vllm_online_parity_protocol.json

# 3aa. Qwen3.5-Omni reviewer-facing stage causal graph
python3 -m benchmarks.eval.build_qwen35_omni_stage_causal_graph \
    --root /home/gangouyu/sglang-omni \
    --output benchmarks/reports/qwen35_omni_stage_causal_graph_zh_20260621.md

# 3ab. Qwen3.5-Omni reviewer-facing caveat adjudication matrix
python3 -m benchmarks.eval.build_qwen35_omni_caveat_adjudication_matrix \
    --root /home/gangouyu/sglang-omni \
    --output benchmarks/reports/qwen35_omni_caveat_adjudication_matrix_zh_20260621.md

# 3ac. Qwen3.5-Omni external share-bundle manifest
python3 -m benchmarks.eval.build_qwen35_omni_share_bundle_manifest \
    --root /home/gangouyu/sglang-omni \
    --strict \
    --json-output results/qwen35_report_audit_20260619/share_bundle_manifest.json

# 3ad. Qwen3.5-Omni deterministic convenience share tarball
python3 -m benchmarks.eval.build_qwen35_omni_share_bundle_package \
    --root /home/gangouyu/sglang-omni \
    --strict \
    --source-manifest results/qwen35_report_audit_20260619/share_bundle_manifest.json \
    --output results/qwen35_report_audit_20260619/qwen35_omni_share_bundle_20260621.tar.gz \
    --json-output results/qwen35_report_audit_20260619/share_bundle_package_manifest.json

# 3ae. Qwen3.5-Omni reproduction preflight
python3 -m benchmarks.eval.preflight_qwen35_omni_repro \
    --root /home/gangouyu/sglang-omni \
    --json-output results/qwen35_report_audit_20260619/preflight_repro.json

# 3af. Qwen3.5-Omni requirement coverage matrix
python3 -m benchmarks.eval.summarize_qwen35_report_coverage \
    --root /home/gangouyu/sglang-omni \
    --strict \
    --json-output results/qwen35_report_audit_20260619/coverage_matrix.json

# 3ag. Qwen3.5-Omni reproducibility environment snapshot
python3 -m benchmarks.eval.build_qwen35_omni_environment_snapshot \
    --root /home/gangouyu/sglang-omni \
    --json-output results/qwen35_report_audit_20260619/environment_snapshot.json

# 3ah. Qwen3.5-Omni report evidence manifest
python3 -m benchmarks.eval.build_qwen35_omni_report_manifest \
    --root /home/gangouyu/sglang-omni \
    --output results/qwen35_report_audit_20260619/manifest.json

# 3ai. Qwen3.5-Omni full report audit pipeline
python3 -m benchmarks.eval.run_qwen35_omni_report_audit \
    --root /home/gangouyu/sglang-omni \
    --summary-output results/qwen35_report_audit_20260619/audit_run_summary.json

# 4. Qwen3-Omni — MMSU (audio comprehension)
python -m benchmarks.eval.benchmark_omni_mmsu \
    --model qwen3-omni --port 8000 \
    --modalities text+audio --max-samples 50

# 5. Qwen3-Omni — MMMU (VLM accuracy, image input)
python -m benchmarks.eval.benchmark_omni_mmmu \
    --model qwen3-omni --port 8000 --max-samples 50 --max-concurrency 16

# 6. Qwen3-Omni — Video-MME (video understanding)
python -m benchmarks.eval.benchmark_omni_videomme \
    --model qwen3-omni --port 8000 --max-samples 50

# 7a. Qwen3-Omni — Video-AMME (video + audio question understanding)
python -m benchmarks.eval.benchmark_omni_videoamme \
    --model qwen3-omni --port 8000 \
    --repo-id zhaochenyang20/Video_AMME_ci \
    --max-samples 50 --max-concurrency 16 \
    --video-fps 2 --video-max-frames 128 --video-max-pixels 401408

# 7b. Qwen3-Omni — Video-AMME Talker (text + audio output)
python -m benchmarks.eval.benchmark_omni_videoamme \
    --model qwen3-omni --port 8000 \
    --repo-id zhaochenyang20/Video_AMME_ci \
    --max-samples 50 --max-concurrency 16 \
    --video-fps 2 --video-max-frames 128 --video-max-pixels 401408 \
    --enable-audio --asr-device cuda:0 --asr-concurrency 32
```

## Eval Scripts

| Script | Task | Model | API |
|--------|------|-------|-----|
| `eval/benchmark_tts_seedtts.py` | TTS speed + WER (unified) | e.g. S2-Pro, Voxtral, Higgs TTS | `/v1/audio/speech` |
| `eval/benchmark_omni_seedtts.py` | TTS speed + WER (unified) | Qwen3-Omni | `/v1/chat/completions` |
| `eval/benchmark_qwen35_speech_synthetic.py` | Synthetic speech stress, fixed local prompts | Qwen3.5-Omni | `/v1/chat/completions` |
| `eval/compute_audio_consistency_from_results.py` | Offline text/audio WER from saved benchmark outputs | Any audio-output benchmark with `per_sample` records | `/v1/audio/transcriptions` or local Whisper |
| `eval/summarize_omni_tail_profiles.py` | Offline tail-stage attribution from benchmark results and request profiles | Any Omni benchmark with profiler output | local JSON |
| `eval/summarize_vllm_offline_runner_overhead.py` | Offline wall-time overhead estimate for vLLM Video-AMME runner artifacts | vLLM offline `videoamme_results.json` | local JSON |
| `eval/summarize_vllm_omni_log_stages.py` | Log-derived vLLM-Omni stage, boundary, and batch-admission timing summary | vLLM Qwen3.5 `run.log` | local log |
| `eval/diagnose_vllm_offline_admission.py` | vLLM offline prompt-feed/admission bottleneck classifier | vLLM `videoamme_results.json` + `run.log` | local JSON/log |
| `eval/summarize_qwen35_stage_interactions.py` | Machine-readable stage interaction summary for Qwen3.5-Omni handoff evidence | Local Qwen3.5 report artifacts | local JSON |
| `eval/build_qwen35_omni_headline_scorecard.py` | Headline c=4/c=8/long-speech/vLLM diagnostic scorecard for sharing and slides | Local Qwen3.5 report artifacts | local JSON |
| `eval/build_qwen35_omni_share_charts.py` | Slide-ready SVG/CSV chart pack derived from audited Qwen3.5 report JSONs | Local Qwen3.5 report artifacts | local JSON/CSV/SVG |
| `eval/build_qwen35_omni_acceptance_matrix.py` | Per-pressure acceptance matrix for SGLang stress, synthetic speech, vLLM diagnostics, and anti-recipes | Local Qwen3.5 report artifacts | local JSON |
| `eval/build_qwen35_omni_confidence_ledger.py` | Safe external wording and confidence-boundary ledger for Qwen3.5-Omni claims | Local Qwen3.5 report artifacts | local JSON |
| `eval/build_qwen35_omni_objective_completion_audit.py` | Original-objective completion audit mapping the user request to current evidence, caveats, and active-goal status | Local Qwen3.5 report artifacts | local JSON |
| `eval/build_qwen35_omni_repro_command_manifest.py` | Machine-readable Qwen3.5-Omni full-audit, SGLang, vLLM, table, chart, preflight, coverage, and manifest rerun commands | Local Qwen3.5 report artifacts | local JSON |
| `eval/build_qwen35_omni_final_readiness.py` | Final Qwen3.5-Omni share-readiness audit with send/no-send gates and caveats | Local Qwen3.5 report artifacts | local JSON |
| `eval/build_qwen35_omni_final_status_summary.py` | One-page Chinese final status summary derived from audit, objective, readiness, manifest, and share-package JSONs | Local Qwen3.5 report artifacts | local Markdown |
| `eval/build_qwen35_omni_regime_decision_matrix.py` | Reviewer-facing Chinese matrix mapping each pressure regime to recommendation, bottleneck, caveat, evidence, and action | Local Qwen3.5 report artifacts | local Markdown |
| `eval/build_qwen35_omni_runtime_comparison_contract.py` | Chinese fair-comparison contract separating strict c=4 headline, SGLang scaling, vLLM c=8 offline diagnostics, and invalid parity comparisons | Local Qwen3.5 report artifacts | local Markdown |
| `eval/build_qwen35_omni_sglang_optimization_lock.py` | Chinese SGLang optimization lock matrix plus JSON gate for image, compiled/graph recipe, c=8 peak, stage handoff, and anti-recipe evidence | Local Qwen3.5 report artifacts | local Markdown/JSON |
| `eval/build_qwen35_omni_vllm_optimization_lock.py` | Chinese vLLM optimization lock matrix plus JSON gate for image, compile, CUDA graph, cache, and prebuild evidence | Local Qwen3.5 report artifacts | local Markdown/JSON |
| `eval/build_qwen35_omni_vllm_online_parity_protocol.py` | Chinese vLLM c=8 online-parity upgrade protocol plus JSON gate for online ingress, WER/ASR, stage profile, and replacement thresholds | Local Qwen3.5 report artifacts | local Markdown/JSON |
| `eval/build_qwen35_omni_stage_causal_graph.py` | Chinese stage-causal graph tying admission, talker cadence, stream hop, code2wav collect/decode, and vLLM offline admission to bottleneck movement | Local Qwen3.5 report artifacts | local Markdown |
| `eval/build_qwen35_omni_caveat_adjudication_matrix.py` | Chinese caveat adjudication matrix classifying shareable caveats, forbidden wording, rerun-triggered upgrades, and number-replacement rules | Local Qwen3.5 report artifacts | local Markdown |
| `eval/build_qwen35_omni_share_bundle_manifest.py` | External share-bundle manifest with recommended reports, machine evidence, chart assets, file sizes, and SHA-256 hashes | Local Qwen3.5 report artifacts | local JSON |
| `eval/build_qwen35_omni_share_bundle_package.py` | Deterministic convenience tarball and checksum from the external share-bundle manifest | Local Qwen3.5 report artifacts | local tar/JSON |
| `eval/preflight_qwen35_omni_repro.py` | Local readiness checks before reproducing the Qwen3.5-Omni report | Local model/data/images/artifacts | local filesystem |
| `eval/summarize_qwen35_report_coverage.py` | User-requirement coverage matrix for the Qwen3.5-Omni handoff report | Local report audit JSONs | local JSON |
| `eval/build_qwen35_omni_environment_snapshot.py` | Reproducibility environment snapshot with GPU, Docker image, git, path, and audit state | Local runtime environment | local JSON |
| `eval/summarize_qwen35_omni_report_artifacts.py` | Qwen3.5-Omni report artifact check and key table regeneration | Local Qwen3.5 report artifacts | local JSON |
| `eval/verify_qwen35_omni_report_claims.py` | Qwen3.5-Omni report claim verifier | Local Qwen3.5 report artifacts | local JSON |
| `eval/build_qwen35_omni_report_manifest.py` | Qwen3.5-Omni report evidence manifest with hashes and git status | Local Qwen3.5 report artifacts | local JSON |
| `eval/run_qwen35_omni_report_audit.py` | One-command Qwen3.5-Omni report audit pipeline | Local Qwen3.5 report artifacts | local JSON/log |
| `eval/benchmark_omni_mmsu.py` | MMSU (audio comprehension) | Qwen3-Omni | `/v1/chat/completions` |
| `eval/benchmark_omni_mmmu.py` | MMMU (VLM accuracy + speed) | Qwen3-Omni | `/v1/chat/completions` |
| `eval/benchmark_omni_videomme.py` | Video-MME (video understanding) | Qwen3-Omni | `/v1/chat/completions` |
| `eval/benchmark_omni_videoamme.py` | Video-AMME (video + audio question understanding) | Qwen3-Omni | `/v1/chat/completions` |

Selected engineering validation reports live under `benchmarks/reports/`. These
reports document local reproduction details, metrics, and bottleneck findings
for larger model bring-up or performance investigations.

| Report | Purpose |
| --- | --- |
| `reports/qwen35_omni_final_share_delivery_note_zh_20260621.md` | Chinese final share delivery note with the files to send, machine evidence, safe claims, caveats, recipient first steps, and sender self-check. |
| `reports/qwen35_omni_one_page_scorecard_zh_20260621.md` | Chinese one-page core-number scorecard for headline comparison, SGLang pressure sweep, short/long speech, stage health, vLLM c8 diagnostic, and gates. |
| `reports/qwen35_omni_share_package_index_zh_20260621.md` | Chinese share-package index with reading order, safe claims, evidence entry points, and defense-question routing. |
| `reports/qwen35_omni_collaboration_brief_zh_20260621.md` | Chinese collaborator-facing brief with the headline result, bottleneck map, and reproduction entry points. |
| `reports/qwen35_omni_share_deck_outline_zh_20260621.md` | Chinese 15-25 minute sharing/PPT outline mapped to headline metrics, stage evidence, caveats, and reproduction gates. |
| `reports/qwen35_omni_requirement_evidence_map_zh_20260621.md` | Chinese original-requirement evidence map tying each user objective to local evidence, confidence boundaries, and reproduction entry points. |
| `reports/qwen35_omni_pressure_matrix_zh_20260621.md` | Chinese pressure-condition matrix summarizing measured regimes, recommendation status, bottlenecks, and evidence paths. |
| `reports/qwen35_omni_metric_source_map_zh_20260621.md` | Chinese metric source map tying headline, pressure, stage, vLLM diagnostic, and anti-recipe numbers to machine evidence and regeneration commands. |
| `reports/qwen35_omni_stage_metric_dictionary_zh_20260621.md` | Chinese stage metric dictionary defining lifecycle, compute, handoff, collect-wait, and vLLM admission metrics for correct breakdown interpretation. |
| `reports/qwen35_omni_defense_qna_zh_20260621.md` | Chinese defense Q&A with ready-to-say answers, unsafe wording boundaries, and evidence links for collaborator questions. |
| `reports/qwen35_omni_optimization_playbook_zh_20260621.md` | Chinese optimization playbook mapping measured bottlenecks to safe knobs, experiment order, acceptance gates, and rollback rules. |
| `reports/qwen35_omni_reproduction_checklist_zh_20260621.md` | Chinese step-by-step reproduction checklist with SGLang/vLLM rerun commands, expected shapes, and acceptance gates. |
| `reports/qwen35_omni_external_handoff_runbook_zh_20260621.md` | Chinese external handoff runbook with the shortest audited reviewer path, rerun order, stage-reading rules, and replacement gates. |
| `reports/qwen35_omni_collaborator_rerun_validation_sheet_zh_20260621.md` | Chinese collaborator rerun validation sheet for environment deltas, SGLang/vLLM rerun checks, stage flags, replacement rules, and return artifacts. |
| `reports/qwen35_omni_sglang_optimization_lock_zh_20260621.md` | Chinese SGLang optimization lock matrix proving the current best recipe uses the intended image, compiled/graph path, c=8 peak, healthy stage handoff, and measured anti-recipes. |
| `reports/qwen35_omni_vllm_optimization_lock_zh_20260621.md` | Chinese vLLM optimization lock matrix proving the baseline uses the intended image, compile/CUDA graph/cache path, and prebuild w4 diagnostic boundary. |
| `reports/qwen35_omni_vllm_online_parity_protocol_zh_20260621.md` | Chinese vLLM c=8 online-parity upgrade protocol that keeps current prebuild w4 evidence diagnostic-only and defines the artifacts/gates required before replacing headline numbers. |
| `reports/qwen35_omni_stage_causal_graph_zh_20260621.md` | Chinese stage-causal graph showing stage-to-stage bottleneck movement, healthy handoff, code2wav non-bottleneck evidence, and vLLM offline admission boundaries. |
| `reports/qwen35_omni_caveat_adjudication_matrix_zh_20260621.md` | Chinese caveat adjudication matrix for safe external wording, forbidden claims, upgrade reruns, and report-number replacement triggers. |
| `reports/qwen35_omni_stress_performance_plan_20260621.md` | Full Qwen3.5-Omni SGLang/vLLM stress-performance report with stage breakdowns, audit gates, and exact reproduction commands. |

Slide-ready SVG/CSV assets for the Qwen3.5-Omni share package are generated
under `results/qwen35_report_audit_20260619/share_charts/`; the audited
manifest is `share_charts/chart_pack_manifest.json`.

The two `*_seedtts.py` scripts merge the previous `benchmark_*_tts_speed.py`
and `voice_clone_*_wer.py` pairs into a single two-phase pipeline: phase 1
generates + persists WAVs while the TTS server runs, phase 2 transcribes through
an ASR server to avoid GPU contention with the TTS server. Use `--generate-only` or
`--transcribe-only` to run a single phase. For TTS, `--concurrency` and
`--max-concurrency` are equivalent (see `benchmark_tts_seedtts.py`).
`benchmark_tts_seedtts.py` also handles model-specific voice-cloning reference
payloads: the default `--ref-format flat` sends `ref_audio`/`ref_text`, while
`--ref-format references` sends `references=[{audio_path, text}]` for Higgs TTS
and MOSS-TTS. MOSS-TTS additionally supports duration control through
`--token-count`.
`benchmark_omni_seedtts.py` documents local vs CI GPU usage in its module
docstring (sequential phases on CI to reduce OOM risk).

## Adding a New Model or Task

- **New model, same task/API type** (e.g. another OAI-compatible TTS model):
  add an eval script under `eval/` that reuses the existing task helpers
  in `tasks/tts.py` (`make_tts_send_fn`, `run_seedtts_transcribe`, …).
- **New task or API type**: add a task class in the relevant `tasks/*.py`
  file (mirroring `VoiceCloneOmni` in `tasks/tts.py`), expose metric
  helpers, and wire it into a new eval script.

## Datasets

Download helpers live in `benchmarks/dataset/prepare.py`:

```bash
python -m benchmarks.dataset.prepare --dataset seedtts       # full SeedTTS
python -m benchmarks.dataset.prepare --dataset seedtts-mini  # smoke-test subset
python -m benchmarks.dataset.prepare --dataset seedtts-50    # 50-sample subset
python -m benchmarks.dataset.prepare --dataset mmmu          # full MMMU (30 subjects)
python -m benchmarks.dataset.prepare --dataset mmmu-ci-50    # MMMU CI subset
python -m benchmarks.dataset.prepare --dataset mmsu          # full MMSU (ddwang2000/MMSU)
python -m benchmarks.dataset.prepare --dataset videomme-ci-50  # Video-MME CI subset
python -m benchmarks.dataset.prepare --dataset videomme      # full Video-MME
python -m benchmarks.dataset.prepare --dataset videoamme-ci-50  # Video-AMME CI subset
```

All datasets are pre-warmed into the default HuggingFace cache via
`datasets.load_dataset(repo_id)`.  SeedTTS Arrow repos stage audio to
process-local tempfiles at load time; no manual `--local-dir` step is needed.

Video-AMME is generated from the Video-MME CI subset by moving the
question/options/instruction into per-sample WAV files. The benchmark request
text only contains routing/format instructions; the actual question content
stays in the dataset WAV files.
