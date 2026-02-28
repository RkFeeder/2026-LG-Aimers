# 2026 LG Aimers – Exaone Model Quantization

## Overview
LG Aimers is an AI boot camp hosted by LG centered around their in-house LLM, **Exaone (Expert AI for Everyone)**. The program emphasizes not just model performance, but efficiency — making capable AI accessible on low-resource hardware. The program concluded with an online hackathon in February 2026.

## Objective
Quantize the **Exaone 4.0 1.2B** baseline model (available on HuggingFace) while maintaining acceptable performance. Submissions were scored based on the ratio of performance to model size relative to the baseline. A score of **0.5 or above** was required to complete the program.

## Approaches

### 1. INT4 Quantization via LLM Compressor
- Applied INT4 quantization using the `llm-compressor` library
- Achieved approximately **75% size reduction**
- Performance degraded significantly — **Score: 0.15**

### 2. FP16 Conversion
- Converted model weights to FP16 (half-precision floating point)
- Simpler method, but more stable performance retention
- **Score: 0.51** — passed the program threshold

## Results

| Method | Score | Size Reduction | Outcome |
|---|---|---|---|
| INT4 Quantization (LLM Compressor) | 0.15 | ~75% | Failed threshold |
| FP16 Conversion | 0.51 | Moderate | Passed |

## What I Learned
Aggressively quantizing a model without first establishing a stable baseline leads to compounded errors that are hard to debug under submission limits. A better strategy is to start with conservative methods (FP16) and incrementally push further — not the other way around. I also gained hands-on experience with LLM compression tradeoffs: size reduction does not come free, and hardware constraints directly shape what approaches are viable.

## Environment
- GPU: RTX 2060 Mobile
- OS: Windows (attempted Ubuntu/vLLM for local testing, encountered compatibility issues)
- Submission limit: 3 per day

## Repository Structure
- `exaone_baseline/` — Baseline model provided by LG (Exaone 4.0 1.2B)
- `exaone_optimized/` — Final submission via FP16 conversion (Score: 0.51)
- `quantize.py` — Quantization script
