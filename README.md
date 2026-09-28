# A Matched-Budget Audit Framework for Recaptioned Image-Text Supervision Distributions

[![Paper](https://img.shields.io/badge/OpenReview-JobgYHJvPo-8c1b13)](https://openreview.net/forum?id=JobgYHJvPo)
[![Project page](https://img.shields.io/badge/Project-page-1f6feb)](https://bootsoflagrangian.github.io/audit-recap-t2i.github.io/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Recaptioned%20Image%20Text-ffcc4d)](https://huggingface.co/collections/BootsofLagrangian/recaptioned-image-text-6ab5bf978e69805c3e33d5ce)
[![Code license](https://img.shields.io/badge/code-Apache--2.0-blue)](LICENSE)

Giyeong Oh, Junghun Park, Yuhan Bae, Youngjae Yu (Seoul National University).
NeurIPS 2026, Evaluations and Datasets Track (poster).

A recaptioned image-text corpus is a supervision distribution induced by a captioning policy, a
captioner, and a source corpus. This repository contains the audit framework that compares such
distributions at a matched caption budget: every caption is cut to the same window of B words, the
window is scored with deterministic text diagnostics, the claims it states are extracted as
controllable basic units (CBUs), and each CBU is re-asked against the paired image by two
independent VLM judges. It also contains the caption-generation runner and prompts behind the
released corpus of Qwen3.5 recaptions for nine public image collections, the tool used for the
human calibration study, and the result summaries behind every table and figure of the paper.

- Paper: <https://openreview.net/forum?id=JobgYHJvPo>
- Project page: <https://bootsoflagrangian.github.io/audit-recap-t2i.github.io/>
- Released captions: [Recaptioned Image Text collection on Hugging Face](https://huggingface.co/collections/BootsofLagrangian/recaptioned-image-text-6ab5bf978e69805c3e33d5ce)

## Contents

- [What the audit measures](#what-the-audit-measures)
- [Key results](#key-results)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Hardware and model serving](#hardware-and-model-serving)
- [Data access](#data-access)
- [Reproducing the paper](#reproducing-the-paper)
- [Results directory](#results-directory)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

## What the audit measures

**Audit target.** A captioner V_c run under policy π on a source corpus C induces the paired
distribution D = D_{π,V_c,C} = {(c, x) : x ∈ C, c ~ V_c(x; π)} with caption marginal D_c. Each
released caption set over the same source rows is a *surface*: ours, a public reference release, or
the naive-policy control. The audit compares surfaces that share source rows.

**Matched budget B.** Every surface is read through the same window. The claim extractor receives the
first B whitespace-delimited words of each caption (B = 64 unless stated otherwise; the CC12M sweep
uses B ∈ {16, 32, 48, 64}). Lengths and densities are counted in *lexical units* (lex): regex word
units `[^\W_]+(?:'[^\W_]+)*` after Unicode normalization, not tokenizer tokens.

**Controllable basic units.** A CBU is one atomic visual claim of one of eight semantic visual-claim
types: object, attribute, relation, count, style, camera, lighting, text rendering. The extractor φ
(`Qwen/Qwen3.5-397B-A17B-FP8`, text only) returns the CBUs a caption window states, under a guided
JSON schema with fields `category`, `unit`, `span`, `target`. Each CBU then becomes one yes/no
question, answered from the image alone by a judge V_J:

```
Is the visual claim '<TARGET>: <UNIT>' supported by the image?
Is the visual claim '<UNIT>' supported by the image?
Is the rendered text claim '<UNIT>' visibly supported by the image?
```

Two judges answer the identical question set: the **Qwen Judge** (`Qwen/Qwen3.5-397B-A17B-FP8`, the
extractor's checkpoint) and the **Gemma Judge** (`google/gemma-4-31B-it`, an independently trained
family). Each answer is `yes`, `no`, or `uncertain`. Extraction and judging use temperature 0 and
schema-constrained JSON through vLLM structured outputs.

**Metrics.** With c_{≤B} the budget window of caption c, s(c, x) the number of CBUs of c the judge
answers `yes` for, and u(c, x) the number it answers `no` for:

| Metric | Definition |
|---|---|
| CBU/cap | CBU/cap(D_c, B) = E_c[ \|φ(c_{≤B})\| ], deduplicated claimed CBUs per caption |
| CBU/100 lex | 100 · E_c[ \|φ(c_{≤B})\| ] / E_c[ min(\|c\|, B) ]; computed as 100 × Σ deduplicated claimed CBUs / Σ retained lexical units, so a caption shorter than B contributes its own length |
| Supported CBU/cap | E[ s(c, x) ] = `yes` answers / audited captions |
| Risk ρ | E[ u(c, x) ] / E[ \|φ(c)\| ] = `no` answers / CBU questions |
| Uncertainty | `uncertain` answers / CBU questions |
| B-eligibility | share of captions that reach B lexical units |
| Prompt-mass support | share of a prompt pool's n-gram mass whose n-grams appear in the captions (bigrams at B = 64 in the paper) |
| n-gram JSD | Jensen-Shannon divergence between the caption and prompt-pool n-gram distributions |
| Pool-wins | number of the seven prompt pools on which ours has higher prompt-mass support than the reference |
| Opener rate | share of captions that open with a third-person caption frame (`The image shows…`, `In this photo…`, `We can see…`), matched by the regex catalog `J_meta_statement` in `scripts/vllm/polishing_check.py` |
| Top-100 raw / content prefix mass | share of total prefix mass held by the 100 most frequent prefixes, raw or after stripping a leading sentence matched by the opener catalog |
| Distinct-3 | unique 3-grams / all 3-grams over the corpus, after lowercasing and Unicode normalization |

**Five axes.** The audit reports the metrics jointly; no axis is collapsed into another.

| Axis (desideratum) | Reads | Failure mode | Metric | Boundary |
|---|---|---|---|---|
| Text budget (coverage) | D_c | too little text inside B | Avg lex; B-eligibility | length is a prerequisite, not quality |
| Prompt-pool support (coverage) | D_c vs. prompt pools | caption register absent from prompts | prompt-mass support ↑; n-gram JSD ↓ | pool-conditioned, not universal intent |
| Claimed density (coverage) | D_c | few controllable claims in the window | CBU/cap ↑, CBU/100 lex, per-type counts | caption-only count |
| Surface concentration (health) | D_c | window spent on repeated form | top-100 prefix mass ↓, Distinct-3 ↑, rep-4 ↓ | surface artifact, not faithfulness |
| Support and risk (faithfulness) | D_cx | claims absent from the image | E[s] ↑, E[u] ↓, ρ ↓ under two judges | judge-conditional proxy |

Uncertainty is reported as mean ± standard deviation over bootstrap resamples: 2,000 caption-level
resamples (seed 0) for the VQA cells and 10,000 image-cluster resamples (seed 1477) for the
judge–human agreement.

Caption-only axes use 50k paired rows per slice (the phenomenon descriptors use about 1M captions per
surface); the image-conditioned axis uses about 5,000 captions per surface (4,494 aligned images on
CC12M). Prompt-pool support uses seven public prompt pools, each sampled at 250,000 records.

## Key results

All values are at B = 64 and are read from the files in [`results/`](results/README.md). Supported
CBU/cap and risk are mean ± std over 2,000 caption-level bootstrap resamples (seed 0), computed on the
requests that both judges answered (`results/cbu_vqa_by_category_b64.json`,
`results/tables/vqa_mean_std.tex`). Claimed CBU/cap and CBU/100 lex count the claims of the
Qwen3.5-397B-A17B-FP8 extractor over every caption with a valid extractor response.

**Cross-corpus pairs (reference → ours).**

| Pair | Claimed CBU/cap | Qwen Judge Sup. CBU/cap | Qwen Judge risk | Gemma Judge Sup. CBU/cap | Gemma Judge risk |
|---|---|---|---|---|---|
| DataComp | 10.44 → 14.45 | 8.49 ± 0.05 → 13.73 ± 0.05 | 0.177 ± 0.003 → 0.035 ± 0.001 | 8.00 ± 0.05 → 12.94 ± 0.05 | 0.219 ± 0.003 → 0.081 ± 0.001 |
| LAION-pop | 11.91 → 14.82 | 10.80 ± 0.04 → 14.22 ± 0.05 | 0.077 ± 0.001 → 0.031 ± 0.001 | 10.22 ± 0.04 → 13.61 ± 0.05 | 0.113 ± 0.002 → 0.060 ± 0.001 |
| PD12M | 9.78 → 15.02 | 8.61 ± 0.05 → 14.29 ± 0.05 | 0.103 ± 0.002 → 0.034 ± 0.001 | 8.23 ± 0.05 → 13.53 ± 0.05 | 0.131 ± 0.002 → 0.066 ± 0.001 |
| Danbooru | 8.18 → 14.33 | 6.38 ± 0.04 → 12.74 ± 0.05 | 0.217 ± 0.003 → 0.058 ± 0.001 | 6.15 ± 0.04 → 11.85 ± 0.04 | 0.235 ± 0.003 → 0.094 ± 0.001 |

Across the four pairs, ours raises claimed CBU/cap by +2.91 to +6.14; under both judges it raises
supported CBU/cap by +3.39 to +6.36 and lowers risk by 0.046 to 0.159. Without count and relation
claims, supported CBU/cap still rises by +2.66 to +5.22 and risk is lower in all eight (pair, judge)
cells (`results/tables/excl_count_relation.tex`).

**Captioning-policy control.** The same captioner (`Qwen/Qwen3.5-35B-A3B-FP8`) with the same decoding
captions the same images under the naive single-message prompt; claims are extracted by
Qwen3.5-397B-A17B-FP8 and both judges answer the same questions (`results/tables/policy_control.tex`).

| Family | Surface | Claimed CBU/cap | CBU/100 lex | Qwen Judge Sup. CBU/cap / risk | Gemma Judge Sup. CBU/cap / risk |
|---|---|---:|---:|---|---|
| CC12M (4,494 images) | Ours | 15.21 | 23.16 | 14.60 ± 0.06 / 0.030 ± 0.001 | 13.82 ± 0.06 / 0.066 ± 0.001 |
| | Naive | 11.32 | 17.26 | 11.03 ± 0.05 / 0.022 ± 0.001 | 10.61 ± 0.05 / 0.045 ± 0.001 |
| | Naive (greedy) | 11.43 | 17.45 | 11.10 ± 0.05 / 0.022 ± 0.001 | 10.79 ± 0.05 / 0.043 ± 0.001 |
| DataComp (4,775 images) | Ours | 14.60 | 22.23 | 13.90 ± 0.06 / 0.036 ± 0.001 | 13.07 ± 0.05 / 0.080 ± 0.001 |
| | Naive | 10.95 | 16.66 | 10.56 ± 0.04 / 0.026 ± 0.001 | 10.07 ± 0.04 / 0.057 ± 0.001 |
| | Naive (greedy) | 10.84 | 16.53 | 10.53 ± 0.04 / 0.021 ± 0.001 | 10.11 ± 0.04 / 0.049 ± 0.001 |

`Naive` uses the decoding of the released captions (temperature 1.0, top_k 20, top_p 0.95);
`Naive (greedy)` decodes at temperature 0. Against `Naive`, the released policy raises claimed CBU/cap
by +3.7 to +3.9 and supported CBU/cap by +3.0 to +3.6 under both judges, with risk within 0.03 of the
naive surface. Greedy decoding gives the same reading: against `Naive (greedy)` (claimed 11.43 on
CC12M, 10.84 on DataComp), claimed CBU/cap rises by +3.8, supported CBU/cap by +3.0 to +3.5, and risk
stays within 0.031.

**Lexical-window sensitivity (CC12M).** Cutting the extractor window at 64 lexical units instead of 64
whitespace words moves claimed CBU/cap by at most 3.0% (Ours 15.21 → 14.75) and CBU/100 lex by at most
0.14, and keeps the order of the four CC12M surfaces
(`results/sensitivity/cc12m_lexical_window_claimed_cbu_summary.json`).

## Repository layout

```
configs/
  recap/                     caption policy and vLLM serving configs
    base.yaml                shared runner defaults
    domains/                 photorealistic and anime_booru caption prompts (paper appendix)
    vllm_serve*.yaml         captioner, Qwen Judge and Gemma Judge server configs
  caption_survey/
    fair_slices.json         the seven paired comparisons: inputs, join keys, manifests
    surfaces.json            registry of every surface: Hugging Face id, caption column, captioner
  eval/human_cbu_cc12m.yaml  human-study configuration
scripts/
  vllm/                      vLLM environment setup, servers, caption runner, opener catalog
  build_*, run_*, summarize_*, export_*, caption_*, ...   audit pipeline (see below)
  human_cbu_eval.py          human-study CLI
  paper/                     table and figure generators, sensitivity helpers
src/audit_recap_t2i/
  recap/                     booru tag grounding used by the anime caption policy
  human_cbu/                 human-study sampler, store, web UI, metrics, export
tests/                       tests for the human-study tool
results/                     result summaries behind the paper's tables and figures
```

The scripts keep the names used in the paper appendix. They read and write relative to the
repository root, so run them from there. Inputs live under `data/`, caption-runner outputs under
`outputs/`, and request/response files of the VLM stages under `artifacts/`; all three are ignored by
git.

## Installation

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.11 or newer; `uv.lock` pins every
dependency.

```bash
git clone https://github.com/BootsofLagrangian/audit-recap-t2i.git
cd audit-recap-t2i
uv sync                       # text diagnostics, request/response pipeline, figures
uv sync --extra eval          # + encoder probes (PyTorch, Transformers, sentence-transformers, FlagEmbedding)
uv sync --extra dev           # + pytest for the human-study tests
```

Run scripts with `uv run python scripts/<name>.py`. Optional environment variables (for example a
Hugging Face access token for gated downloads) can be kept in a `.env` file at the repository root;
the shell drivers source it when it exists.

## Hardware and model serving

All VLM stages talk to OpenAI-compatible endpoints served by [vLLM](https://github.com/vllm-project/vllm).
The serving environment is separate from the project environment:

```bash
bash scripts/vllm/setup_tmp_env.sh /tmp/vllm-env-qwen35
```

This creates a uv virtual environment with `vllm==0.20.1`, builds DeepGEMM from a pinned commit
(`scripts/vllm/install_deepgemm.sh`), and downloads `Qwen/Qwen3.5-35B-A3B-FP8`. The serve scripts look
for the environment at `$VLLM_VENV` (default `/tmp/vllm-env-qwen35`) and use `HF_HOME` (default
`~/.cache/huggingface`). A `vllm` extra (`uv sync --extra vllm`) is also available if you prefer to
serve from the project environment. The released caption files record the vLLM version used for
generation in each dataset's `generation_config.yaml`.

The paper ran on NVIDIA H200 GPUs (141 GB).

| Role | Model | Config | Launch | Layout |
|---|---|---|---|---|
| Captioner V_c | `Qwen/Qwen3.5-35B-A3B-FP8` | `configs/recap/vllm_serve.yaml` | `bash scripts/vllm/serve.sh` | 8 GPUs, data parallel 8, thinking off |
| Captioner, 2-GPU variant (naive control) | `Qwen/Qwen3.5-35B-A3B-FP8` | `configs/recap/vllm_serve_qwen35_dp2_tmp.yaml` | `bash scripts/vllm/serve_qwen35_dp2_tmp.sh start` | 2 GPUs, data parallel 2 |
| Extractor φ and Qwen Judge | `Qwen/Qwen3.5-397B-A17B-FP8` | `configs/recap/vllm_serve_397b_fp8.yaml` | `bash scripts/vllm/serve_397b.sh start` | 8 GPUs, tensor parallel 8, FP8 KV cache |
| Extractor φ and Qwen Judge, 4-GPU | `Qwen/Qwen3.5-397B-A17B-FP8` | `configs/recap/vllm_serve_397b_fp8_tp4.yaml` | `VLLM_CONFIG=configs/recap/vllm_serve_397b_fp8_tp4.yaml CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/vllm/serve_397b.sh start` | 4 GPUs, tensor parallel 4 |
| Gemma Judge | `google/gemma-4-31B-it` | `configs/recap/vllm_serve_gemma4_31b_it.yaml` | `bash scripts/vllm/serve_gemma4_31b_it.sh start` | 8 GPUs, data parallel 8 |
| Gemma Judge, 2-GPU | `google/gemma-4-31B-it` | `configs/recap/vllm_serve_gemma4_31b_it_dp2_tmp.yaml` | `VLLM_CONFIG=configs/recap/vllm_serve_gemma4_31b_it_dp2_tmp.yaml CUDA_VISIBLE_DEVICES=0,1 bash scripts/vllm/serve_gemma4_31b_it.sh start` | 2 GPUs, data parallel 2 |

The judge configs pin, through the `revision` key, the Hugging Face snapshot each config served
(`Qwen/Qwen3.5-397B-A17B-FP8` at `ea5b4f81096f3901c91dea97f81324302495781d`; `google/gemma-4-31B-it`
at `439edf5652646a0d1bd8b46bfdc1d3645761a445` for the 8-GPU config and
`145dc2508c480a64b47242f160d286cff94a2343` for the 2-GPU config). The judge configs set
`allowed-local-media-path: "/"` so that requests can pass images as `file://` paths, and every
server binds `0.0.0.0:8000`; run them on a trusted host. `serve_397b.sh stop` kills every GPU
compute process on the host.

Generating the released corpus took about 12.7k H200 GPU-hours (514.6M caption rows at about 90
accepted captions per second on 8 GPUs). The sampled audits (about 100k text extraction requests and
about 100k image-question requests) and the encoder probes add on the order of 10² H200 GPU-hours.

## Data access

### Released captions

The captions generated for the paper are public, caption-only, one dataset per source family. Scale
counts unique image identities within each family; identities are not deduplicated across families,
and the three LAION subsets can share images.

| Source family | Hugging Face dataset | Scale | Paired reference surface(s) in the paper |
|---|---|---:|---|
| DataComp | [`BootsofLagrangian/datacomp-recap-qwen3p5-35b-a3b`](https://huggingface.co/datasets/BootsofLagrangian/datacomp-recap-qwen3p5-35b-a3b) | ≈325.5M | Recap-DataComp |
| CC12M | [`BootsofLagrangian/cc12m-recap-qwen3p5-35b-a3b`](https://huggingface.co/datasets/BootsofLagrangian/cc12m-recap-qwen3p5-35b-a3b) | ≈11.5M | three CC12M recaption releases (below) |
| LAION-pop | [`BootsofLagrangian/laion-pop-recap-qwen3p5-35b-a3b`](https://huggingface.co/datasets/BootsofLagrangian/laion-pop-recap-qwen3p5-35b-a3b) | ≈0.4M | LAION-pop-Llama |
| PD12M | [`BootsofLagrangian/pd12m-recap-qwen3p5-35b-a3b`](https://huggingface.co/datasets/BootsofLagrangian/pd12m-recap-qwen3p5-35b-a3b) | ≈12.4M | PD12M released |
| CommonCatalog | [`BootsofLagrangian/commoncatalog-cc-by-recap-qwen3p5-35b-a3b`](https://huggingface.co/datasets/BootsofLagrangian/commoncatalog-cc-by-recap-qwen3p5-35b-a3b) | ≈14.6M | release only |
| LAION-Aesthetics | [`BootsofLagrangian/laion-aesthetics-recap-qwen3p5-35b-a3b`](https://huggingface.co/datasets/BootsofLagrangian/laion-aesthetics-recap-qwen3p5-35b-a3b) | ≈23.7M | release only |
| LAION-HighRes-Aesthetic | [`BootsofLagrangian/laion-highres-aesthetic-recap-qwen3p5-35b-a3b`](https://huggingface.co/datasets/BootsofLagrangian/laion-highres-aesthetic-recap-qwen3p5-35b-a3b) | ≈82.7M | release only |
| Megalith-CC0 | [`BootsofLagrangian/megalith-cc0-recap-qwen3p5-35b-a3b`](https://huggingface.co/datasets/BootsofLagrangian/megalith-cc0-recap-qwen3p5-35b-a3b) | ≈8.1M | release only |
| Danbooru | [`BootsofLagrangian/danbooru-recap-qwen3p5-35b-a3b`](https://huggingface.co/datasets/BootsofLagrangian/danbooru-recap-qwen3p5-35b-a3b) | ≈11.3M | Danbooru-Florence |

```python
from datasets import load_dataset

captions = load_dataset("BootsofLagrangian/cc12m-recap-qwen3p5-35b-a3b", split="train", streaming=True)
print(next(iter(captions))["caption_text"])
```

Each dataset card documents its schema, the join keys to source images (URL, URL hash, content hash,
or shard and member), and a `generation_config.yaml` with the prompts, image preprocessing, and
decoding settings used for that family. The cards mark the captions as research data that need
safety and policy filtering before any training use. Source images are not redistributed here; obtain
them from the original releases under their own terms.

### Reference captions

The paired comparisons read the public reference releases below. `configs/caption_survey/surfaces.json`
records the dataset, split, and caption column for every surface.

| Surface | Hugging Face dataset | Caption column |
|---|---|---|
| Recap-DataComp | [`UCSC-VLAA/Recap-DataComp-1B`](https://huggingface.co/datasets/UCSC-VLAA/Recap-DataComp-1B) | `re_caption` |
| CC12M-LLaVA-NeXT | [`CaptionEmporium/conceptual-captions-cc12m-llavanext`](https://huggingface.co/datasets/CaptionEmporium/conceptual-captions-cc12m-llavanext) | `caption_llava` |
| PixelProse (CC12M split) | [`lodestones/pixelprose`](https://huggingface.co/datasets/lodestones/pixelprose), split `cc12m` (mirror of [`tomg-group-umd/pixelprose`](https://huggingface.co/datasets/tomg-group-umd/pixelprose)) | `vlm_caption` |
| CC12M-Qwen3-VL (short tag-style captions, about 12 lex, read as a short-caption special case) | [`undefined443/cc12m-wds-recaption`](https://huggingface.co/datasets/undefined443/cc12m-wds-recaption) | not set (auto-detected) |
| LAION-pop-Llama | [`CaptionEmporium/laion-pop-llama3.2-11b`](https://huggingface.co/datasets/CaptionEmporium/laion-pop-llama3.2-11b) | `caption_long_llama32` |
| PD12M released | [`Spawning/pd12m-full`](https://huggingface.co/datasets/Spawning/pd12m-full) | `caption` |
| Danbooru-Florence | [`KBlueLeaf/danbooru2023-florence2-caption`](https://huggingface.co/datasets/KBlueLeaf/danbooru2023-florence2-caption) | `parsed` |

The image-conditioned axis and the CC12M case study additionally need the source images: DataComp,
CC12M, LAION-pop, PD12M, and Danbooru2023, each obtained from its original release.

### Local layout

The configs and script defaults assume this layout under the repository root. Every default can be
overridden on the command line.

| Path | Contents | Read by |
|---|---|---|
| `outputs/recap/<dataset>/<domain>/shard-*.jsonl` | our captions, one JSON record per image with the join key named in `fair_slices.json` (`image_id` or `url`) and the text in `caption`; this is the format written by `scripts/vllm/run_recap.py` | `build_caption_fair_slices.py` |
| `data/caption-mirrors/<family>/<surface>.jsonl` | reference captions as JSONL with the join key named by `public_key_field` and the text in `caption` | `build_caption_fair_slices.py` |
| `data/manifests/<name>.manifest.parquet` | canonical image manifests with `dataset_key`, `storage_uri`, `canonical_url`, `source_url_sha1`, `stable_image_id`, `sample_id`, `width`, `height`, `bytes`, `sha256_raw` | fair slices, CC12M materialization, human study |
| `data/cc12m-wds/` | CC12M WebDataset tar shards | CC12M image materialization |
| `data/datacomp-images/` | DataComp images for the DataComp slice (`local_image_root` in `fair_slices.json`) | `build_caption_fair_slices.py` |
| `data/prompt-pools/` | raw prompt-pool downloads (`hf-raw/`, `hf-raw-by-repo/`) and prepared pools (`2026-04-24-expanded/`) | prompt-support scripts |
| `data/caption-fair-slices/` | paired slices written by `build_caption_fair_slices.py` | text diagnostics |
| `data/local-images/` | images extracted for the image-conditioned stages | request builders |
| `artifacts/` | request and response JSONL files and summaries of the VLM stages | summarizers, `scripts/paper/` |

## Reproducing the paper

The steps below list the scripts in pipeline order together with the flags recorded in the result
files. Each VLM stage writes append-only JSONL with `--resume`, so interrupted runs continue where
they stopped; the summarizers take `--latest-by-request` to keep the last response per request.

### 1. Captions (released corpus)

`scripts/vllm/run_recap.py` is the resumable caption runner. It reads images from a WebDataset tar
directory, an image directory, or Parquet files, sends one request per image to the captioner
endpoints, and appends JSONL shards with per-image checkpoints.

```bash
bash scripts/vllm/serve.sh &     # Qwen3.5-35B-A3B-FP8, data parallel 8 on port 8000
uv run python scripts/vllm/run_recap.py --dataset cc12m --domain photorealistic \
  --input-dir data/cc12m-wds/ --vllm-ports 8000 --no-endpoint-auto-restart --limit 20000
```

`--domain photorealistic` and `--domain anime_booru` select the two caption policies in
`configs/recap/domains/`; the anime policy grounds the caption on booru tags given with
`--metadata <parquet dir>`. The system and user prompts are the ones reproduced in the paper
appendix. The released captions do not need to be regenerated to run the audit.

### 2. Paired slices and text diagnostics

Phenomenon descriptors, Avg lex, B-eligibility, and the per-family descriptor table:

```bash
uv run python scripts/build_caption_fair_slices.py --all --max-pairs 1000000 --seed 0
uv run python scripts/run_caption_fair_slice_surveys.py --max-pairs 1000000 --seed 0 \
  --max-records 1000000 --token-budgets 16,32,64,128,256 --top-ks 10,100,1000 \
  --ngram-orders 1,2,3 --repeat-ngram-orders 3,4,5,6 \
  --output artifacts/caption-survey/fair_slices_1m.json
uv run python scripts/summarize_recap_fair_slice_cpu_remaining.py \
  --survey-json artifacts/caption-survey/fair_slices_1m.json
```

`caption_corpus_survey.py` is the single-corpus survey these drivers call; it also streams a Hugging
Face dataset directly (`--hf-dataset`). Encoder-token truncation rates (CLIP-77, LongCLIP-248,
SigLIP2-64) come from `caption_tokenizer_truncation_survey.py` (needs the `eval` extra).

### 3. Prompt-pool support

```bash
uv run python scripts/prepare_prompt_pools.py --output-root data/prompt-pools/2026-04-24-expanded
uv run python scripts/caption_prompt_support_bootstrap.py --max-pairs 1000000 --seed 0 \
  --budget 64 --ngram 2 --hash-buckets 262144 --block-size 5000 \
  --max-caption-records 250000 --max-prompt-records 1000000 --bootstrap-reps 2000 \
  --output artifacts/caption-survey/prompt_support_bootstrap_b64_n2_250k.json
```

`prepare_prompt_pools.py` reads the raw pool downloads from the paths listed in its `DEFAULT_SOURCES`
table (under `data/prompt-pools/hf-raw/` and `data/prompt-pools/hf-raw-by-repo/`) and writes one
filtered JSONL per pool. The paper's seven pools are `civitai_flux_prompts_aconexx`,
`flux_improved_k_mktr`, `flux_prompts_chrisgoringe`, `flux_prompts_regpeter`,
`sd_prompts_2m_andyyang`, `sdxl_refiner_prompts_falah`, and `pickapic_rankings`; the eighth pool,
`sd_prompts_dedup_xzuyn`, is a DiffusionDB-derived deduplication kept for sensitivity analysis.
`caption_prompt_ngram_support.py`, `caption_prompt_support_pools.py`, and
`extract_diffusiondb_prompts.py` produce the sensitivity variants.

### 4. Claim extraction (claimed CBU)

Start the extractor (`serve_397b.sh`), then build one request per caption window and run it:

```bash
uv run python scripts/build_caption_cbu_requests.py --input <surface>.jsonl \
  --output artifacts/cbu/<name>.requests.jsonl --surface <surface> --token-budget 64 [--sample-records 5000]
uv run python scripts/run_text_json_requests.py --input artifacts/cbu/<name>.requests.jsonl \
  --output artifacts/cbu/<name>.responses.jsonl --urls http://localhost:8000 \
  --model Qwen/Qwen3.5-397B-A17B-FP8 --concurrency 1024 --max-tokens 4096 --timeout-sec 1800 \
  --structured-json --resume --resume-ok-only
uv run python scripts/summarize_cbu_responses.py --latest-by-request --mode claimed \
  --input artifacts/cbu/<name>.responses.jsonl --output artifacts/cbu/<name>.responses.summary.json
uv run python scripts/export_cbu_metric_tables.py --claimed <label>=artifacts/cbu/<name>.responses.jsonl \
  --output-dir artifacts/cbu/tables --bootstrap-reps 2000 --seed 0
```

The request builder truncates each caption to its first B whitespace-delimited words and embeds the
extraction prompt and JSON schema reproduced in the paper appendix. The summarizer reports
`claimed_dedup_per_caption` (CBU/cap) and `claimed_dedup_per_100_tokens` (CBU/100 lex).

### 5. Image-conditioned verification (supported CBU and risk)

Each claimed CBU is bound to the paired image and turned into a yes/no question:

```bash
uv run python scripts/build_grounded_cbu_verify_requests.py --claimed-responses artifacts/cbu/<name>.responses.jsonl \
  --source-jsonl <surface>.jsonl --output artifacts/grounded-cbu/<name>.requests.jsonl --require-local-image
uv run python scripts/build_cbu_vqa_requests.py --input artifacts/grounded-cbu/<name>.requests.jsonl \
  --output artifacts/vqa-cbu/<name>.requests.jsonl
uv run python scripts/run_cbu_vqa_requests.py --input artifacts/vqa-cbu/<name>.requests.jsonl \
  --output artifacts/vqa-cbu/<name>.responses.<judge>.jsonl --urls http://localhost:8000 \
  --model Qwen/Qwen3.5-397B-A17B-FP8 --concurrency 384 --max-tokens 2048 --timeout-sec 2400 \
  --image-mode file --structured-json --no-evidence --resume --resume-ok-only
uv run python scripts/summarize_cbu_vqa_responses.py --latest-by-request \
  --input artifacts/vqa-cbu/<name>.responses.<judge>.jsonl --output artifacts/vqa-cbu/<name>.summary.json
uv run python scripts/export_cbu_vqa_tables.py --summary artifacts/vqa-cbu/<name>.summary.json \
  --output-md artifacts/vqa-cbu/<name>.md --output-tex artifacts/vqa-cbu/<name>.tex
```

For the Gemma Judge, serve `google/gemma-4-31B-it` and repeat `run_cbu_vqa_requests.py` on the same
request file with `--model google/gemma-4-31B-it`; both judges then answer the identical question
set. `--no-evidence` selects the compact answer-only schema used in the paper. The grounded stage
(`run_grounded_cbu_verify_requests.py`, `summarize_grounded_cbu_verify.py`) is an earlier exact-unit
verification that the reported tables do not use; `build_grounded_cbu_verify_requests.py` is still
the step that attaches image paths to the claims.

`scripts/run_qwen397_pair5k_cbu_pipeline.py` runs stages 4 and 5 for all cross-corpus surfaces at
once. It reads a manifest (`--manifest`) that lists, per comparison, the `summary.json` written by
`build_caption_fair_slices.py`, expects the claimed-CBU requests under
`artifacts/cbu/pair5k/claimed_cbu_v2_<comparison>__<surface>_<budget>_5k.requests.jsonl`, and calls
the stage scripts with the interpreter in `$AUDIT_PYTHON` (default: the current one).

Every VQA cell of the paper, the per-type breakdowns, the count/relation-excluded headline, the
captioning-policy control, and the CC12M judge agreement come from the response files through the
paper-side scripts:

```bash
uv run python scripts/paper/summarize_cbu_vqa_by_category.py --root . \
  --verification-root <verification dir> --output results/cbu_vqa_by_category_b64.json
uv run python scripts/paper/gen_cbu_category_tables.py      # writes results/tables/*.tex
```

`summarize_cbu_vqa_by_category.py` summarizes both judges on the requests that both of them answered
(a parsed answer record came back; when response files are merged, an answered row is kept over an
unanswered one), so each (slice, surface) cell compares the two judges on one question set, and it
attaches the standard deviation of supported CBU/cap and risk over 2,000 caption-level bootstrap
resamples (seed 0). It reads two input directories: `--root`, the working tree with `artifacts/vqa-cbu/`
and `artifacts/cbu/`, and `--verification-root`, a directory whose `responses/` folder holds the
verification runs (both judges on the DataComp 5k sample, on the four naive surfaces, and on the
DataComp `Ours` rows of the control). Its docstring lists the directory layout and its `SOURCES`
constant every file name. `gen_cbu_category_tables.py` writes the CC12M denominators, the
count/relation-excluded headline, the per-type table, the policy-control table, the cross-corpus
question denominators (`vqa_questions.tex`), and `vqa_mean_std.tex` with every VQA cell as mean ± std;
run on `results/cbu_vqa_by_category_b64.json`, it reproduces the six files in `results/tables/` byte
for byte.

### 6. CC12M four-surface slice and the budget sweep

The CC12M case study compares ours with the three CC12M reference surfaces on the 4,494 images that
all four surfaces share.

```bash
uv run python scripts/build_cc12m_four_caption_llava_url_bridge_slice.py \
  --ours-jsonl <ours>.jsonl --qwen-jsonl <short-caption reference>.jsonl --llavanext-jsonl <llavanext>.jsonl \
  --pixelprose-jsonl <pixelprose>.jsonl --manifest data/manifests/cc12m-202603.manifest.parquet \
  --output-dir data/local-images/cc12m-four-caption-url-key-5k --max-rows 5000 --seed 0
uv run python scripts/materialize_cc12m_four_caption_from_manifest.py \
  --input-dir data/local-images/cc12m-four-caption-url-key-5k \
  --output-dir data/local-images/cc12m-four-caption-url-key-5k-local
```

The bridge joins ours and the `--qwen-jsonl` and `--llavanext-jsonl` references on the original
numeric key, joins PixelProse on the normalized LLaVA-NeXT URL, and resolves each image through the canonical manifest;
the materializer extracts the images from the CC12M tar shards and rewrites all four surfaces with the
same local image path. Stages 4 and 5 then run on the four materialized surfaces at B = 64 under both
judges.

The budget sweep takes B = 64 from that extraction and B ∈ {16, 32, 48} from two runs with the same
checkpoint and prompts, one over the first 1,000 images and one over the remaining 3,494:

```bash
uv run python scripts/paper/prepare_cc12m_bgrid_full.py --repo . --work artifacts/cbu/cc12m-bgrid-full
# run run_text_json_requests.py + summarize_cbu_responses.py on each rows1000-4493 request file
uv run python scripts/paper/compare_cbu_rerun.py --original <pilot responses> --rerun <rerun responses> --output <report>.json
uv run python scripts/paper/build_cc12m_budget_frontier_csv.py \
  --summary 16=<b16 summary> --summary 32=<b32 summary> --summary 48=<b48 summary> --summary 64=<b64 summary> \
  --output results/cc12m_budget_frontier_plot.csv
uv run python scripts/paper/gen_cc12m_frontiers.py
```

`prepare_cc12m_bgrid_full.py` reads the B = 64 request file and the 1,000-image pilot requests at the
paths in its `FULL_B64` and `PILOT` constants, rebuilds the requests for all 4,494 images with the
unchanged request builder, checks that the first 1,000 rows equal the pilot requests record for
record, and writes the remaining rows as one run file per budget. `gen_cc12m_frontiers.py` redraws both panels of Figure 2
from `results/` (set `EVAL_DIR` to read and write elsewhere).

The lexical-window sensitivity check cuts the extractor window at B lexical units instead of B
whitespace words, with the builder, prompts, and schema unchanged:

```bash
uv run python scripts/paper/build_lexical_window_cbu_requests.py --work artifacts/cbu/cc12m-lex64 --budget 64
# run run_text_json_requests.py and summarize_cbu_responses.py --mode claimed on the merged request file
```

It reads the B = 64 CC12M request file of this step (override with `--requests`) and writes one
request file per surface plus a merged file; the summary is
`results/sensitivity/cc12m_lexical_window_claimed_cbu_summary.json`.

### 7. Captioning-policy control (naive policy)

The control keeps the captioner and the images fixed and replaces the policy with the Recap-DataComp
instruction as a single user message with no system prompt:

```
Please generate a detailed caption of this image. Please be as descriptive as possible.
```

Each family has two naive surfaces. `Naive` uses matched decoding, the captioner's release sampling
defaults (temperature 1.0, top_k 20, top_p 0.95) that also produced the released captions;
`Naive (greedy)` decodes at temperature 0. Claims are extracted once, by Qwen3.5-397B-A17B-FP8, and both
judges answer the VQA requests built from those claims.

| Surface | Family | Decoding | Results |
|---|---|---|---|
| `naive_qwen35_sampled_cc12m` | CC12M | matched (temperature 1.0, top_k 20, top_p 0.95) | `results/naive_qwen35_sampled_cc12m/` |
| `naive_qwen35_cc12m` | CC12M | greedy (temperature 0) | `results/naive_qwen35_cc12m/` |
| `naive_qwen35_sampled_datacomp` | DataComp | matched (temperature 1.0, top_k 20, top_p 0.95) | `results/naive_qwen35_sampled_datacomp/` |
| `naive_qwen35_datacomp` | DataComp | greedy (temperature 0) | `results/naive_qwen35_datacomp/` |

Each results directory holds the surface's captions (`captions.jsonl.gz`, gzip-compressed JSONL; read
it with `gzip.open(path, "rt")` or `zcat`), its claimed-CBU summary, and one VQA summary per judge; its
README names the producing scripts. The captions carry public image keys only (`image_url`,
`public_lookup_key`, `pair_key`), and personal data transcribed from image text is masked as
`[email]`, `[phone]`, `[name]`, `[id]`, and `[address]`. `results/policy_control_ours_datacomp/`
holds the matching summaries of the released captions on the same DataComp images.

```bash
# captions and claim-extraction requests (greedy by default)
bash scripts/run_cc12m_naive_qwen35_baseline.sh
SURFACE=naive_qwen35_sampled_cc12m CAPTION_TEMPERATURE=1.0 CAPTION_TOP_K=20 CAPTION_TOP_P=0.95 \
  RUN_ROOT=artifacts/recap-ed/cc12m-naive-qwen35-sampled bash scripts/run_cc12m_naive_qwen35_baseline.sh
# claim extraction and the Qwen Judge with the 397B server: steps 4 and 5
# Gemma Judge on the VQA requests built from the Qwen397 claims
bash scripts/run_cc12m_naive_qwen35_gemma_metrics.sh
bash scripts/run_datacomp_naive_qwen35_qwen397_metrics.sh
```

Each driver documents the server it expects and takes its paths, surface name, decoding, and
concurrency from environment variables (`RUN_ROOT`, `SURFACE`, `CAPTION_TEMPERATURE`, `IMAGE_DIR`,
`MODEL`, `URLS`, ...). The CC12M baseline driver starts from the CC12M VQA request file of step 6 and
extracts the images it references from `data/cc12m-wds` (`materialize_cc12m_images_from_requests.py`).
The Gemma driver requires the Qwen397 claimed-CBU responses, builds the VQA requests from them if they
do not exist yet, and never extracts claims with Gemma. The DataComp driver starts from naive DataComp
captions and claimed-CBU requests already placed under its `RUN_ROOT` and `CBU_ROOT`; produce them with
`build_naive_vlm_caption_requests.py`, `run_naive_vlm_caption_requests.py` (`--temperature`, `--top-k`,
`--top-p`), `summarize_naive_vlm_captions.py`, and `build_caption_cbu_requests.py` as in the CC12M
driver. `export_cc12m_naive_qwen35_comparison_tables.py` writes the surface-concentration comparison
for the greedy CC12M surface.

### 8. Encoder-side probes (appendix)

These need `uv sync --extra eval` and a GPU.

```bash
uv run python scripts/caption_embedding_vendi.py --help          # encode / vendi / geometry / knn / support subcommands
uv run python scripts/compute_longclip_retrieval_margin.py --surface ours=<ours>.jsonl ... \
  --output-dir artifacts/longclip/<run> --model zer0int/LongCLIP-GmP-ViT-L-14 --max-length 248 \
  --retrieval-block-size 512 --bootstrap-reps 1000 --trust-remote-code
uv run python scripts/caption_tokenizer_truncation_survey.py --help
```

The input64 LongCLIP mode feeds captions already truncated to 64 lexical units.
`results/sensitivity/` holds the encoder-token truncation of the naive captions, written by
`scripts/paper/encoder_truncation_rates.py` (untruncated tokenization with special tokens), and the
LongCLIP retrieval of the matched-decoding CC12M naive surface in both modes:

```bash
uv run python scripts/paper/encoder_truncation_rates.py \
  --surface naive_qwen35_sampled_cc12m=results/naive_qwen35_sampled_cc12m/captions.jsonl.gz \
  --output artifacts/encoder-truncation/naive.json
```

### 9. Figures

```bash
uv run python scripts/paper/gen_teaser_refined.py      # Figure 1 (right), means over the seven pools
uv run python scripts/paper/gen_cc12m_frontiers.py     # Figure 2
uv run python scripts/paper/gen_per_pool_heatmap.py    # appendix per-pool heatmap
uv run python scripts/plot_caption_survey_curves.py --help
```

The generators overwrite their outputs in `results/`.

### 10. Human calibration study

`scripts/human_cbu_eval.py` runs the blinded study reported in the paper: annotators judge a sampled
claim first against the caption window with the image hidden, then against the image with the
caption hidden, and finally rate the whole pair. Surface identity and judge outputs stay hidden. The
study configuration is `configs/eval/human_cbu_cc12m.yaml` (seed 1477, B = 64, two surface groups,
eight claim types); its inputs are the CC12M claimed-CBU and judge response files of step 6.

```bash
uv run python scripts/human_cbu_eval.py sample --config configs/eval/human_cbu_cc12m.yaml --output-dir artifacts/human-cbu/sample
uv run python scripts/human_cbu_eval.py materialize --config configs/eval/human_cbu_cc12m.yaml --sample-dir artifacts/human-cbu/sample
uv run python scripts/human_cbu_eval.py init-db --config configs/eval/human_cbu_cc12m.yaml --sample-dir artifacts/human-cbu/sample --db artifacts/human-cbu/study.sqlite
uv run python scripts/human_cbu_eval.py invites --db artifacts/human-cbu/study.sqlite --study-id <study id> --count 20 --output artifacts/human-cbu/invites.json
uv run python scripts/human_cbu_eval.py assign --db artifacts/human-cbu/study.sqlite --study-id <study id> --config configs/eval/human_cbu_cc12m.yaml
uv run python scripts/human_cbu_eval.py seal --db ... --study-id ... --ethics-determination-id ... --participant-information-json ...
uv run python scripts/human_cbu_eval.py open --db ... --study-id ... --confirm-open <study id>
uv run python scripts/human_cbu_eval.py serve --db ... --study-id ... --host 127.0.0.1 --port 8765
uv run python scripts/human_cbu_eval.py close --db ... --study-id ...
uv run python scripts/human_cbu_eval.py export --db ... --study-id ... --output-dir artifacts/human-cbu/export
```

Judge–human agreement, with its standard deviation over 10,000 image-cluster bootstrap resamples
(seed 1477), comes from the export directory; the output holds aggregate statistics only:

```bash
uv run python scripts/paper/human_judge_agreement_bootstrap.py artifacts/human-cbu/export \
  > results/human_cbu/judge_human_agreement_bootstrap.json
```

`participant-test` serves the full participant flow in response-discarding memory for rehearsal;
`validate`, `status`, `backup`, and the `adjudication-*` commands cover pre-launch checks, operations,
and disagreement review. The study configuration disables storage of names, email addresses, IP
addresses, and user agents, and the export is aggregate-only unless `--include-public-rows` is given.
Run the tests with:

```bash
uv sync --extra dev
uv run pytest
```

## Results directory

`results/` holds the summary files behind the paper's tables and figures, copied byte for byte from
the paper's result bundles. Two kinds of file are exceptions: the naive-control captions
(compressed, with personal data masked) and the two LongCLIP TSVs in `results/sensitivity/` (interval
columns omitted). [`results/README.md`](results/README.md) maps each file to the table or figure it
feeds and to the script that produces it, marks the superseded files, and lists what the release
does not include.

| File | Paper element |
|---|---|
| `all_cbu_b64_summary.csv` | cross-corpus CBU/cap and CBU/100 lex |
| `cbu_vqa_by_category_b64.json`, `tables/*.tex` | every VQA cell under both judges as mean ± std; per-type tables; count/relation-excluded headline; policy control; question denominators; CC12M denominators |
| `all_vqa_b64_summary.csv` | per-judge rollup of the Qwen Judge and CC12M summaries (see `results/README.md` for its DataComp rows) |
| `cc12m_budget_frontier_plot.csv`, `cc12m_vqa_supported_risk_pareto.csv` | CC12M frontier table and Figure 2 |
| `prompt_support_bootstrap_b64_n2_250k_2026-04-24.tsv`, `teaser_right_v_twinx_v2.{pdf,png}` | Pool-wins, Figure 1 (right, seven-pool means), per-pool heatmap |
| `human_cbu/judge_human_agreement_bootstrap.json` | judge–human agreement, mean ± std |
| `naive_qwen35_*/`, `policy_control_ours_datacomp/` | captioning-policy control: naive captions and summaries, and the released captions on the same DataComp images |
| `datacomp_pair/` | per-judge summaries of the DataComp verification run |
| `sensitivity/` | lexical-window claimed CBU, encoder truncation and LongCLIP retrieval of the naive captions |
| `raw_summaries/` | per-stage summaries (text diagnostics, prompt support, CBU, VQA, embeddings, LongCLIP) |
| `cc12m_cbu_vqa_bootstrap_ci.tsv`, `cc12m_gemma4_vqa_bootstrap_ci.tsv` | superseded CC12M bootstrap interval exports, kept for traceability |

## Citation

```bibtex
@inproceedings{oh2026matchedbudget,
  title     = {A Matched-Budget Audit Framework for Recaptioned Image-Text Supervision Distributions},
  author    = {Oh, Giyeong and Park, Junghun and Bae, Yuhan and Yu, Youngjae},
  booktitle = {Advances in Neural Information Processing Systems, Evaluations and Datasets Track},
  year      = {2026},
  url       = {https://openreview.net/forum?id=JobgYHJvPo}
}
```

## License

The code in this repository is released under the [Apache License 2.0](LICENSE). The released
captions are licensed CC-BY-4.0, as stated on each dataset card. The naive-control caption text in
`results/naive_qwen35_*/captions.jsonl.gz` is also CC-BY-4.0, the same license as the released
captions; these files carry public image keys only, and personal data transcribed from image text
(email addresses, phone numbers, names of private persons on ID cards, badges and certificates,
identity and card numbers, and private street addresses) is masked as `[email]`, `[phone]`, `[name]`,
`[id]`, and `[address]`. The CC-BY-4.0 license covers the generated caption text only. Source images, reference captions, and third-party metadata keep their original
licenses and terms.

## Contact

Please open an issue on this repository for questions about the code or the audit. Author email
addresses are listed in the paper.
