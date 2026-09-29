# Results

This directory holds the summary artifacts behind the tables and figures of the paper. Every
result file is a byte-for-byte copy of the file the producing run wrote, taken from the paper's
result bundle (`artifacts/eval_results/` for the audit results, `artifacts/human_cbu/` for the
annotation-interface screenshots, `artifacts/figures/` for the composed Figure 1 (left) panel, and
the final verification bundle for the policy control, the DataComp
verification run, and the sensitivity checks). No number in this directory was edited by hand. To
refresh a result, copy the updated file from the bundle to the same relative path here.

Two kinds of file are not byte-for-byte copies:

- The naive-control captions (`naive_qwen35_*/captions.jsonl.gz`) are stored gzip-compressed, carry
  public image keys only, and have personal data transcribed from image text masked as `[email]`,
  `[phone]`, `[name]`, `[id]`, and `[address]`; each folder's README gives the counts and shows how to
  read the file. Their text is CC-BY-4.0, like the released captions.
- The two LongCLIP TSVs in `sensitivity/` omit the `pos_ci95`, `i2t_margin_ci95`, and
  `t2i_margin_ci95` interval columns that `scripts/compute_longclip_retrieval_margin.py` also writes,
  so that they carry means and rates only; every other column is unchanged.

Most files are written by a script in `scripts/`, as listed below. The top-level rollup tables
(`all_cbu_b64_summary.csv`, `all_vqa_b64_summary.csv`, `cc12m_vqa_supported_risk_pareto.csv`,
`cc12m_longclip_plot.csv`, `prompt_support_direction_summary.csv`) collect fields from the summary
files under `raw_summaries/` into one table each; the collection step has no separate script.

B = 64 is the text budget: text statistics count lexical units and claim extraction reads the first
B whitespace-delimited words. Several columns keep the names they had before the paper switched its
wording from "token" to "lexical unit": `avg_tokens`, `avg_lexical_tokens`, `cov64`, `elig64`, and
`cbu_100tok` / `cbu_per_100tok` all count regex lexical units (`[^\W_]+(?:'[^\W_]+)*`), not tokenizer
tokens.

Surface names follow `configs/caption_survey/surfaces.json`: `ours_*` is the released recap surface,
`ref_*` a public reference surface. Cross-corpus rows are prefixed by their paired comparison, for
example `pd12m_full_paired__ours_pd12m_img2dataset`.

## Main text

| Paper element | File(s) | Produced by |
|---|---|---|
| Figure 1 (left), caption register (describe-style frame prefix concentration) | `teaser_left_v4.pdf` | composed figure, copied from the manuscript's `artifacts/figures/`; `teaser_left_v2.pdf` and `teaser_left_v3.pdf` are the superseded earlier versions |
| Figure 1 (right), prompt-pool diagnostics (prompt-mass support and n-gram JSD) | `teaser_right_v_twinx_v3.{pdf,png}` | `scripts/paper/gen_teaser_refined.py`, which averages the per-pool block means of `prompt_support_bootstrap_b64_n2_250k_2026-04-24.tsv` over the seven pools; `teaser_right_v_twinx_v2.{pdf,png}` is the superseded earlier rendering |
| Phenomenon descriptors (opener rate, top-100 raw / content prefix mass, Distinct-3, average length) and the per-family appendix companion | `raw_summaries/cpu_text_metrics/fair_slices_1m_normalized_2026-04-24.{json,tsv}`; LAION-pop re-run with URL-fixed pairing in `laion_pop_url_fixed_normalized_2026-04-24.{json,tsv}` | `scripts/run_caption_fair_slice_surveys.py` over the paired slice of each surface, capped at 1M captions (42,231 on LAION-pop, 114,621 on CC12M–Qwen3-VL, 729,237 on PixelProse, 960,394 on CC12M-LLaVA-NeXT, 999,993 on Danbooru, and 1M on DataComp and PD12M; `records` column) |
| Cross-corpus headline at B = 64: Avg lex | `raw_summaries/cpu_text_metrics/fair_slices_1m_normalized_2026-04-24.tsv` (`avg_tokens`) | as above |
| Cross-corpus headline: CBU/cap | `all_cbu_b64_summary.csv` (`cbu_cap`, `cbu_100tok`) | rollup of `raw_summaries/cbu_claimed/claimed_cbu_v2_all7_b64_5k.*.summary.json` (`scripts/summarize_cbu_responses.py --mode claimed`) |
| Cross-corpus headline: Pool-wins | `prompt_support_bootstrap_b64_n2_250k_2026-04-24.tsv` | `scripts/caption_prompt_support_bootstrap.py`; Pool-wins counts the seven pools other than `sd_prompts_dedup_xzuyn` with `delta_mean_local_minus_reference > 0` on `prompt_mass_on_caption_support` |
| Cross-corpus headline: Sup. CBU/cap and Risk under the Qwen Judge and the Gemma Judge | `cbu_vqa_by_category_b64.json` (`all_types` of the `DataComp`, `LAION-pop`, `PD12M`, and `Danbooru` cells; both judges on the requests both answered); the same cells as mean ± std in `tables/vqa_mean_std.tex`; the per-judge summaries of the DataComp verification run in `datacomp_pair/` | `scripts/paper/summarize_cbu_vqa_by_category.py`, `scripts/paper/gen_cbu_category_tables.py`; per-judge summaries by `scripts/summarize_cbu_vqa_responses.py` |
| Captioning-policy control (naive policy, CC12M and DataComp; `Naive` = matched decoding, `Naive (greedy)` = temperature 0) | `tables/policy_control.tex`; `CC12M-control` and `DataComp-control` cells of `cbu_vqa_by_category_b64.json`; `naive_qwen35_sampled_cc12m/`, `naive_qwen35_cc12m/`, `naive_qwen35_sampled_datacomp/`, `naive_qwen35_datacomp/` (captions, claimed-CBU summary, one VQA summary per judge); the DataComp `Ours` rows in `policy_control_ours_datacomp/`; the CC12M `Ours` row is the CC12M case study (`budget = 64` row of `cc12m_budget_frontier_plot.csv` and the `CC12M` cells) | `scripts/run_cc12m_naive_qwen35_baseline.sh`, `scripts/run_cc12m_naive_qwen35_gemma_metrics.sh`, `scripts/run_datacomp_naive_qwen35_qwen397_metrics.sh`, `scripts/summarize_cbu_responses.py --mode claimed`, `scripts/summarize_cbu_vqa_responses.py`, `scripts/paper/summarize_cbu_vqa_by_category.py`, `scripts/paper/gen_cbu_category_tables.py` |
| CC12M frontier at B = 64: CBU/cap and CBU/100lex | `cc12m_budget_frontier_plot.csv` (rows with `budget = 64`) | `scripts/paper/build_cc12m_budget_frontier_csv.py` |
| CC12M frontier at B = 64: Sup. CBU/cap and Risk under both judges | the `CC12M` cells of `cbu_vqa_by_category_b64.json` (requests answered by both judges, with bootstrap std); `all_vqa_b64_summary.csv` (`source = cc12m_qwen`, `cc12m_gemma`) carries the same values | `scripts/paper/summarize_cbu_vqa_by_category.py`; rollups of the two CC12M summaries in `raw_summaries/vqa_image_conditioned/` |
| Figure 2 (left), CC12M supported yield vs. risk | `cc12m_vqa_supported_risk_pareto_v3.{pdf,png}` | `PANELS=left scripts/paper/gen_cc12m_frontiers.py` from the `CC12M` cells of `cbu_vqa_by_category_b64.json` |
| Figure 2 (right), CC12M budget sweep B in {16, 32, 48, 64} | `cc12m_cbu_efficiency_yield_frontier_revised.{pdf,png}` | `scripts/paper/gen_cc12m_frontiers.py` from `cc12m_budget_frontier_plot.csv` |
| Human verification of image support (seven volunteer annotators, 217 primary judgments on 137 claims), judge–human agreement as mean ± std over 10,000 image-cluster bootstrap resamples | `human_cbu/judge_human_agreement_bootstrap.json` (aggregate only) | `scripts/paper/human_judge_agreement_bootstrap.py` on the output of `scripts/human_cbu_eval.py export` |

## Appendix

| Paper element | File(s) | Produced by |
|---|---|---|
| Per-pool prompt-support heatmap | `per_pool_prompt_support_heatmap.{pdf,png}`; per-comparison direction counts over all eight pools in `prompt_support_direction_summary.csv` | `scripts/paper/gen_per_pool_heatmap.py` from `prompt_support_bootstrap_b64_n2_250k_2026-04-24.tsv` (seven pools) |
| VQA question denominators and B-eligibility of the cross-corpus pairs | `tables/vqa_questions.tex` (responses, questions, questions per response, and risk from the Qwen Judge cells of `cbu_vqa_by_category_b64.json`); B-eligibility from `raw_summaries/cpu_text_metrics/fair_slices_1m_normalized_2026-04-24.tsv` (`cov64`); paired deltas in `raw_summaries/cpu_text_metrics/paired_delta_ci.tsv` (`elig64`) | `scripts/paper/gen_cbu_category_tables.py`; see above for the survey; `scripts/summarize_recap_fair_slice_cpu_remaining.py` for the paired deltas |
| CC12M denominators | `tables/cc12m_denominators.tex` | `scripts/paper/gen_cbu_category_tables.py` from `cbu_vqa_by_category_b64.json` |
| Cross-corpus headline without count and relation claims | `tables/excl_count_relation.tex` | `scripts/paper/gen_cbu_category_tables.py` |
| Image support and risk by claim type | `tables/vqa_by_type.tex` | `scripts/paper/gen_cbu_category_tables.py` |
| VQA cells as mean ± bootstrap std (2,000 caption-level resamples, seed 0) | `tables/vqa_mean_std.tex` from `cbu_vqa_by_category_b64.json` (`supported_cap_std`, `risk_std`) | `scripts/paper/summarize_cbu_vqa_by_category.py`, `scripts/paper/gen_cbu_category_tables.py` |
| Lexical-window sensitivity of claimed CBU (CC12M) | `sensitivity/cc12m_lexical_window_claimed_cbu_summary.json`: the four CC12M surfaces with the extractor window cut at the first 64 lexical units instead of the first 64 whitespace words | `scripts/paper/build_lexical_window_cbu_requests.py` builds the requests from the B = 64 CC12M request file; then `scripts/run_text_json_requests.py` and `scripts/summarize_cbu_responses.py --mode claimed` |
| Encoder-token truncation of the naive captions | `sensitivity/naive_encoder_truncation.json` (CLIP-77, LongCLIP-248, SigLIP2-64 per naive surface) | `scripts/paper/encoder_truncation_rates.py` (untruncated tokenization with special tokens, as in `scripts/caption_tokenizer_truncation_survey.py`) |
| LongCLIP retrieval of the matched-decoding CC12M naive surface | `sensitivity/naive_sampled_cc12m_longclip_full.tsv`, `sensitivity/naive_sampled_cc12m_longclip_input64.tsv` (means and rates) | `scripts/compute_longclip_retrieval_margin.py`, which also writes `*_ci95` interval columns; those columns are omitted here |
| Annotation-interface figures of the human verification | `human_cbu/ui_appendix/*.png` | screenshots of the response-discarding `participant-test` mode of `scripts/human_cbu_eval.py` with an invented caption and a synthetic scene |
| DataComp text-space probes by encoder on 50k paired rows: Vendi (the exponential of the eigenvalue entropy of the caption-set kernel), eRank, prompt-to-caption Coverage@10 and Density@10 | `raw_summaries/embedding_vendi_support/caption_embedding_profile.tsv` (Vendi, eRank); `prompt_caption_support.tsv` (Coverage and Density, raw-text protocol rows `raw/raw` and, for BGE-M3, `raw/corpus`) | `scripts/caption_embedding_vendi.py` |
| EmbeddingGemma-300M multi-slice grid | `embeddinggemma_pair_summary.tsv`; `raw_summaries/embedding_vendi_support/embeddinggemma_all_pairs.tsv`, `embeddinggemma_dtype_sanity.json` | `scripts/caption_embedding_vendi.py` |
| LongCLIP retrieval on CC12M, full-caption and input64 modes | `cc12m_longclip_plot.csv` (both modes); `longclip_retrieval_summary.tsv` and `raw_summaries/longclip_retrieval/` (input64 mode) | `scripts/compute_longclip_retrieval_margin.py` |
| Encoder-token truncation rates | `raw_summaries/cpu_text_metrics/tokenizer_truncation_core_both_100k_2026-04-24.{json,tsv}` | `scripts/caption_tokenizer_truncation_survey.py` |
| Prompt-pool sensitivity (three-pool, eight-pool, DiffusionDB, disjoint caption pools) | `raw_summaries/prompt_support/*` | `scripts/caption_prompt_ngram_support.py`, `scripts/caption_prompt_support_pools.py`, `scripts/caption_prompt_support_bootstrap.py` |

## `raw_summaries/`

| Directory | Contents |
|---|---|
| `cpu_text_metrics/` | surveys of every paired slice (capped at 1M captions per surface), paired differences (`paired_delta_ci.tsv`), per-code violation rates (`violation_code_breakdown.tsv`, where `J_meta_statement` is the opener rate), a Re-LAION-Caption19M reference survey, tokenizer truncation, and a CPU sanity manifest for the GPU metrics |
| `prompt_support/` | hashed n-gram prompt-support and JSD runs at 250,000 caption pairs per comparison (114,621 on CC12M–Qwen3-VL) against pools of up to 1M prompts |
| `cbu_claimed/` | claimed-CBU summaries at B = 64 for the cross-corpus 5k samples (`all7`, `completed4`, `cc12m3`), caption-level bootstrap summaries, and the CC12M budget sweep at B in {16, 32, 48} on all 4,494 aligned images (`*_4494.merged.summary.json`, `cc12m_budget_frontier_plot_4494.csv`), with the request rebuild check (`prepare_report.json`) and the 800-request rerun agreement check (`overlap_b32_rows0-199.comparison.json`) |
| `cbu_grounded_legacy/` | summaries of the earlier exact-unit grounded verification stage; kept for traceability and not used in the reported tables |
| `vqa_image_conditioned/` | Qwen Judge summaries for the cross-corpus pairs and CC12M, the Gemma Judge summary for CC12M, and the earlier CC12M bootstrap interval exports (superseded, see below) |
| `embedding_vendi_support/` | encoder-side diversity and prompt-to-caption support profiles |
| `longclip_retrieval/` | LongCLIP retrieval summary (input64 mode) |

`<LOCAL_CACHE>` and `<PROJECT_ROOT>` inside JSON files are placeholders for the machine-specific
directories the runs read from.

These readings need care:

- The CC12M rows of `all_cbu_b64_summary.csv` come from the per-pair 5k CC12M samples. The CC12M
  case study in the paper uses the 4,494 images shared by all four CC12M surfaces; its claimed-CBU
  numbers are the `budget = 64` rows of `cc12m_budget_frontier_plot.csv`.
- Figure 2 is `cc12m_vqa_supported_risk_pareto_v3.{pdf,png}` (left) and
  `cc12m_cbu_efficiency_yield_frontier_revised.{pdf,png}` (right).
- The `datacomp_qwen` rows of `all_vqa_b64_summary.csv` and the DataComp Qwen Judge summary in
  `raw_summaries/vqa_image_conditioned/` come from an earlier Qwen Judge run on the DataComp sample
  (Ours: 4,648 responses, 13.84 supported CBU/cap). The paper's DataComp cells use the verification
  run in `datacomp_pair/` (both judges), restricted to the 4,771 Ours and 4,914 reference requests
  that both judges answered: 13.73 and 8.49 under the Qwen Judge.
- The per-judge summaries in `datacomp_pair/` count every request that judge answered (Ours: 4,848
  under the Qwen Judge, 4,773 under the Gemma Judge); the cells of `cbu_vqa_by_category_b64.json`
  use the requests both judges answered, so the two differ in the second decimal (Qwen Judge Ours
  13.71 in the summary, 13.73 in the cell). In the policy control both judges answered every
  request, so the summaries in `policy_control_ours_datacomp/` and `naive_qwen35_*/` match their
  cells.

## Superseded files

`cc12m_cbu_vqa_bootstrap_ci.tsv` and `cc12m_gemma4_vqa_bootstrap_ci.tsv` (at the top level and, as
identical copies, in `raw_summaries/vqa_image_conditioned/`) are the earlier CC12M bootstrap exports,
which report 95% intervals. They are **superseded** by the mean ± std convention of
`tables/vqa_mean_std.tex` and are kept unchanged for traceability only; the paper does not report
them.

`teaser_left_v2.pdf` and `teaser_left_v3.pdf` are the earlier versions of Figure 1 (left). They are
**superseded** by `teaser_left_v4.pdf` and kept for traceability only.

`cc12m_vqa_supported_risk_pareto.csv` and `cc12m_vqa_supported_risk_pareto_revised.{pdf,png}` are the
earlier export of the CC12M support/risk cells and the Figure 2 (left) panel drawn from it. The export
predates the common set of requests answered by both judges, so its risk values differ from the
final `CC12M` cells of `cbu_vqa_by_category_b64.json` in the third decimal (by at most 0.001). They are
**superseded** by those cells and by `cc12m_vqa_supported_risk_pareto_v3.{pdf,png}` and kept for
traceability only.

`teaser_right_v_twinx_v2.{pdf,png}` is the earlier rendering of Figure 1 (right), with the axis
labelled "prompt coverage". It is **superseded** by `teaser_right_v_twinx_v3.{pdf,png}` (labelled
"prompt-mass support", same values) and kept for traceability only.

## Manuscript result-provenance paths

The result-provenance table of the paper appendix names the artifacts below. Paths are relative to
`results/`.

| Artifact named in the paper | Location in this repository |
|---|---|
| `all_cbu_b64_summary.csv` | `all_cbu_b64_summary.csv` |
| `cbu_vqa_by_category_b64.json` | `cbu_vqa_by_category_b64.json` |
| `cpu_text_metrics/` | `raw_summaries/cpu_text_metrics/` |
| `prompt_support_bootstrap_b64_n2_250k_2026-04-24.tsv` (seven prompt pools) | `prompt_support_bootstrap_b64_n2_250k_2026-04-24.tsv`; `prompt_support_direction_summary.csv` aggregates the direction counts over all eight pools |
| `cc12m_budget_frontier_plot.csv` | `cc12m_budget_frontier_plot.csv` |
| `naive_qwen35_*/` | `naive_qwen35_cc12m/`, `naive_qwen35_sampled_cc12m/`, `naive_qwen35_datacomp/`, `naive_qwen35_sampled_datacomp/` |
| `croissant.json` per dataset | in the dataset repositories of the Hugging Face collection, not in this repository: each dataset carries its full Croissant 1.1 record (core, Responsible-AI, and provenance fields) as `croissant.json` at the repository root, and all nine records pass the `mlcroissant` validator; the Hub's Croissant endpoint of each dataset serves a core record generated from the data files |

## Policy control, verification, and sensitivity folders

| Directory | Contents |
|---|---|
| `naive_qwen35_sampled_cc12m/`, `naive_qwen35_cc12m/` | CC12M naive surfaces (matched decoding; greedy): 4,494 captions each as `captions.jsonl.gz`, claimed-CBU summary, and one VQA summary per judge |
| `naive_qwen35_sampled_datacomp/`, `naive_qwen35_datacomp/` | DataComp naive surfaces (matched decoding; greedy): 4,775 captions each, with the same summaries |
| `policy_control_ours_datacomp/` | the released (`Ours`) captions on the same 4,775 DataComp images: claimed-CBU summary and one VQA summary per judge, all on the Qwen3.5-397B-A17B-FP8 claims |
| `datacomp_pair/` | the DataComp verification run of the cross-corpus pair: one VQA summary per judge over Ours and Recap-DataComp |
| `sensitivity/` | lexical-window claimed CBU (CC12M), encoder-token truncation of the naive captions, and LongCLIP retrieval of the matched-decoding CC12M naive surface; the LongCLIP TSVs carry means and rates only |

`scripts/paper/gen_cbu_category_tables.py` reproduces the six files in `tables/` byte for byte from
`cbu_vqa_by_category_b64.json`.

Not part of the release:

- Row-level human-verification judgments and the census tables derived from them, which
  `scripts/human_cbu_eval.py export` writes from the private annotation database.
- The DataComp crawl-and-survival snapshot and the caption-to-caption kNN-cosine column of the DataComp
  encoder table.
