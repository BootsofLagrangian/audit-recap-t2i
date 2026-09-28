# Results

This directory holds the summary artifacts behind the tables and figures of the paper. Every file
is a byte-for-byte copy of the file with the same relative path in the manuscript's result bundle
(`artifacts/eval_results/` for the audit results, `artifacts/human_cbu/` for the human-study
screenshots). No number in this directory was edited by hand. To refresh a result, copy the updated
file from the bundle to the same relative path here.

Most files are written by a script in `scripts/`, as listed below. The top-level rollup tables
(`all_cbu_b64_summary.csv`, `all_vqa_b64_summary.csv`, `cc12m_vqa_supported_risk_pareto.csv`,
`cc12m_longclip_plot.csv`, `prompt_support_direction_summary.csv`) collect fields from the summary
files under `raw_summaries/` into one table each; the collection step has no separate script.

Several columns keep the names they had before the paper switched its wording from "token" to
"lexical unit": `avg_tokens`, `avg_lexical_tokens`, `cov64`, `elig64`, and `cbu_100tok` / `cbu_per_100tok`
all count regex lexical units (`[^\W_]+(?:'[^\W_]+)*`), not tokenizer tokens.

Surface names follow `configs/caption_survey/surfaces.json`: `ours_*` is the released recap surface,
`ref_*` a public reference surface. Cross-corpus rows are prefixed by their paired comparison, for
example `pd12m_full_paired__ours_pd12m_img2dataset`.

## Main text

| Paper element | File(s) | Produced by |
|---|---|---|
| Figure 1 (right), prompt-pool diagnostics | `teaser_right_v_twinx_v2.{pdf,png}` | `scripts/paper/gen_teaser_refined.py` (per-pair means written as literals; they equal the means over all eight pools in `prompt_support_bootstrap_b64_n2_250k_2026-04-24.tsv`) |
| Phenomenon descriptors (opener rate, top-100 raw / content prefix mass, Distinct-3, average length) and the per-family appendix companion | `raw_summaries/cpu_text_metrics/fair_slices_1m_normalized_2026-04-24.{json,tsv}`; LAION-pop re-run with URL-fixed pairing in `laion_pop_url_fixed_normalized_2026-04-24.{json,tsv}` | `scripts/run_caption_fair_slice_surveys.py` over the 1M-pair fair slices |
| Cross-corpus headline at B = 64: Avg lex | `raw_summaries/cpu_text_metrics/fair_slices_1m_normalized_2026-04-24.tsv` (`avg_tokens`) | as above |
| Cross-corpus headline: CBU/cap | `all_cbu_b64_summary.csv` (`cbu_cap`, `cbu_100tok`) | rollup of `raw_summaries/cbu_claimed/claimed_cbu_v2_all7_b64_5k.*.summary.json` (`scripts/summarize_cbu_responses.py --mode claimed`) |
| Cross-corpus headline: Pool-wins | `prompt_support_bootstrap_b64_n2_250k_2026-04-24.tsv` | `scripts/caption_prompt_support_bootstrap.py`; Pool-wins counts the seven pools other than `sd_prompts_dedup_xzuyn` with `delta_mean_local_minus_reference > 0` on `prompt_mass_on_caption_support` |
| Cross-corpus headline: Qwen Judge Sup. CBU/cap and Risk | `all_vqa_b64_summary.csv` (`supported_cap`, `risk`) | rollup of `raw_summaries/vqa_image_conditioned/*.summary.json` (`scripts/summarize_cbu_vqa_responses.py`) |
| Cross-corpus headline: Gemma Judge Sup. CBU/cap and Risk | `cbu_vqa_by_category_b64.json` (cells with `judge = Gemma-4-31B-IT`, `all_types`) | `scripts/paper/summarize_cbu_vqa_by_category.py` |
| Captioner-control tables (naive policy, CC12M and DataComp) | not in this snapshot, see below | `scripts/run_cc12m_naive_qwen35_baseline.sh`, `scripts/run_cc12m_naive_qwen35_gemma_metrics.sh`, `scripts/run_datacomp_naive_qwen35_qwen397_metrics.sh`, `scripts/export_cc12m_naive_qwen35_comparison_tables.py` |
| CC12M frontier at B = 64: CBU/cap and CBU/100lex | `cc12m_budget_frontier_plot.csv` (rows with `budget = 64`) | `scripts/paper/build_cc12m_budget_frontier_csv.py` |
| CC12M frontier at B = 64: Sup. CBU/cap and Risk under both judges | `cc12m_vqa_supported_risk_pareto.csv`; `all_vqa_b64_summary.csv` (`source = cc12m_qwen`, `cc12m_gemma`) | rollups of the two CC12M summaries in `raw_summaries/vqa_image_conditioned/` |
| Figure 2 (left), CC12M supported yield vs. risk | `cc12m_vqa_supported_risk_pareto_revised.{pdf,png}` | `scripts/paper/gen_cc12m_frontiers.py` from `cc12m_vqa_supported_risk_pareto.csv` |
| Figure 2 (right), CC12M budget sweep B in {16, 32, 48, 64} | `cc12m_cbu_efficiency_yield_frontier_revised.{pdf,png}` | `scripts/paper/gen_cc12m_frontiers.py` from `cc12m_budget_frontier_plot.csv` |
| Human evaluation of image support | aggregate tables not in this snapshot, see below | `scripts/human_cbu_eval.py export` |

## Appendix

| Paper element | File(s) | Produced by |
|---|---|---|
| Per-pool prompt-support heatmap | `per_pool_prompt_support_heatmap.{pdf,png}`; per-comparison direction counts over all eight pools in `prompt_support_direction_summary.csv` | `scripts/paper/gen_per_pool_heatmap.py` from `prompt_support_bootstrap_b64_n2_250k_2026-04-24.tsv` (seven pools) |
| VQA question denominators and B-eligibility | `all_vqa_b64_summary.csv` (`responses`, `questions`, `risk`); `raw_summaries/cpu_text_metrics/fair_slices_1m_normalized_2026-04-24.tsv` (`cov64`); paired deltas in `raw_summaries/cpu_text_metrics/paired_delta_ci.tsv` (`elig64`) | see above; `scripts/summarize_recap_fair_slice_cpu_remaining.py` for the paired deltas |
| CC12M denominators | `tables/cc12m_denominators.tex` | `scripts/paper/gen_cbu_category_tables.py` from `cbu_vqa_by_category_b64.json` |
| Cross-corpus headline without count and relation claims | `tables/excl_count_relation.tex` | `scripts/paper/gen_cbu_category_tables.py` |
| Image support and risk by claim type | `tables/vqa_by_type.tex` | `scripts/paper/gen_cbu_category_tables.py` |
| CC12M VQA bootstrap CIs | `cc12m_cbu_vqa_bootstrap_ci.tsv` (Qwen), `cc12m_gemma4_vqa_bootstrap_ci.tsv` (Gemma); identical copies in `raw_summaries/vqa_image_conditioned/` | 2,000-resample percentile bootstrap (the resampling script is not part of this release) |
| Human-study interface figures | `human_cbu/ui_appendix/*.png` | screenshots of the response-discarding participant-test mode of `scripts/human_cbu_eval.py` with an invented caption and a synthetic scene |
| DataComp text-space probes by encoder (Vendi, eRank, Coverage@10, Density@10) | `raw_summaries/embedding_vendi_support/caption_embedding_profile.tsv` (Vendi, eRank); `prompt_caption_support.tsv` (Coverage and Density, raw-text protocol rows `raw/raw` and, for BGE-M3, `raw/corpus`) | `scripts/caption_embedding_vendi.py` |
| EmbeddingGemma-300M multi-slice grid | `embeddinggemma_pair_summary.tsv`; `raw_summaries/embedding_vendi_support/embeddinggemma_all_pairs.tsv`, `embeddinggemma_dtype_sanity.json` | `scripts/caption_embedding_vendi.py` |
| LongCLIP retrieval on CC12M, full-caption and input64 modes | `cc12m_longclip_plot.csv` (both modes); `longclip_retrieval_summary.tsv` and `raw_summaries/longclip_retrieval/` (input64 mode) | `scripts/compute_longclip_retrieval_margin.py` |
| Encoder-token truncation rates | `raw_summaries/cpu_text_metrics/tokenizer_truncation_core_both_100k_2026-04-24.{json,tsv}` | `scripts/caption_tokenizer_truncation_survey.py` |
| Prompt-pool sensitivity (three-pool, eight-pool, DiffusionDB, disjoint caption pools) | `raw_summaries/prompt_support/*` | `scripts/caption_prompt_ngram_support.py`, `scripts/caption_prompt_support_pools.py`, `scripts/caption_prompt_support_bootstrap.py` |

## `raw_summaries/`

| Directory | Contents |
|---|---|
| `cpu_text_metrics/` | 1M-pair surveys of every paired slice, paired-difference CIs (`paired_delta_ci.tsv`), per-code violation rates (`violation_code_breakdown.tsv`, where `J_meta_statement` is the opener rate), a Re-LAION-Caption19M reference survey, tokenizer truncation, and a CPU sanity manifest for the GPU metrics |
| `prompt_support/` | hashed n-gram prompt-support and JSD runs at 250k caption records per surface against pools of up to 1M prompts |
| `cbu_claimed/` | claimed-CBU summaries at B = 64 for the cross-corpus 5k samples (`all7`, `completed4`, `cc12m3`), caption-level bootstrap CIs, and the CC12M budget sweep at B in {16, 32, 48} on all 4,494 aligned images (`*_4494.merged.summary.json`, `cc12m_budget_frontier_plot_4494.csv`), with the request rebuild check (`prepare_report.json`) and the 800-request rerun agreement check (`overlap_b32_rows0-199.comparison.json`) |
| `cbu_grounded_legacy/` | summaries of the earlier exact-unit grounded verification stage; kept for traceability and not used in the reported tables |
| `vqa_image_conditioned/` | Qwen Judge summaries for the cross-corpus pairs and CC12M, the Gemma Judge summary for CC12M, and the CC12M bootstrap CIs |
| `embedding_vendi_support/` | encoder-side diversity and prompt-to-caption support profiles |
| `longclip_retrieval/` | LongCLIP retrieval summary (input64 mode) |

`<LOCAL_CACHE>` and `<PROJECT_ROOT>` inside JSON files are placeholders for the machine-specific
directories the runs read from.

Two readings need care:

- The CC12M rows of `all_cbu_b64_summary.csv` come from the per-pair 5k CC12M samples. The CC12M
  case study in the paper uses the 4,494 images shared by all four CC12M surfaces; its claimed-CBU
  numbers are the `budget = 64` rows of `cc12m_budget_frontier_plot.csv` and the `claim_cbu_cap`
  column of `cc12m_vqa_supported_risk_pareto.csv`.
- The appendix artifact map names `cc12m_vqa_supported_risk_pareto.png` and
  `cc12m_cbu_efficiency_yield_frontier_by_budget.png`; the camera-ready renderings of these two
  panels are the `*_revised.{pdf,png}` files listed above.

## Not in this snapshot

- Captioner-control (naive policy) summaries for CC12M and DataComp: claimed CBU, both judges, text
  diagnostics, LongCLIP and truncation rows of the naive surface. The appendix artifact map names them
  `naive_qwen35_*/cbu_summary.csv` and `naive_qwen35_*/cpu_text_summary.json`.
- Aggregate tables of the human study. `scripts/human_cbu_eval.py export` writes them from a study
  database; the row-level database stays private.
- The DataComp crawl-and-survival snapshot and the kNN-cosine column of the DataComp encoder table.
