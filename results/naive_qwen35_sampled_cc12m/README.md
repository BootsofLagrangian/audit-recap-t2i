# `naive_qwen35_sampled_cc12m`

Captioning-policy control for CC12M. The captioner of the released corpus
(`Qwen/Qwen3.5-35B-A3B-FP8`) captions the 4,494 CC12M images shared by the four surfaces of the
CC12M case study with the naive single-message prompt and no system prompt, using matched decoding,
the captioner's release sampling defaults (temperature 1.0, top_k 20, top_p 0.95) that also produced
the released captions. This surface is the `Naive` row of the captioning-policy control in the
paper.

```
Please generate a detailed caption of this image. Please be as descriptive as possible.
```

## Files

| File | Content | Produced by |
|---|---|---|
| `captions.jsonl.gz` | the 4,494 captions, one gzip-compressed JSON record per image | `scripts/summarize_naive_vlm_captions.py --output-jsonl` |
| `claimed_cbu_summary.json` | claimed-CBU summary at B = 64 (Qwen3.5-397B-A17B-FP8 extractor) | `scripts/summarize_cbu_responses.py --mode claimed` |
| `vqa_summary_qwen397.json` | VQA summary under the Qwen Judge | `scripts/summarize_cbu_vqa_responses.py` |
| `vqa_summary_gemma4_31b_it.json` | VQA summary under the Gemma Judge | `scripts/summarize_cbu_vqa_responses.py` |

Both judges answer the questions built from the same Qwen3.5-397B-A17B-FP8 claims; neither judge
extracts claims itself.

| Claimed CBU/cap | CBU/100 lex | Qwen Judge Sup. CBU/cap | Qwen Judge risk | Gemma Judge Sup. CBU/cap | Gemma Judge risk |
|---:|---:|---:|---:|---:|---:|
| 11.32 | 17.26 | 11.03 ± 0.05 | 0.022 ± 0.001 | 10.61 ± 0.05 | 0.045 ± 0.001 |

Claimed CBU/cap and CBU/100 lex are over the 4,458 captions with a valid extractor response
(`claimed_cbu_summary.json`). The judge cells are the `CC12M-control` / `Naive` cells of
`../cbu_vqa_by_category_b64.json`, on the requests answered by both judges, as mean ± std over 2,000
caption-level bootstrap resamples (seed 0); `../tables/vqa_mean_std.tex` prints the same values.
Each judge's summary file counts the same 4,397 answered responses (50,450 CBU questions) as those
cells.

## Reading the captions

```python
import gzip, json

with gzip.open("results/naive_qwen35_sampled_cc12m/captions.jsonl.gz", "rt", encoding="utf-8") as f:
    records = [json.loads(line) for line in f]
print(len(records), records[0]["caption"][:200])
```

From a shell: `zcat results/naive_qwen35_sampled_cc12m/captions.jsonl.gz | head -n 1`.

Each record has the fields `surface`, `caption_id`, `source_row`, `family`, `pair_key`,
`public_lookup_key`, `image_url`, `prompt`, `system_prompt` (null), `messages_policy`, `caption`,
and `decoding`. The records carry public keys only: `image_url` and `public_lookup_key` are the
public image URL, and `pair_key` is the CC12M numeric key of the source release (source-local; use
the URL to match across releases). Source images are not included; obtain them from the original
release under its own terms. Personal data that the captioner transcribed from text in the images is
masked with placeholders: `[email]` for email addresses (3 in this file), `[phone]` for phone
numbers (4), `[name]` for names of private persons printed on ID cards, badges, name tags, race
bibs, and certificates (6), `[id]` for identity and card numbers (1), and `[address]` for street
addresses of private persons (0). Names of public figures and of fictional characters are kept.

## License

The caption text in `captions.jsonl.gz` is released under CC-BY-4.0, the same license as the
released captions on Hugging Face; it covers the generated text only. The code that produced it is
Apache-2.0 (see `LICENSE` at the repository root). Source images and their URLs remain subject to
the terms of the original release.

## Regenerating

```bash
SURFACE=naive_qwen35_sampled_cc12m CAPTION_TEMPERATURE=1.0 CAPTION_TOP_K=20 CAPTION_TOP_P=0.95 \
  RUN_ROOT=artifacts/recap-ed/cc12m-naive-qwen35-sampled bash scripts/run_cc12m_naive_qwen35_baseline.sh
```

The baseline driver writes the captions and the claim-extraction requests. Extract the claims with
the Qwen3.5-397B-A17B-FP8 server, run the Qwen Judge on the VQA requests built from those claims,
and run `scripts/run_cc12m_naive_qwen35_gemma_metrics.sh` with `SURFACE=naive_qwen35_sampled_cc12m`
for the Gemma Judge on the same requests (see the header of that script).
