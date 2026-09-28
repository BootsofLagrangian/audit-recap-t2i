# `naive_qwen35_cc12m`

Captioning-policy control for CC12M. The captioner of the released corpus
(`Qwen/Qwen3.5-35B-A3B-FP8`) captions the 4,494 CC12M images of the four-surface slice with the
naive single-message prompt and no system prompt, using greedy decoding (temperature 0). This
surface is the `Naive (greedy)` row of the captioning-policy control in the paper.

```
Please generate a detailed caption of this image. Please be as descriptive as possible.
```

## Files

| Content | Produced by |
|---|---|
| captions, one JSON record per image (`naive_qwen35_cc12m.jsonl`) | `scripts/summarize_naive_vlm_captions.py --output-jsonl` |
| claimed-CBU summary at B = 64 (Qwen3.5-397B-A17B-FP8 extractor) | `scripts/summarize_cbu_responses.py --mode claimed` |
| VQA summary under the Qwen Judge | `scripts/summarize_cbu_vqa_responses.py` |
| VQA summary under the Gemma Judge | `scripts/summarize_cbu_vqa_responses.py` |

Both judges answer the questions built from the same Qwen3.5-397B-A17B-FP8 claims; neither judge
extracts claims itself. The captions are generated text; source images are not included.

## Regenerating

```bash
bash scripts/run_cc12m_naive_qwen35_baseline.sh
```

The baseline driver writes the captions and the claim-extraction requests. Extract the claims
with the Qwen3.5-397B-A17B-FP8 server, run the Qwen Judge on the VQA requests built from those
claims, and run `scripts/run_cc12m_naive_qwen35_gemma_metrics.sh` for the Gemma Judge on the
same requests (see the header of that script).
