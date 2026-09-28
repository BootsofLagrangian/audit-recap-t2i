# `naive_qwen35_sampled_datacomp`

Captioning-policy control for DataComp. The captioner of the released corpus
(`Qwen/Qwen3.5-35B-A3B-FP8`) captions the DataComp images of the control slice with the naive
single-message prompt and no system prompt, using matched decoding, that is the captioner's
release sampling defaults (temperature 1.0, top_k 20, top_p 0.95) used for the released
captions. This surface is the `Naive` row of the captioning-policy control in the paper.

```
Please generate a detailed caption of this image. Please be as descriptive as possible.
```

## Files

| Content | Produced by |
|---|---|
| captions, one JSON record per image (`naive_qwen35_sampled_datacomp.jsonl`) | `scripts/summarize_naive_vlm_captions.py --output-jsonl` |
| claimed-CBU summary at B = 64 (Qwen3.5-397B-A17B-FP8 extractor) | `scripts/summarize_cbu_responses.py --mode claimed` |
| VQA summary under the Qwen Judge | `scripts/summarize_cbu_vqa_responses.py` |
| VQA summary under the Gemma Judge | `scripts/summarize_cbu_vqa_responses.py` |

Both judges answer the questions built from the same Qwen3.5-397B-A17B-FP8 claims; neither judge
extracts claims itself. The captions are generated text; source images are not included.

## Regenerating

Generate the captions with `scripts/build_naive_vlm_caption_requests.py`,
`scripts/run_naive_vlm_caption_requests.py` (`--temperature 1.0 --top-k 20 --top-p 0.95`), and
`scripts/summarize_naive_vlm_captions.py` as in the CC12M baseline driver.
`scripts/run_datacomp_naive_qwen35_qwen397_metrics.sh` then runs the claim extraction and the
Qwen Judge; its header gives the Gemma Judge command on the same VQA requests.
