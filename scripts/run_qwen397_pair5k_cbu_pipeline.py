#!/usr/bin/env python3
"""Run pair-specific 5k CBU extraction, grounding, and VQA stages.

The pipeline keeps file names comparison-specific because the same local surface
can have different matched-image slices against different public references.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


# Interpreter for the stage scripts; override with AUDIT_PYTHON to use another environment.
PY = os.environ.get("AUDIT_PYTHON", sys.executable)
MODEL = "Qwen/Qwen3.5-397B-A17B-FP8"
URL = "http://localhost:8000"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run qwen397 pair5k CBU pipeline")
    parser.add_argument("--manifest", default="artifacts/recap-ed/provenance-2026-04-25/manifest.json")
    parser.add_argument("--budget", choices=["b64", "full"], required=True)
    parser.add_argument(
        "--stage",
        choices=["text", "grounded", "vqa", "all"],
        default="all",
        help="Run one stage or all stages for already-built pair5k requests.",
    )
    parser.add_argument("--concurrency-text", type=int, default=1024)
    parser.add_argument("--concurrency-grounded", type=int, default=768)
    parser.add_argument("--concurrency-vqa", type=int, default=384)
    parser.add_argument("--max-tokens-text", type=int, default=4096)
    parser.add_argument("--max-tokens-grounded", type=int, default=4096)
    parser.add_argument("--max-tokens-vqa", type=int, default=2048)
    parser.add_argument("--timeout-text", type=int, default=1800)
    parser.add_argument("--timeout-grounded", type=int, default=2400)
    parser.add_argument("--timeout-vqa", type=int, default=2400)
    parser.add_argument("--image-mode", choices=["auto", "file", "data", "url"], default="auto")
    parser.add_argument("--skip-danbooru-grounded", action="store_true")
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def entries(manifest_path: Path) -> list[dict[str, str]]:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    out: list[dict[str, str]] = []
    for comparison in data["comparisons"]:
        comp = comparison["comparison"]
        outputs = comparison["summary"]["outputs"]
        for role, path_key, surface in [
            ("ours", "local_jsonl", comparison["summary"]["local_surface"]),
            ("ref", "reference_jsonl", comparison["summary"]["reference_surface"]),
        ]:
            out.append(
                {
                    "comparison": comp,
                    "role": role,
                    "surface": surface,
                    "source_jsonl": outputs[path_key],
                    "tag": f"{comp}__{surface}",
                }
            )
    return out


def paths(entry: dict[str, str], budget: str) -> dict[str, Path]:
    cbu_dir = Path("artifacts/cbu/pair5k")
    grounded_dir = Path("artifacts/grounded-cbu/pair5k")
    vqa_dir = Path("artifacts/vqa-cbu/pair5k")
    base = f"claimed_cbu_v2_{entry['tag']}_{budget}_5k"
    grounded = f"grounded_verify_v2_{entry['tag']}_{budget}_5k"
    vqa = f"cbu_vqa_{entry['tag']}_{budget}_5k"
    return {
        "claimed_request": cbu_dir / f"{base}.requests.jsonl",
        "claimed_response": cbu_dir / f"{base}.responses.qwen397_c1024_mt4096.jsonl",
        "grounded_request": grounded_dir / f"{grounded}.requests.qwen397.jsonl",
        "grounded_response": grounded_dir / f"{grounded}.responses.qwen397_c768_file_mt4096.jsonl",
        "vqa_request": vqa_dir / f"{vqa}.requests.jsonl",
        "vqa_response": vqa_dir / f"{vqa}.responses.qwen397_c384_file_mt2048_compact.jsonl",
    }


def run_text(entry: dict[str, str], p: dict[str, Path], args: argparse.Namespace) -> None:
    run(
        [
            PY,
            "scripts/run_text_json_requests.py",
            "--input",
            str(p["claimed_request"]),
            "--output",
            str(p["claimed_response"]),
            "--urls",
            URL,
            "--model",
            MODEL,
            "--concurrency",
            str(args.concurrency_text),
            "--max-tokens",
            str(args.max_tokens_text),
            "--timeout-sec",
            str(args.timeout_text),
            "--structured-json",
            "--resume",
            "--resume-ok-only",
        ]
    )
    run(
        [
            PY,
            "scripts/summarize_cbu_responses.py",
            "--latest-by-request",
            "--mode",
            "claimed",
            "--input",
            str(p["claimed_response"]),
            "--output",
            str(p["claimed_response"]).replace(".jsonl", ".summary.json"),
        ]
    )


def run_grounded(entry: dict[str, str], p: dict[str, Path], args: argparse.Namespace) -> None:
    p["grounded_request"].parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            PY,
            "scripts/build_grounded_cbu_verify_requests.py",
            "--claimed-responses",
            str(p["claimed_response"]),
            "--source-jsonl",
            entry["source_jsonl"],
            "--output",
            str(p["grounded_request"]),
        ]
    )
    run(
        [
            PY,
            "scripts/run_grounded_cbu_verify_requests.py",
            "--input",
            str(p["grounded_request"]),
            "--output",
            str(p["grounded_response"]),
            "--urls",
            URL,
            "--model",
            MODEL,
            "--concurrency",
            str(args.concurrency_grounded),
            "--max-tokens",
            str(args.max_tokens_grounded),
            "--timeout-sec",
            str(args.timeout_grounded),
            "--image-mode",
            args.image_mode,
            "--structured-json",
            "--resume",
            "--resume-ok-only",
        ]
    )
    run(
        [
            PY,
            "scripts/summarize_grounded_cbu_verify.py",
            "--latest-by-request",
            "--input",
            str(p["grounded_response"]),
            "--output",
            str(p["grounded_response"]).replace(".jsonl", ".summary.json"),
        ]
    )


def run_vqa(entry: dict[str, str], p: dict[str, Path], args: argparse.Namespace) -> None:
    p["vqa_request"].parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            PY,
            "scripts/build_cbu_vqa_requests.py",
            "--input",
            str(p["grounded_request"]),
            "--output",
            str(p["vqa_request"]),
        ]
    )
    run(
        [
            PY,
            "scripts/run_cbu_vqa_requests.py",
            "--input",
            str(p["vqa_request"]),
            "--output",
            str(p["vqa_response"]),
            "--urls",
            URL,
            "--model",
            MODEL,
            "--concurrency",
            str(args.concurrency_vqa),
            "--max-tokens",
            str(args.max_tokens_vqa),
            "--timeout-sec",
            str(args.timeout_vqa),
            "--image-mode",
            args.image_mode,
            "--structured-json",
            "--no-evidence",
            "--resume",
            "--resume-ok-only",
        ]
    )
    run(
        [
            PY,
            "scripts/summarize_cbu_vqa_responses.py",
            "--latest-by-request",
            "--input",
            str(p["vqa_response"]),
            "--output",
            str(p["vqa_response"]).replace(".jsonl", ".summary.json"),
        ]
    )


def main() -> int:
    args = parse_args()
    for entry in entries(Path(args.manifest)):
        if args.skip_danbooru_grounded and entry["comparison"].startswith("danbooru") and args.stage in {"grounded", "vqa", "all"}:
            print(f"skip grounded/vqa for {entry['tag']} until danbooru image path mapping is verified", flush=True)
            if args.stage == "text":
                pass
            else:
                continue
        p = paths(entry, args.budget)
        if args.stage in {"text", "all"}:
            run_text(entry, p, args)
        if args.stage in {"grounded", "all"}:
            run_grounded(entry, p, args)
        if args.stage in {"vqa", "all"}:
            run_vqa(entry, p, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
