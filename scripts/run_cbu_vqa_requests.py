#!/usr/bin/env python3
"""Run VQA-style CBU question requests against an OpenAI-compatible VLM server."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import aiohttp
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

ANSWERS = ["yes", "no", "uncertain"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CBU VQA requests")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--urls", default="http://localhost:8000")
    parser.add_argument("--model", default="Qwen/Qwen3.5-397B-A17B-FP8")
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-sec", type=int, default=2400)
    parser.add_argument("--image-mode", choices=["auto", "file", "data", "url"], default="file")
    parser.add_argument("--structured-json", action="store_true")
    parser.add_argument(
        "--no-evidence",
        action="store_true",
        help="Use compact answer-only schema: question_id, answer, confidence.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-ok-only", action="store_true")
    parser.add_argument("--skip-ok-from", default=None)
    return parser.parse_args()


def iter_requests(path: Path, max_requests: int | None) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if max_requests is not None and len(rows) >= max_requests:
                break
            if line.strip():
                rows.append(json.loads(line))
    return rows


def image_url_for(row: dict[str, Any], mode: str) -> str:
    if mode in {"auto", "data"} and row.get("image_path"):
        path = Path(row["image_path"])
        with Image.open(path) as image:
            if image.mode != "RGB":
                image = image.convert("RGB")
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=88)
        return f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"
    if mode in {"auto", "file"} and row.get("image_path"):
        return Path(row["image_path"]).resolve().as_uri()
    if mode == "file":
        raise ValueError(f"request {row.get('request_id')} has no image_path")
    return row["image_url"]


def response_schema(question_ids: list[str], include_evidence: bool) -> dict[str, Any]:
    item_properties: dict[str, Any] = {
        "question_id": {"type": "string", "enum": question_ids},
        "answer": {"type": "string", "enum": ANSWERS},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    }
    required = ["question_id", "answer", "confidence"]
    if include_evidence:
        item_properties["evidence"] = {"type": "string", "maxLength": 160}
        required.append("evidence")
    return {
        "type": "object",
        "properties": {
            "caption_id": {"type": "string"},
            "question_results": {
                "type": "array",
                "minItems": len(question_ids),
                "maxItems": len(question_ids),
                "items": {
                    "type": "object",
                    "properties": item_properties,
                    "required": required,
                    "additionalProperties": False,
                },
            },
        },
        "required": ["caption_id", "question_results"],
        "additionalProperties": False,
    }


def validate(parsed: Any, row: dict[str, Any], include_evidence: bool) -> str | None:
    if not isinstance(parsed, dict):
        return "top-level response is not an object"
    if not isinstance(parsed.get("caption_id"), str):
        return "caption_id is not a string"
    results = parsed.get("question_results")
    if not isinstance(results, list):
        return "question_results is not an array"
    expected = [question["question_id"] for question in row.get("questions", [])]
    seen = []
    for index, result in enumerate(results):
        if not isinstance(result, dict):
            return f"question_results[{index}] is not an object"
        question_id = result.get("question_id")
        if not isinstance(question_id, str):
            return f"question_results[{index}].question_id is not a string"
        seen.append(question_id)
        if result.get("answer") not in set(ANSWERS):
            return f"question_results[{index}].answer has invalid value"
        if not isinstance(result.get("confidence"), int | float):
            return f"question_results[{index}].confidence is not numeric"
        if include_evidence and not isinstance(result.get("evidence"), str):
            return f"question_results[{index}].evidence is not a string"
    if sorted(seen) != sorted(expected):
        return f"question_id set mismatch: expected={len(expected)} seen={len(seen)}"
    if len(seen) != len(set(seen)):
        return "duplicate question_id in response"
    return None


def payload_for(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    question_ids = [question["question_id"] for question in row.get("questions", [])]
    user_prompt = row["user_prompt"]
    if args.no_evidence:
        user_prompt = user_prompt.replace(
            "- Keep evidence short and grounded in visible image content.\n",
            "- Return only question_id, answer, and confidence for each question; do not include evidence text.\n",
        )
    payload: dict[str, Any] = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "messages": [
            {"role": "system", "content": row["system_prompt"]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": image_url_for(row, args.image_mode)}},
                ],
            },
        ],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if args.structured_json:
        payload["structured_outputs"] = {"json": response_schema(question_ids, include_evidence=not args.no_evidence)}
    return payload


async def post_one(
    session: aiohttp.ClientSession,
    url: str,
    row: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    endpoint = f"{url.rstrip('/')}/v1/chat/completions"
    start = time.perf_counter()
    try:
        async with session.post(
            endpoint,
            json=payload_for(row, args),
            headers={"Authorization": "Bearer sk-fake"},
        ) as response:
            text = await response.text()
            elapsed = time.perf_counter() - start
            if response.status >= 400:
                return {
                    "request_id": row["request_id"],
                    "ok": False,
                    "status": response.status,
                    "elapsed_sec": round(elapsed, 4),
                    "error": text[:4000],
                    "request": row,
                }
            body = json.loads(text)
            content = body["choices"][0]["message"]["content"]
            parsed = None
            parse_error = None
            schema_error = None
            try:
                parsed = json.loads(content)
                schema_error = validate(parsed, row, include_evidence=not args.no_evidence)
            except Exception as exc:  # noqa: BLE001
                parse_error = repr(exc)
            return {
                "request_id": row["request_id"],
                "ok": parse_error is None and schema_error is None,
                "status": response.status,
                "elapsed_sec": round(elapsed, 4),
                "model": args.model,
                "usage": body.get("usage", {}),
                "response_text": content,
                "parsed": parsed,
                "parse_error": parse_error,
                "schema_error": schema_error,
                "request": row,
            }
    except Exception as exc:  # noqa: BLE001
        return {
            "request_id": row["request_id"],
            "ok": False,
            "status": None,
            "elapsed_sec": round(time.perf_counter() - start, 4),
            "error": repr(exc),
            "request": row,
        }


def load_seen(args: argparse.Namespace, output: Path) -> set[str]:
    seen: set[str] = set()
    paths: list[Path] = []
    if args.skip_ok_from:
        paths.append(Path(args.skip_ok_from))
    if args.resume and output.exists():
        paths.append(output)
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (path != output or args.resume_ok_only) and not row.get("ok"):
                    continue
                request_id = row.get("request_id")
                if isinstance(request_id, str):
                    seen.add(request_id)
    return seen


async def run(args: argparse.Namespace) -> int:
    rows = iter_requests(Path(args.input), args.max_requests)
    urls = [item.strip() for item in args.urls.split(",") if item.strip()]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    seen_request_ids = load_seen(args, output)
    rows = [row for row in rows if row.get("request_id") not in seen_request_ids]
    timeout = aiohttp.ClientTimeout(total=args.timeout_sec)
    connector = aiohttp.TCPConnector(limit=args.concurrency)
    sem = asyncio.Semaphore(args.concurrency)
    ok = 0
    total = 0
    mode = "a" if args.resume else "w"
    with output.open(mode, encoding="utf-8") as handle:
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            async def guarded(index: int, row: dict[str, Any]) -> dict[str, Any]:
                async with sem:
                    return await post_one(session, urls[index % len(urls)], row, args)

            tasks = [asyncio.create_task(guarded(index, row)) for index, row in enumerate(rows)]
            for task in asyncio.as_completed(tasks):
                result = await task
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                total += 1
                ok += int(bool(result.get("ok")))
                if total % 10 == 0 or total == len(rows):
                    print(
                        json.dumps(
                            {
                                "completed": total,
                                "ok": ok,
                                "total": len(rows),
                                "skipped_existing": len(seen_request_ids),
                            },
                            ensure_ascii=False,
                        )
                    )
    print(
        json.dumps(
            {"output": str(output), "completed": total, "ok": ok, "skipped_existing": len(seen_request_ids)},
            indent=2,
        )
    )
    return 0


def main() -> int:
    return asyncio.run(run(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
