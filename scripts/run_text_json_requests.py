#!/usr/bin/env python3
"""Run text-only structured JSON requests against OpenAI-compatible endpoints."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

import aiohttp
from build_caption_cbu_requests import CBU_JSON_SCHEMA, UNIT_CATEGORIES


def request_schema(row: dict[str, Any]) -> dict[str, Any]:
    manifest_schema = row.get("schema")
    if isinstance(manifest_schema, dict):
        return manifest_schema
    prompt = row.get("user_prompt", "")
    if isinstance(prompt, str):
        marker = "Return only JSON matching this schema:\n"
        if marker in prompt:
            rest = prompt.split(marker, 1)[1]
            schema_text = rest.split("\n\n", 1)[0]
            try:
                parsed = json.loads(schema_text)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:  # noqa: BLE001
                pass
    return CBU_JSON_SCHEMA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run text-only JSON-schema requests")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--urls", default="http://localhost:8000")
    parser.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B")
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-sec", type=int, default=240)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--structured-json", action="store_true")
    parser.add_argument("--response-format-schema", action="store_true")
    parser.add_argument("--response-format-json", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Append to output and skip previously seen request_ids.")
    parser.add_argument(
        "--resume-ok-only",
        action="store_true",
        help="With --resume, skip only previously successful request_ids so failures are retried.",
    )
    parser.add_argument(
        "--skip-ok-from",
        default=None,
        help="JSONL response log whose successful request_ids should be skipped while writing a separate output.",
    )
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


def validate_cbu_response(parsed: Any) -> str | None:
    if not isinstance(parsed, dict):
        return "top-level response is not an object"
    if not isinstance(parsed.get("caption_id"), str):
        return "caption_id is not a string"
    claimed = parsed.get("claimed_units")
    if not isinstance(claimed, list):
        return "claimed_units is not an array"
    for index, unit in enumerate(claimed):
        if not isinstance(unit, dict):
            return f"claimed_units[{index}] is not an object"
        extra = sorted(set(unit) - {"category", "unit", "span", "target"})
        if extra:
            return f"claimed_units[{index}] has unexpected fields: {extra}"
        missing = [field for field in ["category", "unit", "span", "target"] if field not in unit]
        if missing:
            return f"claimed_units[{index}] is missing fields: {missing}"
        if unit["category"] not in UNIT_CATEGORIES:
            return f"claimed_units[{index}].category has invalid value"
        for field in ["unit", "span", "target"]:
            if not isinstance(unit[field], str):
                return f"claimed_units[{index}].{field} is not a string"
    return None


def payload_for(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "messages": [
            {"role": "system", "content": row["system_prompt"]},
            {"role": "user", "content": row["user_prompt"]},
        ],
        "chat_template_kwargs": {"enable_thinking": args.thinking},
    }
    if args.structured_json:
        payload["structured_outputs"] = {"json": request_schema(row)}
    if args.response_format_schema:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "claimed_cbu", "schema": request_schema(row)},
        }
    if args.response_format_json:
        payload["response_format"] = {"type": "json_object"}
    return payload


async def post_one(
    session: aiohttp.ClientSession,
    url: str,
    row: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    endpoint = f"{url.rstrip('/')}/v1/chat/completions"
    payload = payload_for(row, args)
    start = time.perf_counter()
    try:
        async with session.post(endpoint, json=payload, headers={"Authorization": "Bearer sk-fake"}) as response:
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
                schema_error = validate_cbu_response(parsed)
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


async def run(args: argparse.Namespace) -> int:
    rows = iter_requests(Path(args.input), args.max_requests)
    urls = [item.strip() for item in args.urls.split(",") if item.strip()]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    seen_request_ids: set[str] = set()
    if args.skip_ok_from:
        with Path(args.skip_ok_from).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not row.get("ok"):
                    continue
                request_id = row.get("request_id")
                if isinstance(request_id, str):
                    seen_request_ids.add(request_id)
    if args.resume and output.exists():
        with output.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if args.resume_ok_only and not row.get("ok"):
                    continue
                request_id = row.get("request_id")
                if isinstance(request_id, str):
                    seen_request_ids.add(request_id)
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
                if total % 100 == 0 or total == len(rows):
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
