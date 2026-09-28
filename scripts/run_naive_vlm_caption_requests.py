#!/usr/bin/env python3
"""Run no-system-prompt image caption requests against an OpenAI-compatible vLLM server."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--urls", default="http://localhost:8000")
    parser.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B-FP8")
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=None, help="Sent only when set")
    parser.add_argument("--top-p", type=float, default=None, help="Sent only when set")
    parser.add_argument("--timeout-sec", type=int, default=600)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--image-mode", choices=["auto", "file", "data", "url"], default="file")
    parser.add_argument("--target-resolution", type=int, default=1024)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def iter_requests(path: Path, max_requests: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if max_requests is not None and len(rows) >= max_requests:
                break
            if line.strip():
                rows.append(json.loads(line))
    return rows


def completed_ids(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("ok") and isinstance(row.get("request_id"), str):
                done.add(row["request_id"])
    return done


def image_url_for(row: dict[str, Any], args: argparse.Namespace) -> str:
    mode = args.image_mode
    path_text = row.get("image_path")
    url_text = row.get("image_url")
    if mode in {"auto", "file"} and isinstance(path_text, str) and path_text:
        return Path(path_text).resolve().as_uri()
    if mode in {"auto", "data"} and isinstance(path_text, str) and path_text:
        path = Path(path_text)
        with Image.open(path) as image:
            if image.mode != "RGB":
                image = image.convert("RGB")
            width, height = image.size
            target_pixels = args.target_resolution * args.target_resolution
            if width * height > target_pixels:
                scale = (target_pixels / (width * height)) ** 0.5
                size = (max(1, int(width * scale)), max(1, int(height * scale)))
                image = image.resize(size, Image.Resampling.LANCZOS)
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=args.jpeg_quality)
        return f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"
    if isinstance(url_text, str) and url_text:
        return url_text
    raise ValueError(f"request {row.get('request_id')} has no usable image")


def payload_for(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    payload = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": row["prompt"]},
                    {"type": "image_url", "image_url": {"url": image_url_for(row, args)}},
                ],
            }
        ],
        "chat_template_kwargs": {"enable_thinking": args.thinking},
    }
    if args.top_k is not None:
        payload["top_k"] = args.top_k
    if args.top_p is not None:
        payload["top_p"] = args.top_p
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
            caption = body["choices"][0]["message"]["content"].strip()
            return {
                "request_id": row["request_id"],
                "ok": bool(caption),
                "status": response.status,
                "elapsed_sec": round(elapsed, 4),
                "model": args.model,
                "caption": caption,
                "usage": body.get("usage", {}),
                "request": row,
            }
    except Exception as exc:  # noqa: BLE001
        return {
            "request_id": row.get("request_id"),
            "ok": False,
            "status": None,
            "elapsed_sec": round(time.perf_counter() - start, 4),
            "error": repr(exc),
            "request": row,
        }


async def run(args: argparse.Namespace) -> int:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = iter_requests(Path(args.input), args.max_requests)
    if args.resume:
        done = completed_ids(output)
        rows = [row for row in rows if row.get("request_id") not in done]
    urls = [item.strip() for item in args.urls.split(",") if item.strip()]
    timeout = aiohttp.ClientTimeout(total=args.timeout_sec)
    connector = aiohttp.TCPConnector(limit=args.concurrency)
    sem = asyncio.Semaphore(args.concurrency)
    ok = 0
    total = 0
    mode = "a" if args.resume else "w"
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        async def guarded(index: int, row: dict[str, Any]) -> dict[str, Any]:
            async with sem:
                return await post_one(session, urls[index % len(urls)], row, args)

        tasks = [asyncio.create_task(guarded(index, row)) for index, row in enumerate(rows)]
        with output.open(mode, encoding="utf-8") as handle:
            for future in asyncio.as_completed(tasks):
                result = await future
                total += 1
                ok += int(bool(result.get("ok")))
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                if total % 100 == 0 or total == len(tasks):
                    progress = {"completed": total, "queued": len(tasks), "ok": ok, "bad": total - ok}
                    print(json.dumps(progress), flush=True)
    summary = {"input": args.input, "output": args.output, "attempted": total, "ok": ok, "bad": total - ok}
    print(json.dumps(summary, indent=2))
    return 0


def main() -> int:
    return asyncio.run(run(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
