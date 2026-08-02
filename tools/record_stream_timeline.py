#!/usr/bin/env python3
"""Record per-stream token arrival times for a reproducible throughput video."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import requests
from transformers import AutoTokenizer

from profile_request import build_exact_prompt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--served-name", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--prompt-tokens", type=int, default=4096)
    parser.add_argument("--gen-tokens", type=int, default=1024)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prompt-salt", default="video")
    parser.add_argument("--read-timeout", type=float, default=1800.0)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    prompt, prompt_tokens = build_exact_prompt(
        tokenizer,
        args.prompt_tokens,
        image=False,
        pure_filler=True,
        prompt_salt=args.prompt_salt,
    )
    payload: dict[str, Any] = {
        "model": args.served_name,
        "prompt": prompt,
        "max_tokens": args.gen_tokens,
        "temperature": 0.0,
        "stream": True,
        "ignore_eos": True,
        "return_token_ids": True,
    }
    start = time.perf_counter()
    first: float | None = None
    end: float | None = None
    total_tokens = 0
    chunks = 0
    events: list[dict[str, Any]] = []
    status: int | None = None
    error: str | None = None
    stream_done = False
    try:
        with requests.post(
            f"{args.base_url.rstrip('/')}/completions",
            json=payload,
            stream=True,
            timeout=(30, args.read_timeout),
        ) as response:
            status = response.status_code
            response.raise_for_status()
            for raw in response.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data: "):
                    continue
                data = raw[6:]
                if data == "[DONE]":
                    stream_done = True
                    break
                now = time.perf_counter()
                obj = json.loads(data)
                choice = (obj.get("choices") or [{}])[0]
                ids = choice.get("token_ids")
                n = len(ids) if isinstance(ids, list) else 0
                if first is None:
                    first = now
                total_tokens += n
                chunks += 1
                events.append(
                    {
                        "t_s": now - start,
                        "decode_t_s": now - first,
                        "tokens": total_tokens,
                        "chunk_tokens": n,
                    }
                )
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    end = time.perf_counter()
    ttft_s = None if first is None else first - start
    elapsed_s = end - start
    decode_s = None if first is None else max(end - first, 1e-9)
    result = {
        "schema_version": 1,
        "label": args.label,
        "model": args.served_name,
        "model_dir": args.model_dir,
        "base_url": args.base_url,
        "prompt_tokens": prompt_tokens,
        "requested_prompt_tokens": args.prompt_tokens,
        "requested_completion_tokens": args.gen_tokens,
        "completion_tokens": total_tokens,
        "completion_token_source": "token_ids",
        "http_status": status,
        "error": error,
        "stream_done": stream_done,
        "chunks": chunks,
        "ttft_s": ttft_s,
        "elapsed_s": elapsed_s,
        "decode_s": decode_s,
        "decode_tok_s": None if decode_s is None else total_tokens / decode_s,
        "events": events,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("label", "prompt_tokens", "completion_tokens", "ttft_s", "decode_s", "decode_tok_s", "error")}, ensure_ascii=False))
    if error or total_tokens == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
