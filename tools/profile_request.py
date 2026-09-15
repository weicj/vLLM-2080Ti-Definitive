#!/usr/bin/env python3
import argparse
import base64
import json
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import requests
from transformers import AutoTokenizer


def build_exact_prompt(
    tokenizer: Any,
    target_tokens: int,
    *,
    image: bool,
    pure_filler: bool,
) -> tuple[str, int]:
    if target_tokens <= 0:
        return "Reply with OK.", len(tokenizer.encode("Reply with OK.", add_special_tokens=False))

    if pure_filler:
        prefix = ""
        suffix = ""
    elif image:
        prefix = (
            "Please read the long filler text first. The filler is irrelevant. "
            "Only answer the final image question.\nFILLER START\n"
        )
        suffix = (
            "\nFILLER END\nLook at the image and answer in English: "
            "left shape/color, right shape/color, and visible code text."
        )
    else:
        prefix = "Long filler text follows. FILLER START\n"
        suffix = "\nFILLER END\nReply with exactly: PROFILE_OK"

    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    filler_tokens = max(1, target_tokens - len(prefix_ids) - len(suffix_ids))
    prompt = prefix + (" the" * filler_tokens) + suffix
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(ids) > target_tokens:
        ids = ids[:target_tokens]
        prompt = tokenizer.decode(ids, skip_special_tokens=False)
    return prompt, len(tokenizer.encode(prompt, add_special_tokens=False))


def monitor_gpu(stop: threading.Event, path: Path, interval: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        while not stop.is_set():
            try:
                proc = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,name,pstate,memory.used,memory.free,utilization.gpu",
                        "--format=csv,noheader",
                    ],
                    text=True,
                    capture_output=True,
                    timeout=5,
                    check=False,
                )
                fh.write(f"epoch={time.time():.3f}\n{proc.stdout}\n")
                if proc.stderr:
                    fh.write(f"stderr={proc.stderr}\n")
                fh.flush()
            except Exception as exc:  # noqa: BLE001
                fh.write(f"monitor_error={type(exc).__name__}: {exc}\n")
                fh.flush()
            stop.wait(interval)


def parse_gpu_log(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    stats: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line or line.startswith("epoch=") or line.startswith("stderr="):
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 6 or not parts[0].isdigit():
            continue
        idx = parts[0]
        try:
            used = int(parts[3].split()[0])
            free = int(parts[4].split()[0])
            util = int(parts[5].split()[0])
        except (IndexError, ValueError):
            continue
        row = stats.setdefault(idx, {"samples": 0, "used_mib": [], "free_mib": [], "util_pct": []})
        row["samples"] += 1
        row["used_mib"].append(used)
        row["free_mib"].append(free)
        row["util_pct"].append(util)

    summary: dict[str, Any] = {}
    for idx, row in stats.items():
        used_values = row["used_mib"]
        free_values = row["free_mib"]
        util_values = row["util_pct"]
        summary[idx] = {
            "samples": row["samples"],
            "used_mib_min": min(used_values),
            "used_mib_max": max(used_values),
            "free_mib_min": min(free_values),
            "free_mib_max": max(free_values),
            "util_pct_max": max(util_values),
        }
    return summary


def stream_request(url: str, payload: dict[str, Any], endpoint: str, timeout: float) -> dict[str, Any]:
    start = time.perf_counter()
    first = None
    status = None
    error = None
    stream_done = False
    chunks = 0
    text_parts: list[str] = []
    completion_token_ids: list[int] = []
    usage = None
    raw_preview: list[str] = []

    try:
        with requests.post(f"{url}/{endpoint}", json=payload, stream=True, timeout=(30, timeout)) as resp:
            status = resp.status_code
            resp.raise_for_status()
            for raw in resp.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data: "):
                    continue
                data = raw[6:]
                if len(raw_preview) < 8:
                    raw_preview.append(data[:500])
                if data == "[DONE]":
                    stream_done = True
                    break
                now = time.perf_counter()
                if first is None:
                    first = now
                chunks += 1
                obj = json.loads(data)
                if obj.get("usage"):
                    usage = obj.get("usage")
                choice = (obj.get("choices") or [{}])[0]
                if endpoint == "completions":
                    token_ids = choice.get("token_ids")
                    if isinstance(token_ids, list):
                        completion_token_ids.extend(token_ids)
                    text = choice.get("text")
                else:
                    delta = choice.get("delta") or {}
                    text = delta.get("content")
                if isinstance(text, str):
                    text_parts.append(text)
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"

    end = time.perf_counter()
    return {
        "http_status": status,
        "error": error,
        "stream_done": stream_done,
        "chunks": chunks,
        "text": "".join(text_parts),
        "completion_tokens_from_ids": len(completion_token_ids),
        "usage": usage,
        "raw_events_preview": raw_preview,
        "ttft_s": None if first is None else first - start,
        "elapsed_s": end - start,
    }


def image_card_correct(text: str) -> bool:
    lower = text.lower()
    return (
        "blue" in lower
        and "square" in lower
        and ("orange" in lower or "yellow" in lower)
        and "circle" in lower
        and "k7p" in lower
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--served-name", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument(
        "--endpoint",
        choices=("completions", "chat-text", "chat-image"),
        default="chat-text",
    )
    parser.add_argument("--prompt-tokens", type=int, required=True)
    parser.add_argument("--gen-tokens", type=int, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpu-log", type=Path)
    parser.add_argument("--gpu-interval", type=float, default=5.0)
    parser.add_argument("--read-timeout", type=float, default=1800.0)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument(
        "--pure-filler",
        action="store_true",
        help='Use only repeated " the" tokens. This is for synthetic throughput, not correctness.',
    )
    parser.add_argument("--image-path", type=Path)
    parser.add_argument("--expect-image-card", action="store_true")
    args = parser.parse_args()

    prepare_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    is_image = args.endpoint == "chat-image"
    if args.pure_filler and args.endpoint != "completions":
        raise SystemExit("--pure-filler is only valid with --endpoint completions")
    prompt, prompt_tokens = build_exact_prompt(
        tokenizer,
        args.prompt_tokens,
        image=is_image,
        pure_filler=args.pure_filler,
    )
    prepare_s = time.perf_counter() - prepare_start

    if args.endpoint == "completions":
        payload: dict[str, Any] = {
            "model": args.served_name,
            "prompt": prompt,
            "max_tokens": args.gen_tokens,
            "temperature": 0.0,
            "stream": True,
            "ignore_eos": args.ignore_eos,
            "return_token_ids": True,
        }
        endpoint_path = "completions"
    elif args.endpoint == "chat-text":
        payload = {
            "model": args.served_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": args.gen_tokens,
            "temperature": 0.0,
            "stream": True,
            "ignore_eos": args.ignore_eos,
        }
        endpoint_path = "chat/completions"
    else:
        if args.image_path is None:
            raise SystemExit("--image-path is required for chat-image")
        encoded = base64.b64encode(args.image_path.read_bytes()).decode("ascii")
        payload = {
            "model": args.served_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{encoded}"},
                        },
                    ],
                }
            ],
            "max_tokens": args.gen_tokens,
            "temperature": 0.0,
            "stream": True,
            "ignore_eos": args.ignore_eos,
        }
        endpoint_path = "chat/completions"

    stop = threading.Event()
    monitor_thread = None
    if args.gpu_log is not None:
        monitor_thread = threading.Thread(
            target=monitor_gpu, args=(stop, args.gpu_log, args.gpu_interval), daemon=True
        )
        monitor_thread.start()

    try:
        result = stream_request(args.base_url, payload, endpoint_path, args.read_timeout)
    finally:
        stop.set()
        if monitor_thread is not None:
            monitor_thread.join(timeout=args.gpu_interval + 5)

    text = result.pop("text")
    completion_tokens = int(result.get("completion_tokens_from_ids") or 0)
    token_source = "token_ids"
    if completion_tokens <= 0:
        completion_tokens = len(tokenizer.encode(text, add_special_tokens=False))
        token_source = "text_fallback"

    ttft = result.get("ttft_s")
    decode_s = None
    decode_tok_s = None
    if isinstance(ttft, (int, float)):
        decode_s = max(float(result["elapsed_s"]) - float(ttft), 1e-9)
        decode_tok_s = completion_tokens / decode_s

    record = {
        "label": args.label,
        "endpoint": args.endpoint,
        "model": args.served_name,
        "model_dir": args.model_dir,
        "requested_prompt_tokens": args.prompt_tokens,
        "prompt_tokens": prompt_tokens,
        "requested_completion_tokens": args.gen_tokens,
        "completion_tokens": completion_tokens,
        "completion_token_source": token_source,
        "ignore_eos": args.ignore_eos,
        "pure_filler": args.pure_filler,
        "prepare_s": prepare_s,
        **result,
        "text_chars": len(text),
        "content_sample": text[:300],
        "prefill_tok_s": (prompt_tokens / ttft) if isinstance(ttft, (int, float)) and ttft > 0 else None,
        "decode_s": decode_s,
        "decode_tok_s": decode_tok_s,
        "gpu_log": str(args.gpu_log) if args.gpu_log else None,
        "gpu_summary": parse_gpu_log(args.gpu_log) if args.gpu_log else {},
    }
    if args.expect_image_card:
        record["image_card_correct"] = image_card_correct(text)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
