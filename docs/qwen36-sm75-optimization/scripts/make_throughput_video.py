#!/usr/bin/env python3
"""Render a side-by-side video from recorded SSE token arrival timelines.

Every progress update comes from the timestamp captured by
``record_stream_timeline.py``; the renderer does not interpolate model
throughput or synthesize token events.
"""

from __future__ import annotations

import argparse
import bisect
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    name = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def load(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    events = data.get("events") or []
    data["event_times"] = [float(x["t_s"]) for x in events]
    data["event_tokens"] = [int(x["tokens"]) for x in events]
    return data


def progress(data: dict, now: float) -> tuple[int, float]:
    times = data["event_times"]
    if not times or now < times[0]:
        return 0, 0.0
    i = min(bisect.bisect_right(times, now) - 1, len(times) - 1)
    tokens = data["event_tokens"][i]
    decode_time = max(now - float(data.get("ttft_s") or 0.0), 1e-9)
    return tokens, tokens / decode_time


def draw_lane(draw: ImageDraw.ImageDraw, data: dict, x: int, y: int, w: int, h: int, now: float, title: str, colour: tuple[int, int, int]) -> None:
    draw.rounded_rectangle((x, y, x + w, y + h), radius=18, fill=(25, 31, 43), outline=(66, 78, 98), width=2)
    draw.text((x + 28, y + 22), title, fill=(240, 244, 250), font=font(28, True))
    draw.text((x + 28, y + 66), "same 4096-token prompt • stream=True • max_tokens=1024", fill=(159, 173, 193), font=font(16))

    tokens, live_tps = progress(data, now)
    done = int(data.get("completion_tokens") or 0)
    ttft = float(data.get("ttft_s") or 0.0)
    elapsed = float(data.get("elapsed_s") or 0.0)
    visible_elapsed = min(max(now, 0.0), elapsed)
    if tokens >= done and done:
        live_tps = float(data.get("decode_tok_s") or live_tps)
    draw.text((x + 28, y + 118), f"tokens  {tokens:4d} / {done:4d}", fill=(230, 235, 244), font=font(24, True))
    draw.text((x + 28, y + 155), f"elapsed  {visible_elapsed:6.2f} s", fill=(185, 198, 218), font=font(20))
    draw.text((x + 28, y + 188), f"decode   {live_tps:6.2f} tok/s", fill=colour, font=font(25, True))
    draw.text((x + 28, y + 228), f"TTFT     {ttft:6.2f} s", fill=(185, 198, 218), font=font(20))

    bx, by, bw, bh = x + 28, y + h - 82, w - 56, 27
    draw.rounded_rectangle((bx, by, bx + bw, by + bh), radius=12, fill=(50, 59, 76))
    ratio = 0.0 if not done else min(tokens / done, 1.0)
    if ratio > 0:
        draw.rounded_rectangle((bx, by, bx + max(10, int(bw * ratio)), by + bh), radius=12, fill=colour)
    draw.text((bx, by + 39), "real SSE token arrival timeline", fill=(137, 151, 173), font=font(15))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--optimized", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    baseline = load(args.baseline)
    optimized = load(args.optimized)
    duration = max(float(baseline.get("elapsed_s") or 0.0), float(optimized.get("elapsed_s") or 0.0)) + 2.0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = [
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24", "-s", f"{args.width}x{args.height}", "-r", str(args.fps), "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(args.output),
    ]
    proc = subprocess.Popen(ffmpeg, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        total = int(duration * args.fps)
        lane_w = (args.width - 90) // 2
        for frame_no in range(total + 1):
            now = frame_no / args.fps
            image = Image.new("RGB", (args.width, args.height), (12, 16, 24))
            draw = ImageDraw.Draw(image)
            draw.text((45, 24), "RTX 2080 Ti SM75 • Qwen3.6-27B AWQ • before / after", fill=(248, 250, 255), font=font(30, True))
            draw.text((45, 68), f"real long-output comparison  |  t = {now:05.1f}s  |  no synthetic token events", fill=(164, 178, 199), font=font(17))
            draw_lane(draw, baseline, 30, 115, lane_w, 390, now, "BASELINE  •  no MTP", (240, 133, 92))
            draw_lane(draw, optimized, 60 + lane_w, 115, lane_w, 390, now, "OPTIMIZED  •  MTP3 + FlashQLA + FI", (76, 218, 157))
            b = float(baseline.get("decode_tok_s") or 0.0)
            o = float(optimized.get("decode_tok_s") or 0.0)
            draw.text((45, 548), f"STRICT LONG 1024 TOKENS:  {b:.1f} tok/s  →  {o:.1f} tok/s  ({o / b:.2f}×)", fill=(255, 231, 151), font=font(24, True))
            draw.text((45, 594), "Short 128-token peak (separate run): 38.8 → 74.7 tok/s; do not read it as long-run throughput.", fill=(203, 211, 225), font=font(17))
            draw.text((45, 631), "CUDA Graph stays enabled • GPUs 6,7 • TP=2 • image limit=4 • custom all-reduce disabled for SM75 stability", fill=(137, 151, 173), font=font(15))
            proc.stdin.write(image.tobytes())
    finally:
        proc.stdin.close()
    rc = proc.wait()
    if rc:
        raise SystemExit(rc)
    print(json.dumps({"output": str(args.output), "duration_s": duration, "fps": args.fps}, ensure_ascii=False))


if __name__ == "__main__":
    main()
