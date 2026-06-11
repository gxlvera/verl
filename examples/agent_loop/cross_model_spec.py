"""Cross-model speculation: use a small drafter model (e.g. Qwen3-4B) to GUESS the
tool calls that a larger target model (Qwen3-8B / Qwen3-14B) actually made, on the
HotpotQA multi-turn agent traces.

Method (fully offline replay):
  - For each (sample, turn) in a target run's trace dump we have:
      * shadow_prompt_text : the EXACT non-thinking prompt (= full accumulated context
        the target saw + empty-<think> suffix) that was fed to the same-model shadow.
      * main_text          : the target (thinking) model's real output for that turn
        -> ground-truth tool call.
  - We feed shadow_prompt_text to the DRAFTER (4B, non-thinking, greedy) and parse its
    tool call, then compare drafter-vs-target per turn:
      name match, name+args match, query match.

This isolates "can a different/smaller model guess the big model's tool call" using the
same prompts, so the only thing that changes vs the in-run shadow is the drafter weights.

Usage:
  python cross_model_spec.py --drafter /home/tiger/models/Qwen3-4B \
      --target 8B=logs/trace_dir_direct_nocompress_bs256_20260609_165847 \
      --target 14B=logs/trace_d14_full_await_direct_bs128_20260610_031247 \
      --out logs/cross_spec_4b
"""
import argparse
import glob
import json
import re
from collections import defaultdict

import numpy as np

TOOLCALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def parse_call(text):
    m = TOOLCALL_RE.search(text or "")
    if not m:
        return None
    try:
        obj = json.loads(m.group(1))
    except Exception:
        return None
    name = obj.get("name")
    args = obj.get("arguments", {})
    try:
        args_norm = json.dumps(args, sort_keys=True, ensure_ascii=False).strip().lower()
    except Exception:
        args_norm = str(args).strip().lower()
    query = None
    if isinstance(args, dict):
        query = (args.get("query") or "").strip().lower()
    return (name, args_norm, query)


def load_trace(prefix):
    """Return dict[(rid,turn)] = {'prompt':..., 'main':...}."""
    rec = defaultdict(dict)
    for f in glob.glob(f"{prefix}.*.jsonl"):
        for line in open(f):
            try:
                d = json.loads(line)
            except Exception:
                continue
            k = d.get("kind")
            key = (d.get("request_id"), int(d.get("turn", -1)))
            if k == "shadow_prompt":
                rec[key]["prompt"] = d.get("shadow_prompt_text", "")
            elif k == "main_output":
                rec[key]["main"] = d.get("main_text", "")
    # keep only entries with both prompt and main output
    return {k: v for k, v in rec.items() if "prompt" in v and "main" in v}


def pct(a, b):
    return f"{(a / b * 100):.1f}%" if b else "  n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafter", required=True, help="path to drafter model (Qwen3-4B)")
    ap.add_argument("--target", action="append", required=True,
                    help="label=trace_prefix (can repeat)")
    ap.add_argument("--out", default="logs/cross_spec")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--mem_fraction", type=float, default=0.85)
    ap.add_argument("--limit", type=int, default=0, help="cap #prompts per target (0=all)")
    args = ap.parse_args()

    targets = []
    for t in args.target:
        label, prefix = t.split("=", 1)
        data = load_trace(prefix)
        targets.append((label, prefix, data))
        print(f"[load] {label}: {len(data)} (sample,turn) pairs from {prefix}")

    from sglang import Engine
    print(f"[engine] loading drafter {args.drafter} ...")
    engine = Engine(model_path=args.drafter, tp_size=1,
                    mem_fraction_static=args.mem_fraction,
                    context_length=8192, log_level="warning")
    sp = {"temperature": 0.0, "top_p": 1.0, "max_new_tokens": args.max_new_tokens,
          "stop": ["<|im_end|>"]}

    summary_lines = []
    for label, prefix, data in targets:
        keys = sorted(data.keys(), key=lambda x: (x[1], x[0]))
        if args.limit:
            keys = keys[: args.limit]
        prompts = [data[k]["prompt"] for k in keys]
        print(f"[gen] {label}: drafting {len(prompts)} prompts with 4B ...")
        outs = engine.generate(prompts, sp)
        draft_text = [o["text"] for o in outs]

        # dump raw per-sample results
        with open(f"{args.out}_{label}_raw.jsonl", "w") as fo:
            for k, dt in zip(keys, draft_text):
                fo.write(json.dumps({
                    "request_id": k[0], "turn": k[1],
                    "drafter_text": dt, "target_text": data[k]["main"],
                }, ensure_ascii=False) + "\n")

        per = defaultdict(lambda: {
            "reach": 0, "tgt_call": 0, "drf_call": 0, "both": 0,
            "name": 0, "full": 0, "query": 0,
        })
        for k, dt in zip(keys, draft_text):
            turn = k[1]
            p = per[turn]
            p["reach"] += 1
            tc = parse_call(data[k]["main"])   # target ground truth
            dc = parse_call(dt)                # drafter guess
            if tc:
                p["tgt_call"] += 1
            if dc:
                p["drf_call"] += 1
            if tc and dc:
                p["both"] += 1
                if tc[0] == dc[0]:
                    p["name"] += 1
                if tc[0] == dc[0] and tc[1] == dc[1]:
                    p["full"] += 1
                if tc[2] is not None and tc[2] == dc[2]:
                    p["query"] += 1

        hdr = (f"\n===== drafter=4B  target={label} =====\n"
               f"{'turn':>4} {'reach':>6} {'tgt_call':>8} {'drf_call':>8} {'both':>6} "
               f"{'name=':>7} {'name+args=':>10} {'query=':>7} "
               f"{'match/tgt_call':>13} {'match/reach':>11}")
        print(hdr)
        summary_lines.append(hdr)
        agg = defaultdict(int)
        for t in sorted(per):
            p = per[t]
            for kk in p:
                agg[kk] += p[kk]
            line = (f"{t:>4} {p['reach']:>6} {p['tgt_call']:>8} {p['drf_call']:>8} "
                    f"{p['both']:>6} {pct(p['name'],p['both']):>7} "
                    f"{pct(p['full'],p['both']):>10} {pct(p['query'],p['both']):>7} "
                    f"{pct(p['full'],p['tgt_call']):>13} {pct(p['full'],p['reach']):>11}")
            print(line)
            summary_lines.append(line)
        # tail >=3
        b3 = sum(per[t]['both'] for t in per if t >= 3)
        f3 = sum(per[t]['full'] for t in per if t >= 3)
        tg3 = sum(per[t]['tgt_call'] for t in per if t >= 3)
        rc3 = sum(per[t]['reach'] for t in per if t >= 3)
        tail = (f">=3 full(name+args): match/both={pct(f3,b3)} (n={b3})  "
                f"match/tgt_call={pct(f3,tg3)}  match/reach={pct(f3,rc3)}")
        print(tail)
        summary_lines.append(tail)

    with open(f"{args.out}_summary.txt", "w") as fo:
        fo.write("\n".join(summary_lines) + "\n")
    print(f"\n[done] summary -> {args.out}_summary.txt")
    engine.shutdown()


if __name__ == "__main__":
    main()
