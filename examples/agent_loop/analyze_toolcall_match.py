"""Per-turn analysis of thinking(main) vs non-thinking(shadow) tool calls.

Reads the trace dump (main_output/shadow_output records) for a run, parses the
<tool_call>{...}</tool_call> JSON from each side, and reports per turn:
  - decode length (tokens) stats for thinking vs shadow
  - fraction of pairs where BOTH emit a tool call
  - name-match rate and full (name+arguments) match rate

Usage: python analyze_toolcall_match.py <trace_glob_prefix>
"""
import glob
import json
import re
import sys
from collections import defaultdict

import numpy as np

prefix = sys.argv[1]

TOOLCALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def parse_call(text):
    """Return (name, normalized_args_str) of the FIRST tool call, or None."""
    m = TOOLCALL_RE.search(text or "")
    if not m:
        return None
    try:
        obj = json.loads(m.group(1))
    except Exception:
        return None
    name = obj.get("name")
    args = obj.get("arguments", {})
    # normalize: only compare the 'query' arg (the meaningful retrieval key),
    # but keep full args for a stricter view too.
    try:
        args_norm = json.dumps(args, sort_keys=True, ensure_ascii=False).strip().lower()
    except Exception:
        args_norm = str(args).strip().lower()
    query = None
    if isinstance(args, dict):
        query = (args.get("query") or "").strip().lower()
    return (name, args_norm, query)


main_txt = {}
shadow_txt = {}
main_tok = {}
shadow_tok = {}
for f in glob.glob(f"{prefix}.*.jsonl"):
    for line in open(f):
        try:
            d = json.loads(line)
        except Exception:
            continue
        k = d.get("kind")
        key = (d.get("request_id"), d.get("turn"))
        if k == "main_output":
            main_txt[key] = d.get("main_text", "")
            main_tok[key] = d.get("tokens")
        elif k == "shadow_output":
            shadow_txt[key] = d.get("shadow_text", "")
            shadow_tok[key] = d.get("tokens")

keys = sorted(set(main_txt) & set(shadow_txt), key=lambda x: (x[1], x[0]))
per_turn = defaultdict(lambda: {
    "n": 0, "both_call": 0, "name_match": 0, "full_match": 0, "query_match": 0,
    "main_call": 0, "shadow_call": 0, "mtok": [], "stok": [],
})
for key in keys:
    turn = key[1]
    pt = per_turn[turn]
    pt["n"] += 1
    if main_tok.get(key) is not None:
        pt["mtok"].append(main_tok[key])
    if shadow_tok.get(key) is not None:
        pt["stok"].append(shadow_tok[key])
    mc = parse_call(main_txt[key])
    sc = parse_call(shadow_txt[key])
    if mc:
        pt["main_call"] += 1
    if sc:
        pt["shadow_call"] += 1
    if mc and sc:
        pt["both_call"] += 1
        if mc[0] == sc[0]:
            pt["name_match"] += 1
        if mc[0] == sc[0] and mc[1] == sc[1]:
            pt["full_match"] += 1
        if mc[2] is not None and mc[2] == sc[2]:
            pt["query_match"] += 1


def pct(a, b):
    return f"{(a / b * 100):.1f}%" if b else "  n/a"


def msd(a):
    if not a:
        return "n/a"
    a = np.array(a, float)
    return f"mean={a.mean():.0f} p50={np.percentile(a,50):.0f} p90={np.percentile(a,90):.0f}"


print(f"{'turn':>4} {'pairs':>6} {'main_call':>9} {'shadow_call':>11} {'both':>6} "
      f"{'name=':>7} {'name+args=':>10} {'query=':>7}")
for t in sorted(per_turn):
    pt = per_turn[t]
    print(f"{t:>4} {pt['n']:>6} {pct(pt['main_call'],pt['n']):>9} {pct(pt['shadow_call'],pt['n']):>11} "
          f"{pt['both_call']:>6} {pct(pt['name_match'],pt['both_call']):>7} "
          f"{pct(pt['full_match'],pt['both_call']):>10} {pct(pt['query_match'],pt['both_call']):>7}")

print("\nper-turn decode length (tokens):")
print(f"{'turn':>4} | {'thinking(main)':>34} | {'shadow(non-think)':>34}")
for t in sorted(per_turn):
    pt = per_turn[t]
    print(f"{t:>4} | {msd(pt['mtok']):>34} | {msd(pt['stok']):>34}")

# overall (turn>=3 tail)
tot_both = sum(pt["both_call"] for pt in per_turn.values())
tot_full = sum(pt["full_match"] for pt in per_turn.values())
tail_both = sum(pt["both_call"] for t, pt in per_turn.items() if t >= 3)
tail_full = sum(pt["full_match"] for t, pt in per_turn.items() if t >= 3)
print(f"\nALL turns full(name+args) match: {pct(tot_full, tot_both)}  (n={tot_both})")
print(f"tail(turn>=3) full match:        {pct(tail_full, tail_both)}  (n={tail_both})")
