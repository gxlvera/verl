"""Per-turn empty-<think> analysis for the THINKING (main) model.

Reads main_output records and classifies each per turn:
  - think category: EMPTY (content between <think>..</think> is whitespace only)
                    vs REASONING (has real content)
  - output type: tool_call / answer / other
Also reports the think-content length (chars) distribution per turn.

Usage: python analyze_empty_think.py <trace_glob_prefix>
"""
import glob
import json
import re
import sys
from collections import defaultdict

import numpy as np

prefix = sys.argv[1]
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
TOOLCALL_RE = re.compile(r"<tool_call>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>", re.DOTALL)

per_turn = defaultdict(lambda: {
    "n": 0, "empty": 0, "reason": 0, "no_think": 0,
    "tool": 0, "answer": 0, "other": 0,
    "think_chars": [],
})

for f in glob.glob(f"{prefix}.*.jsonl"):
    for line in open(f):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("kind") != "main_output":
            continue
        t = d.get("turn")
        txt = d.get("main_text", "") or ""
        pt = per_turn[t]
        pt["n"] += 1
        m = THINK_RE.search(txt)
        if m:
            content = m.group(1).strip()
            pt["think_chars"].append(len(content))
            if content == "":
                pt["empty"] += 1
            else:
                pt["reason"] += 1
        else:
            pt["no_think"] += 1
        if TOOLCALL_RE.search(txt):
            pt["tool"] += 1
        elif ANSWER_RE.search(txt):
            pt["answer"] += 1
        else:
            pt["other"] += 1


def pct(a, b):
    return f"{(a/b*100):.1f}%" if b else "n/a"


print("=== THINKING(main) empty-<think> rate per turn (Qwen3-14B, full shadow, no latency) ===")
print(f"{'turn':>4} {'samples':>8} {'empty':>7} {'empty%':>7} {'reason':>7} {'no_think':>8} "
      f"{'think_chars p50/p90':>20}")
for t in sorted(per_turn):
    pt = per_turn[t]
    tc = pt["think_chars"]
    cs = f"{int(np.percentile(tc,50))}/{int(np.percentile(tc,90))}" if tc else "n/a"
    print(f"{t:>4} {pt['n']:>8} {pt['empty']:>7} {pct(pt['empty'],pt['n']):>7} "
          f"{pt['reason']:>7} {pt['no_think']:>8} {cs:>20}")

print("\n=== main output type per turn ===")
print(f"{'turn':>4} {'samples':>8} {'tool_call':>10} {'answer':>8} {'other':>7}")
for t in sorted(per_turn):
    pt = per_turn[t]
    print(f"{t:>4} {pt['n']:>8} {pt['tool']:>10} {pt['answer']:>8} {pt['other']:>7}")
