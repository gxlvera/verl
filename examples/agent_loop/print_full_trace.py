"""Reconstruct and print a complete per-sample multi-turn trace from the trace dump.

Records (one JSON per line) have either:
  - prompt:        {request_id, turn, prompt_len, prompt_text}
  - main_output:   {kind:"main_output", request_id, turn, tokens, main_text}
  - shadow_output: {kind:"shadow_output", request_id, turn, tokens, shadow_text}

Usage:
  python print_full_trace.py <trace_glob_prefix> [request_id]
  # no request_id  -> list candidate request_ids (those with the most turns)
"""
import glob
import json
import sys
from collections import defaultdict

prefix = sys.argv[1]
want_rid = sys.argv[2] if len(sys.argv) > 2 else None

files = glob.glob(f"{prefix}.*.jsonl")
# per (rid, turn) -> {"prompt":.., "main":(tok,txt), "shadow":(tok,txt)}
data = defaultdict(dict)
turns_by_rid = defaultdict(set)
for f in files:
    for line in open(f):
        try:
            d = json.loads(line)
        except Exception:
            continue
        rid = d.get("request_id")
        turn = d.get("turn")
        if rid is None or turn is None:
            continue
        k = d.get("kind")
        if k == "main_output":
            data[(rid, turn)]["main"] = (d["tokens"], d["main_text"])
        elif k == "shadow_output":
            data[(rid, turn)]["shadow"] = (d["tokens"], d["shadow_text"])
        else:  # prompt record (no kind)
            if "prompt_text" in d:
                data[(rid, turn)]["prompt"] = (d.get("prompt_len"), d["prompt_text"])
        turns_by_rid[rid].add(turn)

if not want_rid:
    # list rids with most turns AND that have all three kinds at turn 1
    ranked = sorted(turns_by_rid.items(), key=lambda kv: -len(kv[1]))
    print("top request_ids by #turns (turn set):")
    for rid, ts in ranked[:25]:
        has_t1 = (rid, 1) in data and {"prompt", "main", "shadow"} <= set(data[(rid, 1)].keys())
        print(f"  {rid}  turns={sorted(ts)}  full_turn1={has_t1}")
    sys.exit(0)

rid = want_rid
turns = sorted(turns_by_rid.get(rid, []))
print("#" * 100)
print(f"# FULL TRACE  request_id={rid}  turns={turns}")
print("#" * 100)
for t in turns:
    cell = data[(rid, t)]
    print("\n" + "=" * 100)
    print(f"================================  TURN {t}  ================================")
    print("=" * 100)
    if "prompt" in cell:
        plen, ptext = cell["prompt"]
        print(f"\n----- PROMPT fed to LLM (turn {t}, prompt_len={plen}) -----\n")
        print(ptext)
    if "main" in cell:
        tok, txt = cell["main"]
        print(f"\n----- THINKING (main) OUTPUT (turn {t}, tokens={tok}) -----\n")
        print(txt)
    if "shadow" in cell:
        tok, txt = cell["shadow"]
        print(f"\n----- NON-THINKING (shadow) OUTPUT (turn {t}, tokens={tok}) -----\n")
        print(txt)
