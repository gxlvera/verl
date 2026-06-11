"""Print a full per-sample trace for the COMPRESSED-shadow run, interleaving for
each turn: the MAIN (thinking) prompt, the SHADOW (non-thinking, compressed)
prompt, the MAIN output, and the SHADOW output. Used to verify the shadow prompt
has the prior <think> traces stripped.

Records (one JSON/line):
  - main prompt:   {request_id, turn, prompt_len, prompt_text}            (no kind)
  - shadow prompt: {kind:"shadow_prompt", request_id, turn, prompt_len, shadow_prompt_text}
  - main_output:   {kind:"main_output", request_id, turn, tokens, main_text}
  - shadow_output: {kind:"shadow_output", request_id, turn, tokens, shadow_text}

Usage: python print_compress_trace.py <trace_glob_prefix> <request_id>
"""
import glob
import json
import sys
from collections import defaultdict

prefix = sys.argv[1]
rid = sys.argv[2]

data = defaultdict(dict)
turns = set()
for f in glob.glob(f"{prefix}.*.jsonl"):
    for line in open(f):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("request_id") != rid:
            continue
        t = d.get("turn")
        if t is None:
            continue
        k = d.get("kind")
        if k == "main_output":
            data[t]["main"] = (d["tokens"], d["main_text"])
        elif k == "shadow_output":
            data[t]["shadow"] = (d["tokens"], d["shadow_text"])
        elif k == "shadow_prompt":
            data[t]["sprompt"] = (d["prompt_len"], d["shadow_prompt_text"])
        else:
            if "prompt_text" in d:
                data[t]["mprompt"] = (d.get("prompt_len"), d["prompt_text"])
        turns.add(t)

print("#" * 100)
print(f"# COMPRESSED-SHADOW FULL TRACE  request_id={rid}  turns={sorted(turns)}")
print("#" * 100)
for t in sorted(turns):
    c = data[t]
    print("\n" + "=" * 100)
    print(f"================================  TURN {t}  ================================")
    print("=" * 100)
    if "mprompt" in c:
        pl, txt = c["mprompt"]
        print(f"\n----- MAIN (thinking) PROMPT (turn {t}, len={pl}) -----\n")
        print(txt)
    if "sprompt" in c:
        pl, txt = c["sprompt"]
        print(f"\n----- SHADOW (non-thinking, COMPRESSED) PROMPT (turn {t}, len={pl}) -----\n")
        print(txt)
    if "main" in c:
        tok, txt = c["main"]
        print(f"\n----- MAIN (thinking) OUTPUT (turn {t}, tokens={tok}) -----\n")
        print(txt)
    if "shadow" in c:
        tok, txt = c["shadow"]
        print(f"\n----- SHADOW (non-thinking) OUTPUT (turn {t}, tokens={tok}) -----\n")
        print(txt)
