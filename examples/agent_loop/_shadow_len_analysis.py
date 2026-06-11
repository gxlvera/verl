import glob
import json
import sys
from collections import defaultdict

import numpy as np

tag = sys.argv[1]  # e.g. real_full_await_bs256_20260608_200909
files = glob.glob(f"/home/tiger/verl/logs/sj_{tag}*.jsonl")

# per (kind, turn) -> list of token counts ; kind in {main, shadow}
buckets = defaultdict(list)
for f in files:
    with open(f) as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except Exception:
                continue
            ev = d.get("event")
            if ev not in ("shadow_decode", "main_decode"):
                continue
            if "tokens" not in d:
                continue
            kind = "main" if ev == "main_decode" else "shadow"
            buckets[(kind, d.get("turn"))].append(d["tokens"])

turns = sorted({t for (_, t) in buckets})
print(f"=== {tag} ===  files={len(files)}")
print(f"{'turn':>4} | {'main_n':>7} {'main_mean':>9} | {'shadow_n':>8} {'shadow_mean':>11} | {'shadow/main':>11}")
for t in turns:
    m = np.array(buckets[("main", t)], dtype=float)
    s = np.array(buckets[("shadow", t)], dtype=float)
    mm = m.mean() if len(m) else float("nan")
    sm = s.mean() if len(s) else float("nan")
    ratio = (sm / mm) if (len(m) and mm > 0) else float("nan")
    print(f"{t:>4} | {len(m):>7} {mm:>9.1f} | {len(s):>8} {sm:>11.1f} | {ratio:>11.2f}")

# overall
allm = np.array([x for (k, _), v in buckets.items() if k == "main" for x in v], dtype=float)
alls = np.array([x for (k, _), v in buckets.items() if k == "shadow" for x in v], dtype=float)
if len(allm) and len(alls):
    print(f" all | {len(allm):>7} {allm.mean():>9.1f} | {len(alls):>8} {alls.mean():>11.1f} | {alls.mean()/allm.mean():>11.2f}")
