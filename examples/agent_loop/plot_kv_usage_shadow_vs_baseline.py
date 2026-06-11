#!/usr/bin/env python3
"""Overlay KV usage (SGLang token_usage) for full-shadow vs no-shadow bs256.

token_usage is a per-server fraction in [0,1]; we average it across the 8 servers
at each sample tick. Also overlays num_used_tokens (summed across servers).
"""
import glob
import json
import re
import statistics
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = "/home/tiger/verl/logs"
RUNS = {
    "no shadow": "think_8192_500pt_4tool_20260608_111923",
    "full shadow (every turn)": "think_8192_500pt_shadow_20260608_132248",
}
COLORS = {"no shadow": "tab:blue", "full shadow (every turn)": "tab:red"}


def strip(k):
    return re.sub(r"\{.*?\}", "", k)


def load(tag):
    fp = glob.glob(f"{LOG}/baseline_*{tag}_bs256*_sglang_metrics.jsonl")[0]
    # bucket per timestamp: average token_usage across servers, sum used tokens
    tu = defaultdict(list)
    used = defaultdict(list)
    runn = defaultdict(list)
    with open(fp, errors="ignore") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            if "series" not in d:
                continue
            t = round(d["monotonic_s"], 1)
            for k, v in d["series"].items():
                kk = strip(k)
                if kk == "sglang:token_usage":
                    tu[t].append(v or 0.0)
                elif kk == "sglang:num_used_tokens":
                    used[t].append(v or 0.0)
                elif kk == "sglang:num_running_reqs":
                    runn[t].append(v or 0.0)
    ts = sorted(tu)
    tu_avg = [statistics.mean(tu[t]) for t in ts]
    used_sum = [sum(used[t]) for t in ts]
    run_sum = [sum(runn[t]) for t in ts]
    # active window: running >= 1
    idx = [i for i, r in enumerate(run_sum) if r >= 1]
    i0, i1 = idx[0], idx[-1]
    ts = ts[i0 : i1 + 1]
    t0 = ts[0]
    return [t - t0 for t in ts], tu_avg[i0 : i1 + 1], used_sum[i0 : i1 + 1]


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
print(f"{'run':30s} {'peak KV util':>14s} {'mean KV util':>14s} {'peak used tok':>16s}")
for label, tag in RUNS.items():
    T, tu, used = load(tag)
    ax1.plot(T, tu, color=COLORS[label], label=label)
    ax2.plot(T, used, color=COLORS[label], label=label)
    print(f"{label:30s} {max(tu):14.3f} {statistics.mean(tu):14.3f} {max(used):16.0f}")

ax1.axhline(1.0, color="gray", ls=":", alpha=0.6)
ax1.set_ylabel("token_usage (KV util, avg/server)")
ax1.set_title("bs256, max_len=8192, per_turn=500, thinking ON: KV usage, full-shadow vs no-shadow")
ax1.legend(loc="lower center")
ax1.grid(alpha=0.3)
ax2.set_ylabel("num_used_tokens (sum over servers)")
ax2.set_xlabel("time since first sample (s)")
ax2.legend(loc="upper right")
ax2.grid(alpha=0.3)
fig.tight_layout()
out = f"{LOG}/kv_usage_shadow_vs_baseline_bs256.png"
fig.savefig(out, dpi=120)
print("saved:", out)
