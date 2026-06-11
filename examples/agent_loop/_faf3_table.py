import glob, re, sys
ts = open("/tmp/faf3_ts.txt").read().strip()
TAGS = {
    "baseline (no shadow)": "think_8192_500pt_serpapilat_20260608_144554",
    "full shadow (every turn)": f"faf_full_{ts}",
    "tail-25% shadow": f"faf_tail25_{ts}",
    "tail-15% shadow": f"faf_tail15_{ts}",
}
def g(t, p):
    m = re.search(p, t); return float(m.group(1)) if m else float("nan")
base = None
rows = []
for k, tag in TAGS.items():
    cands = [p for p in glob.glob(f"/home/tiger/verl/logs/baseline_*{tag}_bs256*.log")
             if not any(x in p for x in ("plot_active", "sampler", "metrics"))]
    t = open(cands[0], errors="ignore").read()
    tp = g(t, r"perf/throughput:([0-9.]+)")
    wall = g(t, r"state/total_wall_time:([0-9.]+)")
    lat = g(t, r"tool_latency_s/mean:([0-9.]+)")
    if base is None: base = tp
    rows.append((k, tp, wall, lat, (tp / base - 1) * 100))
print(f"{'run':28s} {'tp(tok/s)':>10s} {'wall(s)':>8s} {'toollat':>8s} {'vs base':>8s}")
for k, tp, wall, lat, d in rows:
    print(f"{k:28s} {tp:10.0f} {wall:8.1f} {lat:8.2f} {d:+7.1f}%")
