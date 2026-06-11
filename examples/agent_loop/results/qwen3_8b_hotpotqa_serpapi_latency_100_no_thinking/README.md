# Qwen3-8B HotpotQA SerpApi Latency Distribution

This run sampled 100 questions from `/root/data/hotpotqa_tool_agent/test.parquet`,
asked `Qwen/Qwen3-8B` to generate one `search_web(query)` tool call per question,
and executed the generated queries through SerpApi.

The final run used Qwen3 with thinking disabled (`enable_thinking=false`) so each
prompt reliably emitted a tool call. All 100 generated search calls succeeded.

## Search Latency Summary

The search API latency distribution is right-skewed with a visible long tail:

| metric | seconds |
| --- | ---: |
| n | 100 |
| mean | 2.720 |
| stdev | 3.086 |
| min | 0.082 |
| p50 | 1.526 |
| p75 | 3.565 |
| p90 | 6.509 |
| p95 | 8.689 |
| p99 | 13.438 |
| max | 18.701 |

Bucket counts:

| latency bucket (s) | count |
| --- | ---: |
| [0, 0.1) | 12 |
| [0.5, 1) | 14 |
| [1, 2) | 34 |
| [2, 3) | 10 |
| [3, 5) | 15 |
| [5, 8) | 9 |
| [8, 10) | 2 |
| [10, 15) | 3 |
| [15, 20) | 1 |

For modeling, the mean alone understates tail latency. Prefer using the median
plus upper percentiles: p50 is about 1.5s, p90 is about 6.5s, and p95 is about
8.7s.

## Files

- `summary.json`: aggregate model and search latency summary.
- `search_latencies.csv`: one row per question/search call with query and elapsed time.
- `../../qwen3_hotpotqa_serpapi_latency.py`: script used to generate tool calls and time SerpApi.
