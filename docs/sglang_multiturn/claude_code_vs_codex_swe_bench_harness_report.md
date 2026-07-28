# Claude Code vs Codex CLI: SWE-bench Harness Report

## Scope

This report analyzes Qwen3.5-4B on SWE-bench Verified through two CLI harnesses:

- **Claude Code full run:** `formal/20260725_144928`, 500 prompts launched and 499 scored.
- **Matched comparison:** indices 0–7 from the Claude Code full run versus Codex
  `smoke8/20260726_092705`.
- **Responses API fix validation:** Codex `responses-fix-smoke8/20260728_121431`.

The 499-task Claude Code statistics and the matched 8-task comparison are intentionally
reported separately. Codex does not yet have a comparable 499-task run.

## Metric definitions

- **Turns:** `num_turns` from the selected trajectory.
- **Total length:** `prompt_len + response_len`.
- **Response length:** the trajectory response buffer, including model output and tool
  responses.
- **Model tokens:** tokens retained by `response_mask` as model-generated training tokens.
- **Backend generated tokens:** Gateway `cumulative_completion_tokens`, which measures
  actual model generation and excludes tool output.

For Claude Code sessions with auxiliary trajectories, the selected trajectory follows the
training selection rule: maximize model-token count, then response length, then turns.

## Claude Code full run: 499 scored tasks

The full run resolved 215 of 499 scored tasks: **43.09%**.

| Cohort | n | Mean turns | Mean total length | Total length sum | Mean response length | Response sum | Mean model tokens | Model-token sum | Mean backend generated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All | 499 | 103.6 | 108,514 | 54,148,475 | 92,476 | 46,145,424 | 25,950 | 12,949,050 | 72,734 |
| Reward = 1 | 215 | 99.2 | 102,806 | 22,103,183 | 87,120 | 18,730,753 | 23,760 | 5,108,413 | 68,766 |
| Reward = 0 | 284 | 106.9 | 112,836 | 32,045,292 | 96,531 | 27,414,671 | 27,608 | 7,840,637 | 75,737 |

Compared with reward=0, reward=1 tasks used 7.2% fewer turns, had 8.9% shorter
trajectories, 9.8% shorter responses, and 13.9% fewer retained model tokens.

## Matched indices 0–7

### Claude Code

| Cohort | n | Mean turns | Mean total length | Total length sum | Mean response length | Response sum | Mean model tokens | Model-token sum | Mean backend generated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All | 8 | 100.5 | 119,770 | 958,157 | 99,863 | 798,900 | 29,114 | 232,910 | 75,848 |
| Reward = 1 | 2 | 95.5 | 107,550 | 215,099 | 93,614 | 187,228 | 26,820 | 53,640 | 83,054 |
| Reward = 0 | 6 | 102.2 | 123,843 | 743,058 | 101,945 | 611,672 | 29,878 | 179,270 | 73,447 |

### Codex baseline

| Cohort | n | Mean turns | Mean total length | Total length sum | Mean response length | Response sum | Mean model tokens | Model-token sum | Mean backend generated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All | 8 | 180.8 | 76,504 | 612,034 | 67,467 | 539,732 | 1,458 | 11,664 | 36,510 |
| Reward = 1 | 1 | 89.0 | 69,371 | 69,371 | 60,989 | 60,989 | 11,168 | 11,168 | 34,771 |
| Reward = 0 | 7 | 193.9 | 77,523 | 542,663 | 68,392 | 478,743 | 70.9 | 496 | 36,758 |

The baseline Codex reward=0 model-token count was invalid. Those seven tasks averaged
68,392 response tokens and 36,758 actual backend-generated tokens, but only 70.9 retained
model tokens. The Responses adapter returned function calls without the assistant
text/thinking emitted in the same turn. On the next request, Gateway interpreted the
missing text as an assistant rewrite and rolled back previously trainable tokens.

### Per-task matched data

| Index and task | CC reward | CC turns | CC total | CC response | CC model | Codex reward | Codex turns | Codex total | Codex response | Codex model |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 · astropy-12907 | 1 | 90 | 78,232 | 65,308 | 29,847 | 1 | 89 | 69,371 | 60,989 | 11,168 |
| 1 · astropy-13033 | 0 | 113 | 97,086 | 83,744 | 30,074 | 0 | 114 | 78,393 | 69,612 | 432 |
| 2 · astropy-13236 | 0 | 69 | 135,883 | 122,808 | 14,325 | 0 | 64 | 33,563 | 25,032 | 17 |
| 3 · astropy-13398 | 0 | 111 | 127,562 | 113,717 | 32,296 | 0 | 483 | 89,865 | 80,548 | 1 |
| 4 · astropy-13453 | 1 | 101 | 136,867 | 121,920 | 23,793 | 0 | 217 | 63,063 | 52,631 | 38 |
| 5 · astropy-13579 | 0 | 87 | 121,175 | 107,046 | 36,899 | 0 | 172 | 140,702 | 131,121 | 0 |
| 6 · astropy-13977 | 0 | 118 | 130,018 | 65,887 | 22,286 | 0 | 61 | 30,025 | 21,049 | 6 |
| 7 · astropy-14096 | 0 | 115 | 131,334 | 118,470 | 43,390 | 0 | 246 | 107,052 | 98,750 | 2 |

On these matched tasks, Codex used 1.80x as many turns. Its selected trajectories were
36.1% shorter, its response lengths were 32.4% shorter, and its actual backend generation
was 51.9% lower. The turn average is sensitive to the 483-turn Codex long tail.

## Claude Code formal-run configuration

The original Ray submission omitted `--limit`, so it selected all 500 dataset rows.

| Setting | Effective value | Scope and evidence |
|---|---:|---|
| Engine | SGLang | Original Ray submission |
| GPUs | 8 x A100 | One node, eight GPUs |
| Tensor parallelism | 1 | Eight single-GPU SGLang replicas |
| Session concurrency | 64 | `--concurrency 64` |
| Rollouts per task | 1 | `--n 1` |
| Gateway actors | 1 | `--gateway-count 1` |
| Rollout max prompt length | 4,096 tokens | Initial dataset prompt limit |
| Rollout max response length | 131,072 tokens | Cumulative trajectory response buffer |
| SGLang model context | 262,144 tokens | Model config default; not overridden |
| Agent max turns | 200 | Claude Code task config |
| Agent timeout | 4,800 seconds | Claude Code task config |
| Sandbox timeout | 7,200 seconds | SWE-bench task config |
| Temperature / top-p | 1.0 / 1.0 | Task model config |
| Whole-episode response budget | 131,072 tokens | `max_total_tokens` |
| Recipe per-turn cap | `null` | `max_tokens_per_turn` in every task log |

The 4,096-token prompt setting limits the initial rollout data prompt. Claude Code adds
its own system prompt, tool schemas, and conversation history afterward, so an actual
model request can exceed 4,096 input tokens.

The 131,072-token response setting is cumulative across the trajectory and includes model
output plus tool responses. It is not a per-request generation allowance.

### Per-request generation limit

Claude Code itself adds `max_tokens=32,000` to each Anthropic model request. This
per-turn field comes from Claude Code, not from the uni-agent recipe or Gateway. A live
SWE-bench capture observed two identical Claude Code warm-up requests, and both carried:

```json
{
  "messages": [{"role": "user", "content": [{"type": "text", "text": "Warmup"}]}],
  "max_tokens": 32000,
  "stream": true
}
```

The recipe still had `max_tokens_per_turn=null`: it did not add a second independent
per-turn cap. Gateway honors Claude Code's request field and sends SGLang the minimum of:

1. Claude Code's request `max_tokens` (32,000 in the captured current CLI),
2. the remaining trajectory response budget, and
3. the output space left in the model context.

The historical 499-task run did not persist raw request payloads, so the captured 32,000
value is strong current-path evidence rather than direct proof of the exact historical
CLI value. The largest completion observed in the 499-task historical Gateway logs was
20,006 tokens.

### Qwen3.5-4B context length

The checkpoint has:

```text
text_config.max_position_embeddings = 262144
```

Therefore, its configured maximum context length is **262,144 tokens (256 Ki tokens)**.
The historical SGLang logs independently confirm this value by rejecting requests over
262,144 tokens.

## Responses API fix validation

The patched Responses adapter emits both `output_text` and `function_call` output items
when the model produces text/thinking and a tool call together. On input, those adjacent
items are reconstructed as one assistant message.

The live validation run used the new Sandbox PSM and eight concurrent Codex SWE-bench
tasks:

- 8/8 sessions scored; mean reward 0.125.
- Wall time: 3,324 seconds.
- 0 assistant-history rollbacks across all sessions.
- 2,274 mixed text-plus-tool-call turns exercised the fixed path.
- Trajectory model tokens: 457,677.
- Gateway backend completion tokens: 457,677.
- Reward=0 mean model tokens increased from the invalid baseline 70.9 to 59,462.6.
- No Gateway, Codex, or Sandbox errors were found.

### Patched-run cohort summary

| Cohort | n | Mean turns | Mean total length | Total length sum | Mean response length | Response sum | Mean model tokens | Model-token sum | Mean backend generated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All | 8 | 288.1 | 106,290 | 850,322 | 97,004 | 776,028 | 57,210 | 457,677 | 57,210 |
| Reward = 1 | 1 | 66.0 | 72,797 | 72,797 | 64,166 | 64,166 | 41,439 | 41,439 | 41,439 |
| Reward = 0 | 7 | 319.9 | 111,075 | 777,525 | 101,695 | 711,862 | 59,463 | 416,238 | 59,463 |

### Patched-run per-trajectory data

| Index and task | Reward | Turns | Prompt length | Response length | Total length | Model tokens | Backend tokens | Text + tool turns | Rollbacks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 · astropy-12907 | 1 | 66 | 8,631 | 64,166 | 72,797 | 41,439 | 41,439 | 62 | 0 |
| 1 · astropy-13033 | 0 | 99 | 9,030 | 75,629 | 84,659 | 47,517 | 47,517 | 95 | 0 |
| 2 · astropy-13236 | 0 | 268 | 8,780 | 91,149 | 99,929 | 39,098 | 39,098 | 264 | 0 |
| 3 · astropy-13398 | 0 | 182 | 9,566 | 131,072 | 140,638 | 74,479 | 74,479 | 178 | 0 |
| 4 · astropy-13453 | 0 | 373 | 10,681 | 93,030 | 103,711 | 46,203 | 46,203 | 369 | 0 |
| 5 · astropy-13579 | 0 | 557 | 9,830 | 131,072 | 140,902 | 91,963 | 91,963 | 553 | 0 |
| 6 · astropy-13977 | 0 | 75 | 9,225 | 58,838 | 68,063 | 22,567 | 22,567 | 71 | 0 |
| 7 · astropy-14096 | 0 | 685 | 8,551 | 131,072 | 139,623 | 94,411 | 94,411 | 682 | 0 |

Indices 3, 5, and 7 reached the exact 131,072-token cumulative response limit. Their
model-token counts remained lower because tool responses consume response-buffer space
but are correctly excluded from the model-token mask.

The exact equality between retained model tokens and backend completion tokens, together
with zero rollbacks, confirms that the Responses round-trip now preserves assistant
thinking content and produces a valid training mask.

## Limitations

- Claude Code full-run metrics cover 499 scored tasks; one of 500 launched tasks did not
  produce a scored trajectory.
- Direct Claude Code versus Codex comparisons cover only matched indices 0–7.
- The historical Claude Code request payload did not record its raw per-request
  `max_tokens`; the 32,000 value comes from a current equivalent-path capture.
