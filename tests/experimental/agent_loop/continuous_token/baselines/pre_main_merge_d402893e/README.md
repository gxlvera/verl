# Continuous Token pre-main-merge baseline

This directory contains the full CT-vs-legacy baseline captured before merging
`main` into `gxl-ct-dev-mm`.

## Provenance

- Code under test: `d402893eae67fb219742ea85977e578bddc479d2`
- Comparison harness source: `gxl-ct-dev-mm-compare` at
  `c3873635b478ed378b12cc49ec026f4172bf976b`
- Result: all text and multimodal cases completed successfully, with no
  CT-vs-legacy mismatch or execution error.

## Contents

### Text

`text/artifacts/` contains 168 JSON files: CT and legacy outputs for 84
model/trajectory cases. Each file includes:

- `token_ids` (prompt and response combined)
- `prompt_ids`
- `response_ids`
- `response_mask` / `loss_mask`
- `response_logprobs` / `logprobs`
- `generation_prompt_ids`

`text/ledger.json` contains the structured comparison result for all 84 cases.

### Multimodal

`vl/ledger_full.json` contains 17 multimodal model/trajectory cases. Its
`legacy` and `ct` objects include the same full token, mask, logprob, and
generation-prompt sequences listed above.

The original VL harness ledger retained only lengths and SHA-256 hashes. The
full sequences in this file were regenerated against the same pre-merge commit
without changing the harness execution or comparison logic. Every regenerated
sequence hashes to the value recorded by the original baseline ledger.

`vl/mismatch.jsonl` is empty because all 17 multimodal cases passed.

## Validation

- Text: 84 passed, 0 mismatches, 0 errors.
- Multimodal: 17 passed, 0 mismatches, 0 errors.
- All regenerated multimodal raw-sequence hashes match the original baseline
  ledger.
- For every case, CT and legacy `token_ids`, `loss_mask`, and `logprobs` are
  identical.

The raw JSON is kept uncompressed because the complete baseline is only about
7 MiB and direct files are easier to inspect and diff.
