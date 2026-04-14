# EXP-002 Results

## 1. Basic Info

- Experiment ID: EXP-002
- Title: Answer Agreement Baseline
- Owner: Agent 02
- Date: 2026-04-14
- Status: in_progress

---

## 2. Objective

This experiment should establish:

1. answer agreement under budget
2. unified budget metrics
3. subset-level robustness statistics

---

## 3. What Was Executed In This Iteration

### 3.1 Added/Updated Files

- Added: `scripts/eval_answer_agreement.py`
- Added: `tests/test_eval_answer_agreement.py`
- Added: `docs/experiments/EXP-002-answer-agreement-baseline/README.md`
- Added: `docs/experiments/EXP-002-answer-agreement-baseline/RESULTS.md`
- Added: `docs/experiments/EXP-002-answer-agreement-baseline/ACCEPTANCE.md`
- Added: `docs/experiments/EXP-002-answer-agreement-baseline/artifacts/sample_eval_dataset.jsonl`

### 3.2 Verification Commands

```bash
python -m unittest tests/test_eval_answer_agreement.py -v
python scripts/eval_answer_agreement.py --help
ls -lh checkpoints/releases/v2.2 data/coco
find data/coco/val2017 -maxdepth 1 -type f | wc -l
```

### 3.3 Verification Outcome

- Unit tests: pass (3/3)
- CLI argument parsing/help: pass
- Assets: `edge_weights.pth` (11MB), `cloud_weights.pth` (486MB), COCO `val2017` (5000 images) downloaded
- Compatibility links: `data/coco/train -> val2017`, `data/coco/val -> val2017`, `checkpoints/gan_bottleneck/split/*.pth -> releases/v2.2/*.pth`
- Runtime check: `splitoculo` conda env is available and local Qwen cache check returns `qwen_cache:present`

---

## 4. Metrics Implemented

### 4.1 Agreement Metric

- `agreement_rate` = normalized exact match between `prediction` and `reference`
- normalization includes lowercase + punctuation stripping + whitespace collapse

### 4.2 Budget Metrics

- `avg_payload_bytes`
- `avg_payload_bits`

### 4.3 Subset Metrics

For each subset tag:

- `num_samples`
- `agreement_rate`

---

## 5. Real Evaluation Table

### Run-003 (first successful live run)

Artifacts:

- `docs/experiments/EXP-002-answer-agreement-baseline/artifacts/run-003/summary.json`
- `docs/experiments/EXP-002-answer-agreement-baseline/artifacts/run-003/rows.json`
- `docs/experiments/EXP-002-answer-agreement-baseline/artifacts/run-003/summary.md`

Setup:

- Checkpoint: `v2.2` (`edge_weights.pth` + `cloud_weights.pth`)
- Device: Apple runtime (MPS path enabled in eval command)
- Dataset: `coco_val20_caption_proxy.jsonl` (caption-as-reference proxy), `max_samples=8`

Result table:

| budget | samples | agreement_rate | avg_payload_bytes | avg_payload_bits |
|---|---:|---:|---:|---:|
| v22 | 8 | 0.0000 | 4184.00 | 33472.00 |

Interpretation:

- The edge-cloud inference path is verified end-to-end (8/8 requests succeeded, no 500 errors).
- Exact-match agreement is 0 because COCO caption proxy is a strict semantic mismatch target for free-form generation.
- Budget metric is stable at ~4.1 KB payload (base64 bytes) for `bottleneck_dim=64`.

### Run-20260414_141417 (20-sample expanded run)

Artifacts:

- `docs/experiments/EXP-002-answer-agreement-baseline/artifacts/run-20260414_141417/summary.json`
- `docs/experiments/EXP-002-answer-agreement-baseline/artifacts/run-20260414_141417/rows.json`
- `docs/experiments/EXP-002-answer-agreement-baseline/artifacts/run-20260414_141417/summary.md`

Result table:

| budget | samples | agreement_rate | avg_payload_bytes | avg_payload_bits |
|---|---:|---:|---:|---:|
| v22 | 20 | 0.0000 | 4184.00 | 33472.00 |

Interpretation:

- 20/20 requests completed successfully (`num_failures=0`).
- Transmission size stays stable at around 4.1KB per sample.
- Exact-match answer agreement is still 0.0 under the current caption-proxy reference setup.

---

## 6. Success Cases / Failure Cases

### 6.1 Success (Engineering)

1. Evaluation script now provides stable output artifacts for agreement/budget/subset analysis.
2. Output schema is reusable by future experiments (EXP-003+).
3. Live edge-cloud batch evaluation is now running successfully with v2.2 checkpoints.
4. Expanded run from 8 samples to 20 samples completed without request failures.

### 6.2 Failure / Limitation

1. Current references are caption-proxy, not teacher logits / teacher answer targets.
2. Exact-match metric is too strict for descriptive free-form captions.
3. No OCR/digits-specific slice benchmark yet.

---

## 7. Comparison To Prior Baseline

- Before: no unified answer-agreement evaluator.
- After: baseline evaluator and test coverage exist; ready to run once assets are available.

---

## 8. Judgment For Paper Direction

Current iteration supports the direction at infrastructure level only.
Empirical support is pending live run data.

---

## 9. Next Actions

1. Replace caption-proxy references with teacher outputs/logits-based references.
2. Add semantic agreement metric (token-F1 / BLEU / BERTScore) in parallel with exact-match.
3. Expand from 8 to 20/50 samples and add OCR/digits tagged subsets.
4. Add at least one more budget config (e.g., different bottleneck/transmission setting).
