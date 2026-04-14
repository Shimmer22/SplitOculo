# EXP-002: Answer Agreement Baseline

## 1. Basic Info

- Experiment ID: EXP-002
- Title: Answer Agreement Baseline
- Stage: B
- Owner: Agent 02
- Date: 2026-04-14
- Status: in_progress

---

## 2. Goal

Build a reproducible baseline to answer:

1. Under different transmission budgets, does the split pipeline preserve final answers?
2. Can we report budget with a unified protocol (bytes/bits/tokens)?
3. Can we slice metrics by evidence-sensitive subsets (OCR, digits, fine-grained tags)?

---

## 3. Relation To Paper Direction

This experiment establishes the evaluation gate before selector work (EXP-003+).
Without this baseline, any later method gain is not comparable.

---

## 4. Preconditions

Required reads:

1. `docs/MASTER_EXPERIMENT_PLAN.md`
2. `docs/PROJECT_FRAMEWORK.md`
3. `docs/experiments/EXP-001-bootstrap-baseline/RESULTS.md`

Required runtime assets:

1. Edge checkpoint(s)
2. Cloud server for matching checkpoint
3. Evaluation image set and references

Current prepared assets (2026-04-14):

1. `checkpoints/releases/v2.2/edge_weights.pth`
2. `checkpoints/releases/v2.2/cloud_weights.pth`
3. `checkpoints/gan_bottleneck/split/{edge_weights.pth,cloud_weights.pth}` (symlinked to v2.2)
4. `data/coco/val2017` (5000 images)
5. `data/coco/train` and `data/coco/val` (symlinked to `val2017` for script compatibility)

---

## 5. Implementation Scope In This Iteration

1. Add a dedicated baseline evaluator script:
   - `scripts/eval_answer_agreement.py`
2. Define one concrete agreement statistic:
   - normalized exact-match agreement rate
3. Add budget metrics:
   - average payload bytes
   - average payload bits
4. Add subset statistics:
   - per-subset sample count and agreement rate
5. Add reproducible outputs:
   - `summary.json`, `rows.json`, `failures.json`, `summary.md`

---

## 6. Data Format

Input supports `.json` / `.jsonl`.

Each sample should contain:

- `image` or `image_path`
- `reference` (or `reference_answer` / `target_answer`)

Optional fields:

- `sample_id`
- `prompt`
- `subsets` (or `subset`)

See template:

- `docs/experiments/EXP-002-answer-agreement-baseline/artifacts/sample_eval_dataset.jsonl`

---

## 7. Execution Commands

### 7.1 Unit Tests (TDD)

```bash
python -m unittest tests/test_eval_answer_agreement.py -v
```

### 7.2 CLI Check

```bash
python scripts/eval_answer_agreement.py --help
```

### 7.3 Real Evaluation (when assets ready)

```bash
python scripts/eval_answer_agreement.py \
  --dataset <path-to-eval-jsonl> \
  --server http://127.0.0.1:8080 \
  --budget b16=<edge_ckpt_b16.pth> \
  --budget b32=<edge_ckpt_b32.pth> \
  --output-dir docs/experiments/EXP-002-answer-agreement-baseline/artifacts/run-001
```

---

## 8. Current Iteration Conclusion

Completed in this iteration:

1. Baseline evaluator script implemented.
2. Core metric logic covered by tests.
3. Experiment docs and acceptance checklist prepared.

Not completed yet:

1. Real answer agreement table with live edge/cloud inference.

Reason:

- No local checkpoint/image benchmark assets found in repository snapshot.

---

## 9. Next Step

1. Install runtime dependencies in the active environment (`torch`, `timm`, `transformers`, `Pillow`, etc.).
2. Prepare a small evaluation set (>= 20 samples) with references and subset tags.
3. Start cloud service with `checkpoints/gan_bottleneck/split/cloud_weights.pth`.
4. Run command in section 7.3 with `--budget v22=checkpoints/gan_bottleneck/split/edge_weights.pth`.
5. Update `RESULTS.md`, `ACCEPTANCE.md`, and `docs/EXPERIMENT_INDEX.md` status.
