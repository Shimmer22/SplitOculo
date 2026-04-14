# EXP-002 Acceptance Checklist

## 1. Basic Info

- Experiment ID: EXP-002
- Title: Answer Agreement Baseline
- Reviewer: Agent 02
- Date: 2026-04-14
- Status: in_progress

---

## 2. Checklist

- [x] 我已阅读主文档
- [x] 我已阅读整体框架文档
- [x] 我已阅读本实验文档
- [x] 我已明确本实验回答的问题
- [x] 我已列出所修改或新增的文件
- [x] 我已在实验文档中记录实验设计
- [x] 我已在结果文档中记录结果与结论
- [x] 我已说明实验失败或成功原因
- [x] 我已给出是否继续下一阶段的建议
- [x] 我已更新实验索引

---

## 3. Completion Scope For This Iteration

Completed:

1. Added answer-agreement baseline evaluator script.
2. Added unit tests for agreement and aggregation logic.
3. Added EXP-002 docs set (`README/RESULTS/ACCEPTANCE`).
4. Added sample dataset template.
5. Downloaded official v2.2 checkpoints and COCO val2017 dataset, and linked them to expected paths.
6. Completed first live run (`run-003`) with 8 samples, 0 request failures.
7. Completed expanded live run (`run-20260414_141417`) with 20 samples, 0 request failures.

Pending:

1. Teacher-based reference pipeline (teacher answer/logits) for paper-aligned agreement target.

Reason:

- Current run uses caption-proxy references; this validates pipeline but is not the final paper-target reference.

---

## 4. Changed Files

- `scripts/eval_answer_agreement.py`
- `tests/test_eval_answer_agreement.py`
- `docs/experiments/EXP-002-answer-agreement-baseline/README.md`
- `docs/experiments/EXP-002-answer-agreement-baseline/RESULTS.md`
- `docs/experiments/EXP-002-answer-agreement-baseline/ACCEPTANCE.md`
- `docs/experiments/EXP-002-answer-agreement-baseline/artifacts/sample_eval_dataset.jsonl`
- `docs/EXPERIMENT_INDEX.md`

---

## 5. Record Sync Statement (Required)

1. 本次实验结果记录到了哪个 `RESULTS.md`
   - `docs/experiments/EXP-002-answer-agreement-baseline/RESULTS.md`
2. 本次实验设计更新到了哪个 `README.md`
   - `docs/experiments/EXP-002-answer-agreement-baseline/README.md`
3. 本次实验是否更新了 `docs/EXPERIMENT_INDEX.md`
   - 是，已更新
4. 若新增了脚本/指标/图表，分别放在哪些路径
   - Script: `scripts/eval_answer_agreement.py`
   - Tests: `tests/test_eval_answer_agreement.py`
   - Dataset template: `docs/experiments/EXP-002-answer-agreement-baseline/artifacts/sample_eval_dataset.jsonl`

---

## 6. Final Acceptance Note

Current status is `in_progress` because empirical live-run evidence is not generated yet. Baseline infrastructure and documentation are ready.
