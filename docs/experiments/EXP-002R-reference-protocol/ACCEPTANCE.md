# EXP-002R Acceptance Checklist

## 1. Basic Info
- Experiment ID: EXP-002R
- Title: Reference Protocol
- Reviewer: Agent 03
- Date: 2026-04-14
- Status: in_progress

---

## 2. Checklist
- [x] 我已阅读主文档
- [x] 我已阅读整体框架文档
- [x] 我已阅读 EXP-002 结果文档
- [x] 我已定义 reference 字段规范
- [x] 我已明确 teacher answer/logits 口径
- [x] 我已给出口径版本与变更说明
- [x] 我已同步更新实验索引
- [x] 我已给出 EXP-003 复用说明

---

## 3. Completion Scope For This Iteration
Completed:
1. AI2D 评测数据准备脚本已实现并完成 20 样本数据集生成。
2. 输出格式探针脚本已实现，并完成 5 样本 teacher/student compact 输出记录。
3. `eval_answer_agreement.py` 已扩展支持：
   - VQA 多选输出解析
   - label 对比打分
   - teacher/student/distill 三类 loss 统计
4. 对应单测已补齐并通过。
5. 2R 文档（README/RESULTS/ACCEPTANCE）与实验索引已同步。

Pending:
1. 20 样本 teacher-enriched 全量评测产物（`summary.json`）尚未完成。

Reason:
- 本轮以“已有运行结果记录与协议固化”为主，先固定口径与工具链。

---

## 4. Changed Files
- `scripts/prepare_ai2d_reference.py`
- `scripts/probe_vqa_output_format.py`
- `scripts/eval_answer_agreement.py`
- `tests/test_eval_answer_agreement.py`
- `docs/experiments/EXP-002R-reference-protocol/README.md`
- `docs/experiments/EXP-002R-reference-protocol/RESULTS.md`
- `docs/experiments/EXP-002R-reference-protocol/ACCEPTANCE.md`
- `docs/experiments/EXP-002R-reference-protocol/artifacts/README.md`
- `docs/experiments/EXP-002R-reference-protocol/artifacts/ai2d_eval20.jsonl`
- `docs/experiments/EXP-002R-reference-protocol/artifacts/probe_run_5.json`
- `docs/experiments/EXP-002R-reference-protocol/artifacts/ai2d_eval20_teacher5.jsonl`
- `docs/EXPERIMENT_INDEX.md`

---

## 5. 记录同步说明（Required）
1. 结果记录路径：
   - `docs/experiments/EXP-002R-reference-protocol/RESULTS.md`
2. 设计记录路径：
   - `docs/experiments/EXP-002R-reference-protocol/README.md`
3. 是否更新 `docs/EXPERIMENT_INDEX.md`：
   - 是，已更新
4. 新增脚本/文件路径：
   - Script: `scripts/prepare_ai2d_reference.py`
   - Script: `scripts/probe_vqa_output_format.py`
   - Script: `scripts/eval_answer_agreement.py`
   - Test: `tests/test_eval_answer_agreement.py`
   - Artifacts: `docs/experiments/EXP-002R-reference-protocol/artifacts/*`
