# EXP-002R Results

## 1. 基本信息
- 实验编号：EXP-002R
- 实验标题：Reference Protocol
- 日期：2026-04-14
- 状态：in_progress

## 2. 本轮修改记录（参考 EXP-002 记录风格）
### 2.1 新增/修改文件
- 新增：`scripts/prepare_ai2d_reference.py`
- 新增：`scripts/probe_vqa_output_format.py`
- 修改：`scripts/eval_answer_agreement.py`
- 修改：`tests/test_eval_answer_agreement.py`
- 修改：`docs/experiments/EXP-002R-reference-protocol/README.md`
- 修改：`docs/experiments/EXP-002R-reference-protocol/RESULTS.md`
- 修改：`docs/experiments/EXP-002R-reference-protocol/ACCEPTANCE.md`
- 新增：`docs/experiments/EXP-002R-reference-protocol/artifacts/README.md`
- 新增：`docs/experiments/EXP-002R-reference-protocol/artifacts/ai2d_eval20.jsonl`
- 新增：`docs/experiments/EXP-002R-reference-protocol/artifacts/probe_run_5.json`
- 新增：`docs/experiments/EXP-002R-reference-protocol/artifacts/ai2d_eval20_teacher5.jsonl`

### 2.2 验证命令
```bash
python -m unittest tests/test_eval_answer_agreement.py -v
conda run -n splitoculo python scripts/prepare_ai2d_reference.py --hf-dataset lmms-lab/ai2d --hf-split test --max-samples 20 --output docs/experiments/EXP-002R-reference-protocol/artifacts/ai2d_eval20.jsonl
```

### 2.3 验证结果
- 单测：7/7 通过。
- AI2D：成功生成 20 样本评测集（含 image/prompt/options/label/reference）。

## 3. 运行结果（仅基于已完成 probe）
### 3.1 Artifacts
- `docs/experiments/EXP-002R-reference-protocol/artifacts/probe_run_5.json`
- `docs/experiments/EXP-002R-reference-protocol/artifacts/ai2d_eval20_teacher5.jsonl`

### 3.2 5 样本 probe 观察
1. teacher 输出格式稳定：5/5 为单字母（A/B/C/D）。
2. teacher 解析后与 label 对齐：5/5 命中。
3. student compact 输出存在明显异常：
   - 样例输出包括 `Establish`、`群众`、`[,]`、`৺`、`Sun`
   - 仅 1 条能解析成选项字母，其余为空
   - 对 label 命中率：0/5

## 4. 协议结论（v1）
1. reference 字段采用 `label`（多选字母）作为统一基准。
2. teacher 口径采用 `teacher_output` 文本 + 统一解析规则。
3. student 口径采用 split cloud 返回文本 + 同一解析规则。
4. loss 口径采用二值误差：
   - `teacher_label_loss`
   - `student_label_loss`
   - `distill_loss`

## 5. 风险与限制
1. 当前仅有 5 样本 teacher-enriched probe 结果，尚未形成完整 20 样本 `summary.json` 表。
2. `student compact` 在 AI2D 多选任务上的输出质量明显退化，需要后续系统性排查（模型、prompt、decode、层匹配等）。

## 6. 下一步建议
1. 用同一命令将 teacher 写回扩展到 20 样本（`ai2d_eval20_teacher20.jsonl`）。
2. 运行 `eval_answer_agreement.py --require-teacher-output` 产出完整 20 样本 loss/accuracy 结果表。
3. 在 EXP-002 中回填 teacher-reference 口径，替代 caption-proxy。
