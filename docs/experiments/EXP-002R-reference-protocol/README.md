# EXP-002R: Reference Protocol

## 1. 基本信息
- 实验编号：EXP-002R
- 实验标题：Reference Protocol
- 所属阶段：B+
- 当前状态：in_progress
- 负责人：Agent 03
- 创建日期：2026-04-14
- 最后更新：2026-04-14

## 2. 实验目标
把“参考答案/参考分布”的确定过程独立出来，形成后续实验统一口径。

本轮聚焦：
1. 固化 AI2D 多选 VQA 的 reference schema。
2. 固化 teacher 输出字段（`teacher_output`）与 student compact 对比字段。
3. 固化输出解析、label 对比、loss 统计规则，供 EXP-002/003 复用。

## 3. 参考口径（v1）
### 3.1 样本字段规范（jsonl）
每行一个样本，核心字段：
- `sample_id`
- `image`
- `question`
- `prompt`
- `options`（`{"A":"...","B":"..."...}`）
- `label`（标准答案字母）
- `reference`（v1 与 `label` 相同）
- `subsets`
- `teacher_output`（由 probe 或 teacher 生成流程写回）

### 3.2 输出解析规则（v1）
对 teacher/student 文本输出做统一解析：
1. 优先识别显式字母模式：`(C)`、`answer: C`、`option C`、单字符 `C`。
2. 若未识别字母，回退到选项文本匹配（仅匹配长度>=3 的选项文本，避免单字母误匹配）。
3. 若仍失败，视为“无有效选项”。

### 3.3 打分与 loss（v1）
- `teacher_correct`：teacher 解析答案是否命中 `label`
- `student_correct`：student 解析答案是否命中 `label`
- `teacher_label_loss`：`teacher_correct` 命中=0，未命中=1
- `student_label_loss`：`student_correct` 命中=0，未命中=1
- `distill_loss`：student 解析结果与 teacher 解析结果一致=0，否则=1

## 4. AI2D 数据集下载与使用
详细说明见：
- `docs/experiments/EXP-002R-reference-protocol/artifacts/README.md`

最小命令（下载并生成 20 样本）：
```bash
conda run -n splitoculo python scripts/prepare_ai2d_reference.py \
  --hf-dataset lmms-lab/ai2d \
  --hf-split test \
  --max-samples 20 \
  --output docs/experiments/EXP-002R-reference-protocol/artifacts/ai2d_eval20.jsonl
```

## 5. 本轮执行入口
### 5.1 输出格式探针（teacher + student compact）
```bash
conda run -n splitoculo python scripts/probe_vqa_output_format.py \
  --dataset docs/experiments/EXP-002R-reference-protocol/artifacts/ai2d_eval20.jsonl \
  --output docs/experiments/EXP-002R-reference-protocol/artifacts/probe_run_5.json \
  --max-samples 5 \
  --teacher-model "$MODEL_DIR" \
  --teacher-local-only \
  --teacher-device mps \
  --student-checkpoint checkpoints/gan_bottleneck/split/edge_weights.pth \
  --student-device mps \
  --server http://127.0.0.1:8080 \
  --write-teacher-jsonl docs/experiments/EXP-002R-reference-protocol/artifacts/ai2d_eval20_teacher5.jsonl
```

### 5.2 统一评测（含 label/loss）
```bash
conda run -n splitoculo python scripts/eval_answer_agreement.py \
  --dataset <teacher_enriched_jsonl> \
  --server http://127.0.0.1:8080 \
  --budget v22=checkpoints/gan_bottleneck/split/edge_weights.pth \
  --require-teacher-output \
  --device mps \
  --output-dir docs/experiments/EXP-002R-reference-protocol/artifacts/<run_name>
```

## 6. 与后续实验关系
- EXP-002：可替换 caption-proxy reference，回填 teacher-based 口径。
- EXP-003：直接复用本协议字段与 loss 指标进行 selector 比较。

## 7. 本轮结论
1. AI2D 数据准备与 schema 已落地。
2. teacher/student compact 输出解析规则已固化为 v1。
3. 5 样本 probe 已验证 teacher 输出模式稳定（单字母），student 侧存在明显退化，需要在后续正式评测中量化。
