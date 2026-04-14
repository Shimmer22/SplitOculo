# EXP-002R Artifacts Usage

## 1. 数据来源
- 数据集：`lmms-lab/ai2d`（HuggingFace）
- 用途：多选 VQA 参考口径（label-based reference）

## 2. 下载并生成评测集
在 `splitoculo` 环境执行：

```bash
conda run -n splitoculo python scripts/prepare_ai2d_reference.py \
  --hf-dataset lmms-lab/ai2d \
  --hf-split test \
  --max-samples 20 \
  --output docs/experiments/EXP-002R-reference-protocol/artifacts/ai2d_eval20.jsonl
```

说明：
1. 脚本会把 HF 图像样本落盘到 `data/ai2d/images/`。
2. 输出 jsonl 已是 EXP-002R v1 schema，可直接用于 probe/eval。

## 3. 生成 teacher 输出并探测 teacher/student 输出格式
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

产物：
- `probe_run_5.json`：包含 teacher/student 原始输出与解析结果。
- `ai2d_eval20_teacher5.jsonl`：在原数据上新增 `teacher_output` 字段。

## 4. 正式评测（label 对比 + loss）
```bash
conda run -n splitoculo python scripts/eval_answer_agreement.py \
  --dataset docs/experiments/EXP-002R-reference-protocol/artifacts/ai2d_eval20_teacher5.jsonl \
  --server http://127.0.0.1:8080 \
  --budget v22=checkpoints/gan_bottleneck/split/edge_weights.pth \
  --require-teacher-output \
  --device mps \
  --output-dir docs/experiments/EXP-002R-reference-protocol/artifacts/run-xxx
```

输出：
- `summary.json`
- `rows.json`
- `failures.json`
- `summary.md`

## 5. 字段说明（核心）
- 输入：`label`, `options`, `teacher_output`, `prediction`
- 输出：`teacher_choice`, `student_choice`, `teacher_correct`, `student_correct`
- loss：`teacher_label_loss`, `student_label_loss`, `distill_loss`
