# EXP-002R AI2D Reference Protocol Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 完成 AI2D 数据准备 + teacher/student compact 推理 + 输出解析 + label 对比 + loss 统计，并回填 EXP-002R 文档。

**Architecture:** 在现有 `eval_answer_agreement.py` 上扩展为 2R 通用评测入口：支持 teacher 与 student 双路推理、VQA 多选答案解析、标签打分与 loss 计算。新增 AI2D 数据准备脚本产出统一 jsonl 协议数据，并用临时探针脚本验证输出格式后固定解析规则。

**Tech Stack:** Python, requests, torch, transformers(Qwen2.5-VL), unittest, existing SplitOculo edge/cloud scripts

---

### Task 1: 先写失败测试（输出解析与 loss 聚合）

**Files:**
- Modify: `tests/test_eval_answer_agreement.py`
- Test: `tests/test_eval_answer_agreement.py`

- [ ] **Step 1: 为多选解析和 loss 统计补测试用例（先失败）**
- [ ] **Step 2: 运行 `python -m unittest tests/test_eval_answer_agreement.py -v` 并确认失败**

### Task 2: 实现评测扩展能力

**Files:**
- Modify: `scripts/eval_answer_agreement.py`

- [ ] **Step 1: 实现 VQA 输出解析（A/B/C/D 与文本回退）**
- [ ] **Step 2: 实现 teacher/student compact 双路记录与 label 对比打分**
- [ ] **Step 3: 实现 loss 统计字段（teacher_label_loss / student_label_loss / distill_loss）**
- [ ] **Step 4: 运行单测并通过**

### Task 3: 新增 AI2D 数据准备与输出探针脚本

**Files:**
- Create: `scripts/prepare_ai2d_reference.py`
- Create: `scripts/probe_vqa_output_format.py`

- [ ] **Step 1: 产出 AI2D -> jsonl 的 reference schema 生成脚本**
- [ ] **Step 2: 产出 teacher/student 输出格式探针脚本（调试用）**
- [ ] **Step 3: 运行 `--help`/最小 dry-run 验证脚本可执行**

### Task 4: 执行 2R 实验并回填文档

**Files:**
- Modify: `docs/experiments/EXP-002R-reference-protocol/README.md`
- Modify: `docs/experiments/EXP-002R-reference-protocol/RESULTS.md`
- Modify: `docs/experiments/EXP-002R-reference-protocol/ACCEPTANCE.md`
- Modify: `docs/EXPERIMENT_INDEX.md`

- [ ] **Step 1: 跑 AI2D 数据准备（若本地缺数据则记录阻塞）**
- [ ] **Step 2: 跑 teacher/student compact 推理评测并产出 artifacts**
- [ ] **Step 3: 回填 2R 文档与索引状态**
