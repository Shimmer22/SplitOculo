# EXP-001 验收单

## 1. 基本信息

- 实验编号：EXP-001
- 实验标题：Bootstrap & Baseline Recovery
- 验收人：Agent 01
- 日期：2026-04-14
- 当前状态：completed

---

## 2. 验收检查项

请逐项填写。未完成项不得勾选。

- [x] 我已阅读当前工程的关键入口文件
- [x] 我已阅读 `docs/MASTER_EXPERIMENT_PLAN.md`
- [x] 我已形成 `docs/PROJECT_FRAMEWORK.md`
- [x] 我已形成 `docs/EXPERIMENT_INDEX.md`
- [x] 我已建立 `docs/experiments/EXP-001-bootstrap-baseline/`
- [x] 我已完成 `README.md`
- [x] 我已完成 `RESULTS.md`
- [x] 我已在 `RESULTS.md` 中写明当前工程支撑能力与主要缺口
- [x] 我已在 `RESULTS.md` 中说明后续最建议优先开展的实验
- [x] 我已更新实验索引中的 `EXP-001`
- [x] 我已说明是否建议进入 `EXP-002`
- [x] 我已记录本次工作中阅读的关键路径
- [x] 我已记录本次工作中发现的关键脚本或模块路径

---

## 3. 修改与新增文件清单

请列出本次实际新增、修改、补充的文件。

### 3.1 新增文件
- `docs/PROJECT_FRAMEWORK.md`

### 3.2 修改文件
- `docs/EXPERIMENT_INDEX.md`
- `docs/experiments/EXP-001-bootstrap-baseline/README.md`
- `docs/experiments/EXP-001-bootstrap-baseline/RESULTS.md`
- `docs/experiments/EXP-001-bootstrap-baseline/ACCEPTANCE.md`

---

## 4. 关键阅读路径记录

请列出本次为完成任务而重点阅读的文件或目录。

1. `docs/MASTER_EXPERIMENT_PLAN.md`
   - 用途：确认实验阶段、约束、记录规范与后续路线。
   - 为什么重要：这是后续所有实验文档的上位约束。

2. `README.md` 与 `README-zh.md`
   - 用途：确认当前项目对外定义的主流程、运行入口和已有结果摘要。
   - 为什么重要：用于判断仓库当前公开主干与实验定位。

3. `scripts/train_gan.py`
   - 用途：确认当前训练主干、模型组装关系与 checkpoint 字段结构。
   - 为什么重要：后续实验插入点和主路径判断主要基于这里。

4. `scripts/cloud_server.py`
   - 用途：确认云端真实推理路径、恢复逻辑和回答输出接口。
   - 为什么重要：answer agreement 与预算评测最终都要围绕它建立。

5. `scripts/edge_client.py`
   - 用途：确认边端编码、量化和发送协议。
   - 为什么重要：selector 和预算控制最自然的插入点在这里。

6. `core/qwen_extractor.py`
   - 用途：确认中间层特征提取和 split layer 语义。
   - 为什么重要：决定后续实验到底在对齐和操作哪一层表示。

---

## 5. 本次工作的验收判断

### 5.1 是否完成本实验目标
- 结论：完成

### 5.2 理由
- 已形成统一工程入口文档 `docs/PROJECT_FRAMEWORK.md`。
- 已完成实验总索引初始化与 `EXP-001` 状态回填。
- 已将 `EXP-001` 的目标、结论、缺口、后续优先级写入实验文档。
- 已明确后续 agent 的最小阅读路径与实验插入点。

### 5.3 是否存在阻塞
- 否

### 5.4 若存在阻塞，请写明
- 阻塞点：无
- 影响：无
- 建议处理方式：无

---

## 6. 对后续实验的建议

### 建议优先开展的实验
- 实验编号：EXP-002
- 原因：当前最缺的是与论文目标一致的 answer agreement 与预算评测骨架，而不是再补一条新主干。

### 暂不建议开展的实验
- 实验编号或方向：EXP-003 及之后的方法实验；GUI/C++ 侧扩展；大规模训练。
- 原因：没有统一评测口径前，方法实验结论不可解释，工程扩展也不能直接增加论文证据。

---

## 7. 记录同步说明

本节为强制填写项。

### 7.1 本次实验设计记录到哪里
- `docs/experiments/EXP-001-bootstrap-baseline/README.md`

### 7.2 本次实验结果记录到哪里
- `docs/experiments/EXP-001-bootstrap-baseline/RESULTS.md`

### 7.3 本次实验验收记录到哪里
- `docs/experiments/EXP-001-bootstrap-baseline/ACCEPTANCE.md`

### 7.4 本次是否更新实验索引
- 是
- 若是，路径：
  - `docs/EXPERIMENT_INDEX.md`

### 7.5 本次是否生成或更新整体框架文档
- 是
- 若是，路径：
  - `docs/PROJECT_FRAMEWORK.md`

### 7.6 若发现了关键脚本或关键模块，请列出
- `scripts/precompute_qwen_features.py`
- `scripts/train_gan.py`
- `scripts/split_checkpoint.py`
- `scripts/infer_hybrid.py`
- `scripts/cloud_server.py`
- `scripts/edge_client.py`
- `core/qwen_extractor.py`
- `models/bottleneck.py`
- `models/cloud_upsampler.py`

---

## 8. 最终签收结论

请在此写出最终一句话结论：

> 当前工程已可支撑进入 `EXP-002`，建议优先补齐答案保持型评测与预算统计骨架；暂不建议直接进入 selector 方法实现。

---

## 9. 更新记录

### 2026-04-14
- 初始化验收单模板

### 2026-04-14
- 完成 EXP-001 验收填写
