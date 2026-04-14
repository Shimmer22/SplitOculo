# EXP-002R: Reference Protocol

## 1. 基本信息
- 实验编号：EXP-002R
- 实验标题：Reference Protocol
- 所属阶段：B+
- 当前状态：planned
- 负责人：TBD
- 创建日期：2026-04-14

## 2. 实验目标
把“参考答案/参考分布”的确定过程独立出来，形成后续实验统一口径。

核心问题：
1. 参考应基于 teacher 文本答案、teacher logits，还是两者结合。
2. 文本后处理如何统一（大小写、标点、同义表达）。
3. 字段和版本如何固化，避免后续实验口径漂移。

## 3. 预期产物
1. reference schema（样本字段规范）
2. teacher 生成脚本与导出格式
3. reference version 文档（v1/v2 变更）
4. 与 EXP-002/003 对接说明

## 4. 与后续实验关系
- EXP-002 使用该口径回填主评测结果。
- EXP-003 直接复用该口径进行 selector 比较。

## 5. 进入条件
1. 已阅读 `docs/MASTER_EXPERIMENT_PLAN.md`
2. 已阅读 `docs/PROJECT_FRAMEWORK.md`
3. 已阅读 `docs/experiments/EXP-002-answer-agreement-baseline/RESULTS.md`

## 6. 验收目标
1. 同一输入在不同实验中 reference 一致。
2. 口径定义可复现且有版本记录。
3. EXP-003 不再重复讨论 reference 定义。
