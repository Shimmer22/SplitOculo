# 实验总索引

> 说明：本文件用于维护所有实验的统一索引。  
> 所有 agent 完成实验后，必须更新本文件。  
> 状态建议使用：`planned` / `in_progress` / `blocked` / `completed` / `archived`

---

## 索引表

| 编号 | 标题 | 阶段 | 状态 | 负责人 | 起始日期 | 最后更新 | 核心目标 | 核心结论 | 下一步 |
|---|---|---|---|---|---|---|---|---|---|
| EXP-001 | bootstrap-baseline | A | completed | Agent 01 | 2026-04-14 | 2026-04-14 | 建立整体框架文档，确认工程最小闭环与后续实验入口 | 当前工程可支撑后续实验，但必须先补齐答案保持型评测与预算统计骨架 | 启动 EXP-002，建立 answer agreement baseline |
| EXP-002 | answer-agreement-baseline | B | in_progress | Agent 02 | 2026-04-14 | 2026-04-14 | 建立答案保持型评测基线与预算评测口径 | 已完成 20 样本实测（run-20260414_141417：0 失败，平均约 4.1KB/样本），当前为 caption-proxy 参考口径 | 切换 teacher-reference 口径并扩展到 50+ 样本与多预算对照 |
| EXP-002R | reference-protocol | B+ | planned | TBD | TBD | 2026-04-14 | 将参考答案/参考分布定义过程独立成可复用实验协议 | 尚未开始；目的是在 EXP-003 前固定 teacher reference 口径 | 输出 teacher answer/logits 口径与字段规范，供后续实验统一复用 |
| EXP-003 | selector-mvp | C | planned | TBD | TBD | 2026-04-14 | 实现最小可行 selector，验证是否存在正向信号 | 需等待 EXP-002 给出统一评测口径后再推进 | 在 transmission tokens 位置插入最小 selector |
| EXP-004 | necessity-check | D | planned | TBD | TBD | 2026-04-14 | 验证方法是否比 relevance-aware 更接近 necessity-aware | 依赖 EXP-003 先得到可比较的 selector 原型 | 做删除实验、补集退化与失败案例对照 |
| EXP-005 | discriminative-check | E | planned | TBD | TBD | 2026-04-14 | 验证方法是否更能保持决策边界与竞争答案区分性 | 依赖 EXP-004 建立必要性证据后再继续 | 评估是否值得扩大规模与进入更正式实验 |

---

## 状态说明

- `planned`：已规划，尚未正式开始
- `in_progress`：正在执行
- `blocked`：存在阻塞，无法继续
- `completed`：已完成，结果已记录
- `archived`：已归档，不再继续

---

## 更新规则

每次更新本文件时，必须至少同步以下内容：

1. 更新对应实验的状态
2. 更新最后更新时间
3. 补充核心结论
4. 写明下一步建议
5. 若实验被阻塞，写明阻塞原因

---

## 维护日志

### 2026-04-14
- 初始化实验索引并完成首轮填写
- 将 `EXP-001` 更新为 `completed`
- 记录 `EXP-001` 结论：当前工程可支撑后续实验，但需先补齐 answer agreement 评测与预算统计骨架
- 明确下一步应优先启动 `EXP-002`
- 启动 `EXP-002`，状态更新为 `in_progress`
- 记录 `EXP-002` 当前进展：已新增 `scripts/eval_answer_agreement.py` 与测试，待真实推理结果回填
- 回填 `EXP-002` 首次真实推理结果（run-003）：8 样本全部成功，agreement 在 caption-proxy 严格精确匹配下为 0.0
- 回填 `EXP-002` 扩展结果（run-20260414_141417）：20 样本全部成功，平均 payload 4184 bytes，agreement 仍为 0.0
- 新增 `EXP-002R` 规划，用于在进入 EXP-003 前单独固化 reference protocol
