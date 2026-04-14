# EXP-001：Bootstrap & Baseline Recovery

## 1. 基本信息

- 实验编号：EXP-001
- 实验标题：Bootstrap & Baseline Recovery
- 所属阶段：阶段 A
- 当前状态：completed
- 负责人：Agent 01
- 创建日期：2026-04-14
- 最后更新：2026-04-14

---

## 2. 实验目标

本实验不是论文方法实验，而是启动实验。  
它需要回答以下问题：

1. 当前 `SplitOculo` 工程是否已经存在可复用的最小运行闭环
2. 当前工程的主流程、目录结构、模块边界、运行入口是否足够清晰
3. 哪些脚本和模块将成为后续实验的基础
4. 当前工程与目标论文方向之间的差距是什么
5. 后续应优先开展哪个实验，而不是直接进入大规模实现

---

## 3. 本实验与论文主线的关系

本实验服务于后续所有实验，但不直接验证论文方法。  
其作用是为后续 agent 建立：

- 工程结构统一认知
- 可复用实验入口
- 最小运行路径
- 后续实验插入点判断
- 文档化实验体系

若本实验做不好，后续实验会出现重复阅读工程、重复踩坑、路径不一致等问题。

---

## 4. 前置依赖

本实验执行前，应阅读：

1. `docs/MASTER_EXPERIMENT_PLAN.md`
2. `README.md`
3. `README-zh.md`
4. 当前工程中与运行、边端、云端、模型组装、量化/传输相关的核心脚本

---

## 5. 本实验的范围

### 5.1 应做内容
- 梳理工程目录和关键模块
- 梳理边端—云端主流程
- 确认至少一条最小运行路径
- 确认后续 agent 建议阅读路径
- 确认可能的实验插入点
- 建立整体框架文档
- 建立实验总索引

### 5.2 不应做内容
- 不实现 selector
- 不做大规模训练
- 不做大规模 benchmark
- 不重构整个工程
- 不跳过本实验直接进入后续阶段

---

## 6. 需要产出的文件

本实验至少应产出：

1. `docs/PROJECT_FRAMEWORK.md`
2. `docs/EXPERIMENT_INDEX.md`
3. `docs/experiments/EXP-001-bootstrap-baseline/RESULTS.md`
4. `docs/experiments/EXP-001-bootstrap-baseline/ACCEPTANCE.md`

---

## 7. 关注问题

本实验重点关注：

1. 当前工程的运行主路径在哪里
2. 哪些模块是后续实验最可能修改的
3. 哪些模块相对稳定，不应轻易改动
4. 当前工程最大的优势和最大缺口分别是什么
5. 未来最优先应开展的实验是哪个

---

## 8. 执行计划

### Step 1：阅读与梳理
- 阅读关键入口文件
- 记录工程目录和模块分工
- 识别边端、云端、压缩、恢复、评测相关脚本

### Step 2：运行路径确认
- 确认至少一条最小运行路径
- 记录运行依赖、配置与注意事项
- 标记当前路径中的潜在坑点

### Step 3：形成框架文档
- 输出 `docs/PROJECT_FRAMEWORK.md`
- 明确后续 agent 最小阅读路径
- 明确实验插入点建议

### Step 4：结果判断
- 形成 `RESULTS.md`
- 回答当前工程是否足以支撑后续论文实验
- 指出最优先开展的后续实验

---

## 9. 本次实际阅读范围

### 9.1 已重点阅读
- `docs/MASTER_EXPERIMENT_PLAN.md`
- `README.md`
- `README-zh.md`
- `scripts/precompute_qwen_features.py`
- `scripts/train_gan.py`
- `scripts/split_checkpoint.py`
- `scripts/cloud_server.py`
- `scripts/edge_client.py`
- `scripts/infer_hybrid.py`
- `core/qwen_extractor.py`
- `models/bottleneck.py`
- `models/cloud_upsampler.py`
- `electron_gui/main.js`
- `electron_gui/README.md`

### 9.2 已局部阅读
- `core/framework.py`
- `core/utils.py`
- `data/dataset.py`
- `cpp_edge_client/README.md`

### 9.3 暂不深入的部分
- `electron_gui/renderer/*`
- `cpp_edge_client/src/main.cpp`
- `models/mobile_vlm.py` 与其他备用骨干

原因：这些内容对完成 bootstrap 判断有帮助，但不是当前训练与部署主路径。

---

## 10. 预期结论模板

本实验最终应至少得出如下结论中的一类：

### 情况 A：工程基础清晰，可直接进入后续实验
- 当前工程已具备稳定主路径
- 只需补齐评测与实验接口即可

### 情况 B：工程可用，但仍需先做评测骨架
- 当前系统能跑，但论文目标所需指标缺失
- 应优先进入 `EXP-002`

### 情况 C：工程理解仍不足，需补充整理
- 当前运行路径不清晰或多处阻塞
- 需先补齐框架文档与运行说明后再继续

---

## 11. 本次结论落点

本次 `EXP-001` 的实际结论属于“情况 B”：

1. 当前工程主干已经完整，训练、拆权重、边云推理都存在。
2. 后续 agent 已可以不全量扫仓库，而先依赖 `docs/PROJECT_FRAMEWORK.md` 与本实验文档。
3. 但论文主线所需的 answer agreement、预算统计和 selector 对照评测仍未建立。
4. 因此后续最优先应进入 `EXP-002`，而不是直接做 selector 或大规模训练。

---

## 12. 更新记录

### 2026-04-14
- 初始化实验文档模板

### 2026-04-14
- 基于仓库关键入口文件完成 bootstrap 阅读与范围收口
- 明确本实验结论属于“工程可用，但仍需先做评测骨架”
