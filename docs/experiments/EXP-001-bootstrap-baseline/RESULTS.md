# EXP-001 结果记录

## 1. 基本信息

- 实验编号：EXP-001
- 实验标题：Bootstrap & Baseline Recovery
- 负责人：Agent 01
- 日期：2026-04-14
- 当前状态：completed

---

## 2. 本实验回答的问题

本实验需要回答：

1. 当前工程是否存在可复用的最小运行闭环
2. 当前工程结构是否足以支撑后续论文实验
3. 后续 agent 是否可以不全量阅读仓库而开展实验
4. 当前工程对目标论文方向的支持与缺口是什么

---

## 3. 阅读与梳理范围

### 3.1 已阅读内容
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

### 3.2 局部重点阅读内容
- `core/framework.py`
- `core/utils.py`
- `data/dataset.py`
- `cpp_edge_client/README.md`

### 3.3 未深入阅读但已判断可暂缓的内容
- `electron_gui/renderer/`
- `cpp_edge_client/src/main.cpp`
- `models/mobile_vlm.py`
- `models/levit.py`
- `models/mobile_vit.py`

---

## 4. 工程主流程总结

### 4.1 边端主流程

当前边端主流程以 `scripts/edge_client.py` 为准：

1. 从拆分后的边端 checkpoint 中加载学生 CNN、projector 和可选 bottleneck encoder。
2. 对输入图片做固定预处理。
3. 生成 transmission tokens。
4. 若启用 bottleneck，则先做维度压缩。
5. 将结果做 `uint8` 量化，并带上 `scale`、`zero_point`。
6. 把特征编码为 base64 后通过 HTTP 发送到云端。

这条路径说明：当前工程不是只做本地 tensor 流转，而是已经定义了真实的边端传输格式。

### 4.2 云端主流程

当前云端主流程以 `scripts/cloud_server.py` 为准：

1. 接收边端发送的 base64 特征与 prompt。
2. 完成反序列化、反量化。
3. 用 bottleneck decoder 恢复特征维度。
4. 用 upsampler 恢复到目标 Qwen 视觉层 token 数。
5. 依据 `split_layer` 做分布匹配。
6. 加载或复用 Qwen2.5-VL，从视觉中间层继续推理。
7. 返回回答文本与推理时延。

这说明当前工程已经具备“中间层恢复 -> Qwen 续推 -> 生成答案”的完整云端闭环。

### 4.3 最小运行闭环

当前最推荐的最小运行闭环是两段式：

1. 先用 `scripts/infer_hybrid.py` 在单机内验证 checkpoint 能完成“边端编码 -> 云端恢复 -> Qwen 续推”。
2. 再用 `scripts/split_checkpoint.py`、`scripts/cloud_server.py`、`scripts/edge_client.py` 跑真实 HTTP 边云推理。

运行入口：

- 离线闭环：`scripts/infer_hybrid.py`
- 在线闭环：`scripts/cloud_server.py` + `scripts/edge_client.py`

依赖脚本：

- `scripts/split_checkpoint.py`
- `scripts/cloud_server.py`
- `scripts/edge_client.py`
- `scripts/infer_hybrid.py`

限制：

1. 需要现成 checkpoint，否则从零训练成本很高。
2. 需要可用的 Qwen2.5-VL 权重。
3. 环境依赖声明不完整，`requirements.txt` 未覆盖所有实际依赖。

---

## 5. 当前工程的三个主要优点

### 优点 1
- 描述：训练链路与部署链路已经贯通，不是只有研究草图。
- 证据：`scripts/precompute_qwen_features.py`、`scripts/train_gan.py`、`scripts/split_checkpoint.py`、`scripts/cloud_server.py`、`scripts/edge_client.py` 形成完整串联。
- 对后续实验的意义：后续 agent 可以直接在真实主干上补评测或插方法，而不是先补工程底座。

### 优点 2
- 描述：切分层语义已经被明确建模，中间层对齐基础较扎实。
- 证据：`core/qwen_extractor.py` 支持 `-1/0/N` 层特征提取，README 也已有层级消融与分布统计总结。
- 对后续实验的意义：后续做 selector、必要性判断或预算实验时，有清晰的特征层定义可依赖。

### 优点 3
- 描述：预算控制点已经天然存在，方法插入空间清晰。
- 证据：`models/bottleneck.py` 控制传输维度，`scripts/edge_client.py` 负责量化和发送，`models/cloud_upsampler.py` 负责恢复。
- 对后续实验的意义：后续做 answer agreement、selector MVP 或预算日志时，不需要大改目录，只需围绕 transmission tokens 主路径插入。

---

## 6. 当前工程的三个主要缺口

### 缺口 1
- 描述：缺少 answer agreement under budget 的统一评测入口。
- 影响：即使训练和部署能跑，也无法直接回答论文核心问题。
- 是否阻塞后续实验：阻塞 `EXP-003` 之后的方法比较，但不阻塞 `EXP-002` 的建立。

### 缺口 2
- 描述：依赖与运行环境说明不完整。
- 影响：后续 agent 可能因为 `transformers`、Qwen 模型、本地缓存等问题重复踩坑。
- 是否阻塞后续实验：不完全阻塞，但会拖慢复现和协作效率。

### 缺口 3
- 描述：当前结果主要围绕特征恢复和 split-layer 分析，缺少 selector 插入后的标准对照结构。
- 影响：若直接进入方法实现，很容易做出无法解释的实验结果。
- 是否阻塞后续实验：阻塞直接进入 `EXP-003` 及之后的必要性、区分性验证。

---

## 7. 对目标论文方向的支撑判断

### 7.1 当前已具备的基础
- 已有 split inference 主流程
- 已有压缩/传输基础
- 已有切分层中间特征提取与训练主干
- 已有边端/云端拆权重与真实 HTTP 推理能力
- 已有离线混合推理路径，可做最小验证
- 已有部分 split-layer 与分布统计分析能力

### 7.2 当前仍不足的地方
- 缺少 answer agreement 评测
- 缺少预算统一评测口径
- 缺少 selector 插入实验骨架
- 缺少面向论文主线的必要性与区分性验证入口
- 依赖说明和运行说明仍需补全

### 7.3 综合判断

综合判断：当前工程可支撑后续实验，但必须先补齐评测骨架。

理由：

1. 工程主干已经完整，训练、拆权重、在线推理都存在。
2. 后续 agent 已可基于 `docs/PROJECT_FRAMEWORK.md` 和本实验结果快速上手，不需要再全量扫仓库。
3. 但论文核心不是“能恢复特征”，而是“在预算下是否保持正确答案”，这一层目前仍未形成评测闭环。

---

## 8. 推荐的后续实验顺序

### 最优先实验
- 推荐编号：EXP-002
- 推荐原因：现有工程最缺的不是主干实现，而是与论文目标对齐的答案保持型评测口径。
- 预期回答的问题：
  1. 在不同预算下，当前系统的回答保持率是多少。
  2. 不同 split 配置与预算设置对答案一致性影响多大。
  3. selector 后续应以什么评测接口为目标。

### 不建议立刻做的内容
- 内容：直接实现 selector、直接做大规模训练、直接做大规模 benchmark。
- 原因：没有统一评测口径前，方法实验很难解释，训练投入也无法形成可靠结论。

---

## 9. 关键路径与插入点判断

### 9.1 若后续做 answer agreement 评测，建议修改位置
- 模块/脚本：`scripts/edge_client.py`、`scripts/cloud_server.py`，以及新增独立评测脚本。
- 理由：当前回答文本就是从这条边云链路产出的，在这里采集 prompt、回答、预算、延迟最自然。

### 9.2 若后续加 selector，建议插入位置
- 模块/脚本：优先是 `scripts/train_gan.py` 与 `scripts/edge_client.py`，具体插入点是 projector 输出后、量化发送前。
- 理由：这个位置已经形成 transmission tokens，既靠近预算控制点，也能尽量少破坏云端恢复主干。

### 9.3 若后续加实验日志/预算统计，建议插入位置
- 模块/脚本：`scripts/train_gan.py`、`scripts/edge_client.py`、`scripts/cloud_server.py`。
- 理由：训练统计、发送预算和最终回答分布分别在这三处最容易拿到原始信息。

---

## 10. 是否建议进入 EXP-002

- 建议：是
- 理由：当前 bootstrap 已确认工程主干清晰，后续最缺的是论文目标对齐的评测骨架，正好对应 `EXP-002`。
- 若否，阻塞点是：无

---

## 11. 本实验最终结论

请用 3～8 条简洁结论总结：

1. 当前工程已经具备从训练到边云推理的最小完整主干，不是只有概念验证代码。
2. 后续 agent 已可以优先依赖 `docs/PROJECT_FRAMEWORK.md`，无需再次全量阅读仓库。
3. 真实主路径是 `precompute -> train_gan -> split_checkpoint -> cloud_server + edge_client`，离线最小验证路径是 `infer_hybrid.py`。
4. transmission tokens 形成于边端 projector 之后，这里是后续 selector 和预算控制最自然的插入点。
5. 当前最大的空白不是主干实现，而是 answer agreement、预算统计和 selector 对照评测。
6. 因此后续最合理的下一步是启动 `EXP-002`，而不是直接进入方法实现或大规模训练。

---

## 12. 后续行动建议

### 立刻应该做的事
1. 建立基于 `scripts/edge_client.py` 与 `scripts/cloud_server.py` 的 answer agreement baseline。
2. 定义统一预算口径，至少覆盖 transmission token 数、bottleneck 维度、payload 大小和回答文本。
3. 形成一个独立的小样本评测脚本，避免每次手工点 GUI 或人工对比日志。

### 暂时不要做的事
1. 不要一开始就重写训练主干或重构目录。
2. 不要在没有统一评测指标前直接做 selector 对比实验。
3. 不要把 GUI 或 C++ 客户端当成论文主线优先事项。

---

## 13. 更新记录

### 2026-04-14
- 初始化结果文档模板

### 2026-04-14
- 补充工程主流程、最小运行闭环、主要优点与主要缺口

### 2026-04-14
- 完成综合判断、后续实验优先级与插入点建议
