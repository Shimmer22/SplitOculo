# SplitOculo 工程整体框架文档

## 1. 项目目标

SplitOculo 当前不是通用多模态工程项目，而是一个围绕“边云协同 VLM 切分推理”展开的研究原型。  
它要解决的问题是：端侧不上传原图，也不承载完整 VLM，而是在端侧提取压缩后的中间视觉特征，再由云端恢复并继续执行 Qwen2.5-VL 的视觉后半段与文本生成。

对后续实验而言，这个仓库的价值不只是“能训练一个模型”，而是已经同时具备三条基础能力：

1. 训练侧：可把 CNN 特征学习到 Qwen 视觉中间层表示。
2. 部署侧：可把训练产物拆成边端权重和云端权重，并通过 HTTP 跑真实边云推理。
3. 分析侧：可做切分层级对齐和特征分布统计，为后续实验提供先验。

---

## 2. 当前研究定位

当前仓库已经完成的是“Split Inference 原型”阶段，而不是“论文主方法验证”阶段。

更具体地说，现有代码已经回答了这些问题：

1. 能否用轻量学生端模型提取传输特征。
2. 能否把低分辨率特征上采样回 Qwen 视觉空间。
3. 能否把训练检查点拆成边端与云端两部分并跑真实推理链路。
4. 哪些 Qwen 视觉层更适合作为切分层。

但它还没有回答论文主线最关心的问题：

1. 在不同预算下，最终答案是否仍保持一致。
2. 哪些传输证据是真正“必要”的，而不只是“相关”的。
3. selector 应该插在什么位置，以及如何与现有传输链路结合。

因此，后续 agent 不应把这里当成“缺一个方法实现的完整论文仓库”，而应把它当成“已经有主干链路，但缺少论文目标对齐评测和方法插槽”的实验底座。

---

## 3. 仓库结构总览

### 3.1 顶层目录职责

- `core/`
  - 共用基础模块。包含早期 split 实验框架、Qwen 特征提取器和通用工具。
- `models/`
  - 当前主干模型组件。重点是 bottleneck、projector、cloud upsampler。
- `scripts/`
  - 当前最重要的运行入口。训练、预计算、拆权重、边端、云端、离线推理都在这里。
- `data/`
  - 数据集和基础数据加载工具。训练脚本默认依赖 `train/`、`val/` 结构。
- `electron_gui/`
  - 一个桌面 GUI 壳层，本质上是对 `scripts/cloud_server.py` 和 `scripts/edge_client.py` 的可视化封装。
- `cpp_edge_client/`
  - 面向 ONNX 的 C++ 边端客户端，适合后续做设备端演示或轻量部署验证。
- `docs/`
  - 当前实验协作入口。后续 agent 应优先阅读这里而不是全量扫代码。

### 3.2 当前最关键的文件

- `docs/MASTER_EXPERIMENT_PLAN.md`
  - 定义实验阶段、记录约束和后续实验路线。
- `scripts/precompute_qwen_features.py`
  - 静态训练数据的起点，用 Qwen 提取目标层视觉特征。
- `scripts/train_gan.py`
  - 当前主训练脚本，负责学生端、projector、bottleneck、upsampler、discriminator 的训练。
- `scripts/split_checkpoint.py`
  - 把单体训练产物拆成 `edge_weights.pth` 与 `cloud_weights.pth`。
- `scripts/cloud_server.py`
  - 云端推理服务，负责解码、上采样、恢复到 Qwen 视觉链路并产出回答。
- `scripts/edge_client.py`
  - 端侧推理客户端，负责图像编码、量化、HTTP 发送与结果接收。
- `scripts/infer_hybrid.py`
  - 单机离线混合推理路径，不依赖 HTTP，适合做最小功能验证和本地调试。
- `core/qwen_extractor.py`
  - 仓库里与 Qwen 中间层对齐关系最关键的模块。

---

## 4. 主流程说明

### 4.1 训练主流程

当前训练主路径是：

1. 用 `scripts/precompute_qwen_features.py` 从 Qwen2.5-VL 提取某一视觉层的目标特征。
2. 用 `scripts/train_gan.py` 训练学生端 CNN、projector、可选 bottleneck、云端 upsampler，以及 GAN 判别器。
3. 训练结束后生成 AIO checkpoint。
4. 用 `scripts/split_checkpoint.py` 将 AIO checkpoint 拆分成边端与云端权重，供部署使用。

这条链路说明：当前工程的“训练目标”是把边端特征映射到 Qwen 视觉中间层，而不是直接训练完整任务回答质量。

### 4.2 边端主流程

`scripts/edge_client.py` 的核心流程是：

1. 加载边端检查点中的学生 CNN、projector 和 bottleneck encoder。
2. 对输入图片做固定尺寸预处理。
3. 提取边端特征并压缩成 transmission tokens。
4. 将特征做 `uint8` 量化并编码成 base64。
5. 通过 HTTP `POST /infer` 发给云端服务。
6. 接收云端返回的回答文本与推理耗时。

这说明现有边端路径已经是“真实传输式”的，而不是仅在单机内传 tensor。

### 4.3 云端主流程

`scripts/cloud_server.py` 的核心流程是：

1. 加载云端检查点中的 bottleneck decoder 与 upsampler。
2. 接收边端发来的 base64 特征、scale、zero point 和 prompt。
3. 反序列化、反量化并恢复 edge tokens。
4. 经 bottleneck decoder 和 upsampler 恢复成目标 Qwen 视觉层 token。
5. 根据 `split_layer` 做特征分布匹配。
6. 延迟加载 Qwen2.5-VL，并从视觉中间层继续完成后续视觉模块和文本生成。
7. 返回回答文本与延迟指标。

### 4.4 单机最小验证流程

`scripts/infer_hybrid.py` 提供的是不经过 HTTP 的最小闭环：

1. 单机加载训练产物。
2. 在同一进程内执行边端编码、云端上采样和 Qwen 续推。
3. 得到最终回答。

这条路径适合做：

1. checkpoint 可用性验证。
2. split layer 行为调试。
3. 不想先处理服务端和网络问题时的最小功能测试。

---

## 5. 关键模块说明

### 5.1 `core/qwen_extractor.py`

这是后续所有与“切分层”“目标特征”“中间层对齐”相关实验的基础模块。  
它支持提取：

1. `layer = -1` 的像素 patch。
2. `layer = 0` 的 patch embedding 输出。
3. `layer = N` 的中间 transformer block 输出。

如果后续要做 selector、answer agreement 评测或必要性分析，必须先明确你面对的是哪一层特征，以及该层 token 的语义是什么。

### 5.2 `models/bottleneck.py`

这是传输预算的最直接控制位。  
后续任何与“预算”“带宽”“压缩强度”相关的实验，第一候选改动点通常都在这里，或者由这里派生。

### 5.3 `models/cloud_upsampler.py`

这是从低分辨率传输 token 恢复到目标 Qwen 视觉 token 的核心模块。  
当前支持多种上采样策略，其中 `TransformerUpsampler` 是更接近当前主干的实现。

### 5.4 `scripts/train_gan.py`

这是当前仓库最核心的训练脚本。  
它同时定义了：

1. 静态特征训练模式与动态特征训练模式。
2. 训练阶段切换。
3. 学生端与云端恢复模块的组装方式。
4. 输出 checkpoint 的字段格式。

如果后续实验需要新增训练日志、预算统计、轻量 selector 监督或新的评测钩子，优先从这里找接口。

### 5.5 `scripts/cloud_server.py` 与 `scripts/edge_client.py`

这两个脚本共同定义了当前最真实的“边云推理协议”。  
任何后续需要验证“真实传输预算”“服务响应时延”“线上插入 selector”的工作，都应优先围绕它们，而不是只改离线训练脚本。

### 5.6 `electron_gui/main.js`

GUI 不定义算法逻辑，它只负责拉起 Python 子进程并展示状态。  
如果后续只是做论文实验，不应优先改 GUI；若需要演示端云流程是否可视化可用，再回到这里。

### 5.7 `core/framework.py`

这是仓库里较早期的通用 split benchmark 框架，用于 FLOPs、参数量和特征大小分析。  
它有参考价值，但不是当前训练与部署主干。后续 agent 可以借鉴它的评测结构，不建议把主要实验直接建立在它上面。

---

## 6. 当前运行入口

### 6.1 训练和数据准备入口

- `python scripts/precompute_qwen_features.py ...`
- `python scripts/train_gan.py ...`
- `python scripts/train_with_upsampler.py ...`
- `python scripts/measure_feature_stats.py ...`
- `python scripts/plot_training.py ...`

### 6.2 部署和推理入口

- `python scripts/split_checkpoint.py ...`
- `python scripts/cloud_server.py ...`
- `python scripts/edge_client.py ...`
- `python scripts/infer_hybrid.py ...`

### 6.3 工程外壳入口

- `cd electron_gui && npm start`
- `python scripts/export_onnx.py ...`
- `cpp_edge_client/src/main.cpp`

---

## 7. 怎样跑通最小闭环

### 7.1 最推荐的最小闭环

对后续实验最推荐的第一条最小闭环不是 GUI，也不是 C++ 客户端，而是：

1. 准备可用 checkpoint。
2. 先用 `scripts/infer_hybrid.py` 在单机内确认“边端编码 -> 云端恢复 -> Qwen 续推”能跑通。
3. 再用 `scripts/split_checkpoint.py` 生成边端/云端权重。
4. 启动 `scripts/cloud_server.py`。
5. 用 `scripts/edge_client.py` 做一次真实 HTTP 推理。

原因很简单：

1. `infer_hybrid.py` 能先排除网络和服务封装问题。
2. `cloud_server.py + edge_client.py` 才代表真实边云实验主路径。
3. GUI 和 C++ 客户端都依赖这条路径，本身不是主干。

### 7.2 最小闭环依赖

至少需要具备：

1. Python 3.10+。
2. `requirements.txt` 中的 PyTorch、timm、Flask、requests 等依赖。
3. Qwen2.5-VL 模型可用，在线下载或本地缓存均可。
4. 一份训练完成的 checkpoint，或已拆分的边端/云端权重。
5. 可用于测试的输入图片。

### 7.3 当前最现实的限制

如果没有现成 checkpoint，本仓库的“从零训练到部署”路径成本很高，因为它依赖：

1. 数据目录准备。
2. Qwen 特征预计算。
3. GPU 资源。
4. Qwen 模型加载条件。

所以后续 agent 在做 bootstrap、评测和 selector 原型时，优先使用已有 checkpoint 或局部最小样本，而不是从零重训。

---

## 8. 数据、依赖与模型说明

### 8.1 Python 依赖

根目录 `requirements.txt` 当前只覆盖了基础 Python 依赖：

1. `torch`
2. `torchvision`
3. `timm`
4. `pandas`
5. `numpy`
6. `matplotlib`
7. `tqdm`
8. `flask`
9. `requests`

需要注意的是，脚本里还实际使用了 `transformers`、`PIL` 等能力，但它们没有明确写进 `requirements.txt`。这是一个已知环境缺口。

### 8.2 数据组织

训练和预计算默认假设数据按 `train/`、`val/` 组织。  
`scripts/precompute_qwen_features.py` 同时支持：

1. ImageFolder 结构。
2. 平铺图片目录，如 COCO 风格。

### 8.3 模型依赖

主干依赖的核心大模型是 `Qwen/Qwen2.5-VL-3B-Instruct`。  
这意味着：

1. 没有 Qwen 权重时，云端推理无法给出完整回答。
2. 所有“答案保持型”实验最终都要回到 Qwen 的续推结果，而不是只看中间层 MSE。

---

## 9. 已有实验能力

当前仓库已经具备这些实验能力：

1. 不同 Qwen 视觉层的中间特征抽取。
2. 静态或动态方式训练 split pipeline。
3. 加入 bottleneck 做低维压缩。
4. 比较不同 split layer 的可恢复性。
5. 测量各层特征分布统计。
6. 将训练产物部署为真实边云服务。

但还不具备这些论文关键能力：

1. answer agreement under budget 的统一统计。
2. 预算口径统一的实验记录。
3. selector 插入后的对照评测。
4. necessity-aware 或 discriminative-aware 的验证脚本。

---

## 10. 实验插入点建议

### 10.1 若后续要加 answer agreement 指标，最可能改哪里

优先修改：

1. `scripts/edge_client.py`
2. `scripts/cloud_server.py`
3. 新增独立评测脚本，复用这两个脚本的输入输出协议

原因：

1. answer agreement 最终要比较的是“同一输入与 prompt 下的回答保持情况”。
2. 现有边云推理链路已经能返回回答文本与延迟。
3. 在这个层面补日志和结果采样，比直接改训练脚本更贴近论文目标。

### 10.2 若后续要加 selector，最可能插入哪里

建议优先考虑两个层面：

1. 训练/离线原型层：`scripts/train_gan.py` 与 `models/`
2. 在线部署层：`scripts/edge_client.py`

更具体地说，selector 最自然的插入位置是在“projector 输出后、量化发送前”。  
原因是这里已经形成 transmission tokens，语义较清晰，也最接近预算控制点。

### 10.3 若后续要加实验日志，最可能放哪里

优先位置：

1. `scripts/train_gan.py`
2. `scripts/edge_client.py`
3. `scripts/cloud_server.py`
4. 新建 `scripts/eval_*.py`

不建议把实验日志主逻辑放在 GUI 中，因为 GUI 只适合展示，不适合承担可复现实验记录。

### 10.4 若后续要做小规模 baseline，最推荐先用哪条路径

最推荐路径：

1. 先基于 `scripts/infer_hybrid.py` 做本地最小验证。
2. 再基于 `scripts/cloud_server.py + scripts/edge_client.py` 做小样本真实链路评测。

不推荐一上来就改 `electron_gui/` 或 `cpp_edge_client/`，因为它们会引入额外工程变量，却不会直接增加论文证据。

---

## 11. 哪些部分适合后续做实验插入

适合优先插实验的部分：

1. `models/bottleneck.py`
   - 适合做预算、压缩强度和传输维度实验。
2. `models/cloud_upsampler.py`
   - 适合做恢复质量、结构替换和上采样策略实验。
3. `scripts/train_gan.py`
   - 适合加训练统计、实验开关、小规模 selector 原型。
4. `scripts/edge_client.py`
   - 适合加 token 选择、预算统计、在线日志。
5. `scripts/cloud_server.py`
   - 适合加响应记录、答案对齐统计和线上评测协议。
6. `core/qwen_extractor.py`
   - 适合加新的目标层或特征抽取逻辑。

---

## 12. 哪些部分不要轻易改

以下部分不建议后续 agent 在没有明确实验理由时随意改动：

1. `core/qwen_extractor.py` 中对 Qwen 中间层的抽取逻辑。
   - 这是整个对齐假设的基础，改错会让训练目标与部署目标失配。
2. `scripts/split_checkpoint.py` 的 checkpoint 字段结构。
   - 边端、云端和 GUI 都依赖这个约定。
3. `scripts/cloud_server.py` 中恢复到 Qwen 视觉链路的主流程。
   - 这是当前真实推理主干，轻率改动会让“能跑通”这一基础失效。
4. `scripts/edge_client.py` 的 payload 协议。
   - 后续若要加字段，应尽量向后兼容，而不是直接破坏现有字段。

---

## 13. 已知问题与坑点

### 13.1 依赖声明不完整

`requirements.txt` 没有覆盖 `transformers` 等运行所需依赖。  
这意味着 README 的环境步骤并不完整，后续实验记录里要补充这一点。

### 13.2 训练指标和论文目标未对齐

当前主干更偏“特征重建”和“可恢复部署”，而不是“答案保持”。  
这会导致训练看起来合理，但论文问题仍未被真正验证。

### 13.3 README 中的实验结果不等于当前标准部署结果

README 已明确提到一组 split-layer ablation 没有打开 bottleneck。  
后续 agent 不应把这组数直接当作完整压缩部署结论。

### 13.4 GUI 默认配置路径偏向本地开发环境

`electron_gui/main.js` 中默认 checkpoint 路径是特定环境的绝对路径示例，不适合作为可移植配置依据。

### 13.5 `core/framework.py` 与当前主干存在代际差异

这个模块更像早期通用分析框架，不应误认为当前训练与部署主路径。

---

## 14. 后续 agent 最小阅读路径

### 14.1 若只做 `EXP-002` 答案保持型评测

建议阅读顺序：

1. `docs/MASTER_EXPERIMENT_PLAN.md`
2. `docs/PROJECT_FRAMEWORK.md`
3. `docs/experiments/EXP-001-bootstrap-baseline/RESULTS.md`
4. `scripts/edge_client.py`
5. `scripts/cloud_server.py`
6. `scripts/infer_hybrid.py`

重点是理解：当前回答是如何产生的、哪里能采集回答、哪里能记录预算。

### 14.2 若只做 `EXP-003` selector MVP

建议阅读顺序：

1. `docs/MASTER_EXPERIMENT_PLAN.md`
2. `docs/PROJECT_FRAMEWORK.md`
3. `scripts/train_gan.py`
4. `models/bottleneck.py`
5. `models/cloud_upsampler.py`
6. `scripts/edge_client.py`

重点是理解 transmission tokens 在什么位置形成，以及 selector 插进去后如何保持部署链路最小改动。

### 14.3 若只做评测或结果统计

建议阅读顺序：

1. `docs/MASTER_EXPERIMENT_PLAN.md`
2. `docs/EXPERIMENT_INDEX.md`
3. `scripts/edge_client.py`
4. `scripts/cloud_server.py`
5. `core/framework.py`

其中 `core/framework.py` 只作为评测结构参考，不作为当前主干评测入口。

### 14.4 若只做 selector 插入点验证

建议最先读：

1. `scripts/train_gan.py`
2. `scripts/edge_client.py`
3. `models/bottleneck.py`
4. `core/qwen_extractor.py`

---

## 15. 对后续 agent 的操作建议

1. 优先复用当前训练和部署主干，不要先重写目录结构。
2. 先补 answer agreement 和预算评测，再谈 selector 是否有效。
3. 小规模实验优先，避免一开始就重训大模型或大规模跑表。
4. 任何新实验都尽量复用现有 checkpoint 字段和 HTTP 协议，减少工程漂移。
5. GUI 和 C++ 客户端暂时不是论文主线，除非实验明确要求真实设备演示，否则先别把精力放在这两处。

---

## 16. 一句话总结

当前 SplitOculo 已经具备“训练 -> 拆权重 -> 边云推理”的最小完整主干，足以支撑后续论文实验；但在论文真正需要的 answer agreement、预算评测和 selector 验证上仍是空白，因此后续最优先的工作不是重构主干，而是补齐评测骨架并在 transmission tokens 位置建立方法插槽。
