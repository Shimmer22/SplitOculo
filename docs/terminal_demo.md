# SSH 终端 Demo

`scripts/terminal_demo.py` 用于把完整演示放在同一台服务器上运行。它保留
SplitOculo 的 HTTP 边云边界，但服务只监听 `127.0.0.1`，不需要 Electron、
浏览器或公网中转。

## 交互运行（默认）

SSH 登录服务器后直接运行：

```bash
python scripts/terminal_demo.py
```

终端会显示一个轻量菜单：使用 `↑/↓` 移动选框、`Enter` 进入或确认，在项目
列表中用 `Space` 勾选。文件路径、Prompt 和数值参数使用普通文本输入。选择
“运行 Demo”时配置会自动持久化，下次启动直接沿用；默认保存在：

```text
~/.config/splitoculo/terminal_demo.json
```

可通过环境变量 `SPLITOCULO_DEMO_CONFIG` 指定其他配置位置。HTTP 密码不会写入
配置文件。运行时模型加载和客户端日志只占用底部一行并原地更新，每个方案的
结果会按终端宽度横向排列；每张卡片只显示回答、端侧编码耗时、模拟时延和
去网络 TTFT。

推理使用流式接口。当前方案生成时，回答会在终端逐字出现；任一方案完成后，
它会立即加入上方的横排结果栏，不必等待其余方案。云端同时保留原来的
`/infer` 和 `/infer_qwen` 非流式接口，并新增 `/infer_stream` 与
`/infer_qwen_stream`，因此已有调用方不受影响。

“运行设置”中提供关闭、BLE 62.5 KB/s、BLE 125 KB/s、1 MB/s 和自定义带宽
预设。模拟时延根据每个方案的实际请求体大小计算，因此不同方案会显示不同值。

交互会话中 cloud checkpoint 与 Qwen 模型保持常驻。运行结束后按 Enter 返回
菜单再次执行不会重新加载云端服务；只有退出程序，或更改模型、checkpoint、
端口时才会关闭并重新启动本次会话创建的服务。

运行时会先检查指定端口。端口没有服务时，命令自动启动并预热
`scripts/cloud_server.py`；结束或中断时，只关闭自己启动的服务。端口已有兼容
服务时会复用它，并核对 Qwen 与 cloud checkpoint，避免静默混用权重。

## 参数模式与本地 3B 示例

参数模式继续保留，用于脚本化实验或首次直接写入一组明确参数。使用仓库中的
3B 成对权重和时序权重：

```powershell
E:\anaconda\envs\cnn_vit\python.exe scripts\terminal_demo.py `
  --input C:\path\to\short-video.mp4 `
  --cloud-checkpoint checkpoints\qwen_vit_h1280_layer4_224_b64_t256\split_imported\cloud_weights.pth `
  --edge-checkpoint checkpoints\qwen_vit_h1280_layer4_224_b64_t256\split_imported\edge_weights.pth `
  --temporal-pair-checkpoint checkpoints\temporal_pair_ucf101\temporal_pair_best.pth `
  --qwen-path Qwen/Qwen2.5-VL-3B-Instruct `
  --projects baseline,so,temporal,codec `
  --max-frames 4 `
  --sample-fps 2 `
  --offline
```

首次检查建议先使用 `--projects baseline` 和单张图片，再逐步加入端云、时序和
codec 方案。选择 `so` 时需要 edge checkpoint；选择 `temporal` 或 `codec` 时还
需要 temporal-pair checkpoint。

## Linux 服务器示例

服务器使用自身完整的 32B 权重路径，不要复制本地 Windows 默认路径：

```bash
python scripts/terminal_demo.py \
  --input /data/demo/short-video.mp4 \
  --cloud-checkpoint /data/checkpoints/split/cloud_weights.pth \
  --edge-checkpoint /data/checkpoints/split/edge_weights.pth \
  --temporal-pair-checkpoint /data/checkpoints/temporal_pair_best.pth \
  --qwen-path Qwen/Qwen2.5-VL-32B-Instruct \
  --projects baseline,so,temporal,codec \
  --max-frames 8 \
  --sample-fps 2 \
  --offline
```

图像、图像帧目录、H.264/H.265 等压缩视频以及 RGB/BGR/灰度裸帧均由现有
`demo_client.py` 读取。使用 `--help` 可查看带宽、裸帧和 codec 参数。

如果提示 `/health` 缺少字段，说明指定端口运行的是旧版
`cloud_server.py`。先更新服务器仓库并重启旧服务，不要让终端 Demo 自动覆盖
外部进程。
