# v0.1.0 · Occlusion-Aware ByteTrack

基于 ByteTrack 的面向遮挡多目标跟踪，本科毕业设计的首次公开工程版本。

## 包含内容

- YOLOX-x 检测与 ByteTrack 两阶段关联。
- OSNet ReID、长时遮挡恢复和相机运动补偿。
- 视频演示、MOT17/MOT20 评估配置、模块消融与 ReID 训练入口。
- 中文快速开始、方法说明、数据准备和验证记录。

## 下载与运行

| 附件 | 内容 |
| --- | --- |
| `occlusion-aware-bytetrack-source-v0.1.0.zip` | 完整源码与文档，不含模型和数据集 |
| `bytetrack-occlusion-reid-weights-20261011.zip` | MOT17 微调的 OSNet-x1.0 权重与校验清单 |
| `SHA256SUMS-v0.1.0.txt` | 两个 ZIP 的 SHA-256 校验值 |

检测模型需从 [ByteTrack 上游](https://github.com/ifzhang/ByteTrack#pretrained-models) 单独下载；把 `bytetrack_x_mot17.pth.tar` 和 ReID 包中的 `osnet_x1_0_mot17.pth` 放到 `pretrained/`，然后按 [项目首页](https://github.com/lhy864/occlusion-aware-bytetrack#快速开始) 安装并运行。

## 验证范围

95 个 Python 文件、13 个 JSON 配置通过静态检查，四个命令行入口正常，纯 Python wheel 构建成功；在本地 Python 3.9 / PyTorch 2.7.1 GPU 环境完成三帧完整推理，每帧输出 15 条轨迹。

这不是完整数据集性能复测，干净环境安装、CPU 推理和完整 MOT 评估未在本次发布中验证。源码保留上游 MIT 许可证与署名；模型、数据集及依赖遵循各自条款。论文全文、检测报告、个人视频和数据集不随本版本发布。
