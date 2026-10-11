# 权重放置说明

源码包不包含权重，文件名必须与配置一致。

| 文件 | 用途 | 获取方式 |
| --- | --- | --- |
| `bytetrack_x_mot17.pth.tar` | MOT17 与视频演示的 YOLOX-x 检测器 | ByteTrack 上游模型库 |
| `osnet_x1_0_mot17.pth` | 终版 OSNet-x1.0 外观特征 | 本项目 ReID 补充包或自行微调 |
| `bytetrack_x_mot20.tar` | MOT20 测试配置的检测器 | ByteTrack 上游模型库，保存为配置要求的名称 |
| `osnet_x1_0_msmt17_combineall.pth` | ReID 微调初始化 | Torchreid 模型库 |

ByteTrack 上游入口：https://github.com/ifzhang/ByteTrack#pretrained-models

Torchreid 模型入口：https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO

ReID 微调过程见 [训练说明](../docs/REID_TRAINING.md)。补充包 `WEIGHTS_MANIFEST.json` 记录本地微调权重的大小与 SHA-256，下载后可据此核对。

上游 MOT17 检测器属于 ByteTrack 已发布模型，不将其重新描述为本项目从零训练的模型。MOT20 文件名与上游下载名可能不同，需要按配置重命名。
