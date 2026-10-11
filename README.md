<div align="center">

# Occlusion-Aware ByteTrack

**基于 ByteTrack 的面向遮挡多目标跟踪**

YOLOX-x 检测 · ByteTrack 关联 · OSNet 身份辅助 · 长时遮挡恢复 · 相机运动补偿

[![Python](https://img.shields.io/badge/Python-3.9-3776AB?logo=python&logoColor=white)](#快速开始)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1-EE4C2C?logo=pytorch&logoColor=white)](#快速开始)
[![License](https://img.shields.io/badge/License-MIT-238636)](LICENSE)
[![Weights](https://img.shields.io/badge/ReID-Weights-0F766E)](https://github.com/lhy864/occlusion-aware-bytetrack/releases/tag/v0.1.0)

[演示效果](#演示效果) · [快速开始](#快速开始) · [方法概览](#方法概览) · [实验评估](#实验评估) · [阅读导航](#阅读导航)

</div>

---

这是一个本科毕业设计的工程整理版，基于 [ByteTrack](https://github.com/ifzhang/ByteTrack) 扩展，用于检测视频中的行人，并在连续帧中分配和保持目标 ID。项目重点处理行人交叉、目标短暂消失及运动相机引起的关联困难。

**输入：** 视频或 MOT 序列。**输出：** 带目标框与 ID 的视频，以及 MOT 格式轨迹文件。

## 演示效果

![近距离交叉与遮挡场景的跟踪结果](assets/occlusion_success.png)

*毕业设计保存的定性案例：连续画面展示目标交叉时的 ID 保持情况，原始画面来自 MOT17。*

| 场景中的困难 | 本项目的处理 |
| --- | --- |
| 两名行人距离很近，检测框发生重叠 | 使用 OSNet 外观特征辅助关联 |
| 行人被遮挡，一段时间后重新出现 | 保留丢失轨迹，通过外观与运动条件尝试恢复原 ID |
| 相机移动，预测位置出现整体偏移 | 使用光流与仿射估计补偿轨迹位置 |

严重遮挡和相似衣着下仍可能出现身份切换。成功与失败案例见 [方法说明](docs/METHOD.md)。

## 快速开始

### 1. 获取代码与安装环境

本地验证环境：Windows、Python 3.9.25、PyTorch 2.7.1+cu128。MOT 批量评估需要 NVIDIA GPU，CPU 演示可使用单独配置。

```bash
git clone https://github.com/lhy864/occlusion-aware-bytetrack.git
cd occlusion-aware-bytetrack

conda create -n bytetrack-occlusion python=3.9 -y
conda activate bytetrack-occlusion
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

<details>
<summary>CPU 环境与安装注意事项</summary>

CPU 环境可将 PyTorch 命令中的 `cu128` 改为 `cpu`。GPU 环境的 CUDA 版本应与显卡、驱动匹配，其他环境组合尚未验证。

公开版本使用标准 `pycocotools` 进行检测 AP 评估，不需要编译原版 YOLOX 的快速 COCO C++ 扩展。`cython-bbox` 如果需要从源码构建，Windows 需安装 C++ Build Tools，Linux 需安装 C/C++ 编译工具。

</details>

### 2. 放置两个模型文件

| 模型文件 | 用途 | 下载入口 |
| --- | --- | --- |
| `bytetrack_x_mot17.pth.tar` | YOLOX-x 行人检测 | [ByteTrack 上游模型库](https://github.com/ifzhang/ByteTrack#pretrained-models) |
| `osnet_x1_0_mot17.pth` | OSNet-x1.0 外观特征 | [本项目 ReID 权重](https://github.com/lhy864/occlusion-aware-bytetrack/releases/download/v0.1.0/bytetrack-occlusion-reid-weights-20261011.zip) |

将检测权重和 ReID 补充包中的 `.pth` 文件放到：

```text
pretrained/
├── bytetrack_x_mot17.pth.tar
└── osnet_x1_0_mot17.pth
```

源码仓库不包含模型权重。MOT20 配置另外需要专用检测权重，来源与文件名见 [权重说明](pretrained/README.md)；ReID 也可以 [自行微调](docs/REID_TRAINING.md)。

### 3. 跟踪自己的视频

将视频放到 `videos/input.mp4`，运行：

```bash
python tools/demo_track.py --config configs/demo/mytest_yolox_x_occlusion_best.json --path videos/input.mp4
```

使用其他视频时，只需替换 `--path` 后的文件路径。

<details>
<summary>CPU 演示命令</summary>

```bash
python tools/demo_track.py --config configs/demo/video_cpu.json --path videos/input.mp4
```

CPU 配置关闭半精度，YOLOX-x 的运行速度会较慢。

</details>

### 4. 查看结果

```text
YOLOX_outputs/mytest_yolox_x_occlusion_best/track_vis/
├── <运行时间>/input.mp4    # 带检测框与目标 ID 的视频
└── <运行时间>.txt          # MOT 格式轨迹
```

## 方法概览

```text
视频帧 → YOLOX-x 检测 → 高 / 低分候选框
                              ↓
       运动预测 + CMC → ByteTrack 两阶段关联 ← OSNet 外观特征
                              ↓
                      长期丢失轨迹恢复
                              ↓
                       更新轨迹并输出 ID
```

| 模块 | 作用 | 代码位置 |
| --- | --- | --- |
| ByteTrack | 高分关联、低分补充关联与轨迹状态管理 | `yolox/tracker/byte_tracker.py` |
| ReID | 裁剪行人并提取归一化外观特征 | `yolox/tracker/reid_extractor.py` |
| 长时恢复 | 对长期丢失轨迹使用更严格的外观与运动条件 | `byte_tracker.py`、`matching.py` |
| CMC | 通过 LK 光流与 RANSAC 估计相机运动 | `yolox/tracker/gmc.py` |

<details>
<summary>展开完整系统结构图</summary>

![YOLOX-x、ByteTrack、ReID、长时恢复与 CMC 的系统结构](assets/architecture.png)

</details>

终版默认配置启用 ReID、长时恢复和 CMC。源码保留了 gallery、时序匹配等早期实验开关，这些开关未包含在默认方案中。参数含义与实现对应关系见 [方法说明](docs/METHOD.md)。

## 实验评估

数据集需从 [MOTChallenge](https://motchallenge.net/) 自行获取，按 [评估说明](docs/EVALUATION.md) 放置数据并生成标注 JSON。

| 任务 | 对应配置 |
| --- | --- |
| MOT17 训练集本地评估 | `configs/eval/mot17_train_yolox_x_occlusion_best.json` |
| MOT17 测试集结果生成 | `configs/eval/mot17_test_yolox_x_submission.json` |
| MOT20 测试集结果生成 | `configs/eval/mot20_test_yolox_x_final_submission.json` |
| 五组模块消融 | `configs/eval/mot17_half_yolox_x_osnet_x1_0_ablation_*.json` |

```bash
python tools/track.py --config configs/eval/mot17_test_yolox_x_submission.json
```

测试配置生成各序列的 MOT 结果和提交 ZIP，本地不计算隐藏 GT 的官方指标。训练集结果用于工程检查与误差分析，不能代替独立测试集结果；HOTA 需通过相应官方评估或 TrackEval 获取。

### 本次发布验证

| 检查 | 结果 |
| --- | --- |
| Python 语法与 JSON 配置 | 95 个 Python 文件、13 个配置通过检查 |
| 主要命令行入口 | 演示、评估、ReID 数据导出、ReID 训练入口正常 |
| 模型与跟踪链路 | 三帧 GPU 推理成功，每帧输出 15 条轨迹 |
| 安装包构建 | 纯 Python wheel 构建成功 |

以上是运行检查，不是完整数据集性能测试。环境与验证范围见 [验证记录](docs/VALIDATION.md)。

## 阅读导航

| 想做什么 | 从哪里开始 |
| --- | --- |
| 理解算法与调整参数 | [方法说明](docs/METHOD.md) |
| 准备数据、评估、运行消融 | [评估说明](docs/EVALUATION.md) |
| 训练自己的 OSNet 权重 | [ReID 微调](docs/REID_TRAINING.md) |
| 下载模型与源码附件 | [v0.1.0 Release](https://github.com/lhy864/occlusion-aware-bytetrack/releases/tag/v0.1.0) |
| 检查源码完整性与基本推理 | `python scripts/verify_release.py` |

<details>
<summary>展开项目目录</summary>

```text
occlusion-aware-bytetrack/
├── configs/       演示、评估、消融与训练配置
├── exps/          YOLOX 实验定义
├── yolox/         检测、跟踪、外观与评估实现
├── tools/         推理、数据转换与训练入口
├── scripts/       发布包检查与短视频推理验证
├── assets/        系统结构与定性案例
├── docs/          方法、训练和评估说明
├── pretrained/    模型放置目录
├── datasets/      数据集放置目录
└── videos/        视频输入目录
```

</details>

## 来源与许可

本项目在 ByteTrack 上扩展，ByteTrack、YOLOX 和 OSNet 的原始工作属于各自作者。保留上游 [MIT 许可证](LICENSE) 和源码署名，具体来源见 [第三方说明](THIRD_PARTY_NOTICES.md)。模型、数据集与可选依赖遵循各自的分发条款。

源码仓库只发布工程代码和项目文档，论文全文、检测报告、个人视频与数据集不随源码分发。

<details>
<summary>ByteTrack 原始论文引用</summary>

```bibtex
@inproceedings{zhang2022bytetrack,
  title={ByteTrack: Multi-Object Tracking by Associating Every Detection Box},
  author={Zhang, Yifu and Sun, Peize and Jiang, Yi and Yu, Dongdong and Weng, Fucheng and Yuan, Zehuan and Luo, Ping and Liu, Wenyu and Wang, Xinggang},
  booktitle={European Conference on Computer Vision},
  year={2022}
}
```

</details>
