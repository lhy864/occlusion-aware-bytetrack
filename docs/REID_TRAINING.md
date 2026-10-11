# OSNet-x1.0 微调

本项目使用 `torchreid==0.2.5` 的 `torchreid.reid.models` 包布局，OSNet 模型来自 Torchreid / deep-person-reid。源码包含数据导出与训练入口，训练结果保存为推理可直接加载的权重。

## 导出 MOT17 行人裁剪图

MOT17 数据应具有 `train/<sequence>/img1`、`gt/gt.txt` 和 `seqinfo.ini`。

```bash
python tools/build_mot17_reid_dataset.py --config configs/reid/build_mot17_reid_dataset.json
```

也可以显式指定数据位置：

```bash
python tools/build_mot17_reid_dataset.py --config configs/reid/build_mot17_reid_dataset.json --mot-root datasets/mot
```

默认仅导出 FRCNN 序列，并按前后半段划分训练和验证裁剪，避免相同视频重复导出。结果保存在 `datasets/reid_mot17`，同时包含身份标签和 `metadata.jsonl`。

## 准备初始化模型

初始化权重为 `pretrained/osnet_x1_0_msmt17_combineall.pth`，来源见 [Torchreid 模型库](https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO)。当前特征加载工具支持在该文件缺失时通过记录的上游地址下载，离线使用应提前放好权重。

初始化模型与本项目微调后的 `osnet_x1_0_mot17.pth` 是两个不同文件，不能只改文件名就视为微调完成。

## 启动训练

```bash
python tools/train_mot17_reid.py --config configs/reid/train_osnet_x1_0_mot17.json
```

配置使用 256×128 输入、40 个 epoch、每批 32 张图片和每身份 4 个样本。主要训练参数以 JSON 为准，训练记录保存在 `outputs/osnet_x1_0_mot17/`，导出的推理权重为：

```text
pretrained/osnet_x1_0_mot17.pth
```

数据加载在当前机器上存在多进程问题时，可以添加 `--num-workers 0`。不做验证的配置为 `train_osnet_x1_0_mot17_no_val.json`；该开关仅跳过训练过程的验证，不改变已经导出的数据划分。

自行微调的结果受初始化、数据划分、依赖版本及硬件影响，应保留日志和权重校验值。ReID 补充包中的权重是本地已有的微调结果，本次源码整理没有重新训练模型。
