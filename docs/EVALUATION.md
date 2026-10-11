# 数据集准备与评估

## 目录约定

下载 MOT17 / MOT20 后，将实际数据放在以下目录。目录需要包含序列图片和原有标注文件，不能只放压缩文件或公开检测结果。

```text
datasets/
  mot/
    train/MOT17-02-FRCNN/img1/000001.jpg
    train/MOT17-02-FRCNN/gt/gt.txt
    train/MOT17-02-FRCNN/det/det.txt
    train/MOT17-02-FRCNN/seqinfo.ini
    test/MOT17-01-FRCNN/img1/000001.jpg
    annotations/
  MOT20/
    train/MOT20-01/img1/000001.jpg
    test/MOT20-04/img1/000001.jpg
    annotations/
```

上面只列出部分序列，各数据集需保留完整的对应划分。MOT17 的 DPM、FRCNN、SDP 是相同视频的不同公开检测版本。

## 标注转换

在项目根目录运行：

```bash
python tools/convert_mot17_to_coco.py
python tools/convert_mot20_to_coco.py
```

脚本生成 `annotations/train.json`、`test.json`、`train_half.json` 和 `val_half.json`。半集转换还会写入用于评估的 `gt_train_half.txt`、`gt_val_half.txt` 等文件。训练集转换需要图片、GT 与 `det/det.txt`，测试集不需要公开 GT。

MOT17 转换脚本在训练划分默认只取 FRCNN 序列，避免相同视频重复计数；测试划分保留各检测版本的序列。COCO JSON 决定实际参与评估的序列，比较结果时应使用相同划分。

转换脚本的 `DATA_PATH` 已设为项目相对目录，可将数据放到该位置，也可自行调整脚本中的目录。ReID 数据导出工具另外支持 `--mot-root`。

## MOT17 本地评估

```bash
python tools/track.py --config configs/eval/mot17_train_yolox_x_occlusion_best.json
```

输出位于：

```text
YOLOX_outputs/mot17_train_yolox_x_occlusion_best/
  val_log.txt
  track_results/*.txt
```

该配置评估 MOT17 训练集。检测和 ReID 权重使用过 MOT17 数据，本地训练集结果用于工程检查和误差分析，不能当作未知数据上的独立测试结果。

## 半集消融

现有五组配置使用相同的检测器及基础关联参数，通过开关比较增强模块：

| 配置后缀 | ReID | 长时恢复 | CMC |
| --- | --- | --- | --- |
| `a_base` | 关闭 | 关闭 | 关闭 |
| `b_reid` | 开启 | 关闭 | 关闭 |
| `c_reid_long` | 开启 | 开启 | 关闭 |
| `d_reid_cmc` | 开启 | 关闭 | 开启 |
| `e_final` | 开启 | 开启 | 开启 |

```bash
python tools/track.py --config configs/eval/mot17_half_yolox_x_osnet_x1_0_ablation_a_base.json
python tools/track.py --config configs/eval/mot17_half_yolox_x_osnet_x1_0_ablation_b_reid.json
python tools/track.py --config configs/eval/mot17_half_yolox_x_osnet_x1_0_ablation_c_reid_long.json
python tools/track.py --config configs/eval/mot17_half_yolox_x_osnet_x1_0_ablation_d_reid_cmc.json
python tools/track.py --config configs/eval/mot17_half_yolox_x_osnet_x1_0_ablation_e_final.json
```

这组配置保留了 `legacy_seq_tuning=true`，评估器会按序列调整部分跟踪阈值和保留长度。复现实验时应同时记录该设置、检测分辨率、权重校验值、数据划分和硬件，不能把 JSON 中的默认阈值视为所有序列实际使用的阈值。

严格的半集验证还需要核对检测器和 ReID 的训练范围，保证验证帧未被用于训练；已有完整 MOT17 检测权重不能据此视为严格的半集训练模型。

## 测试集结果打包

```bash
python tools/track.py --config configs/eval/mot17_test_yolox_x_submission.json
python tools/track.py --config configs/eval/mot20_test_yolox_x_final_submission.json
```

对应生成：

```text
YOLOX_outputs/mot17_test_yolox_x_submission/mot17_test_yolox_x_submission.zip
YOLOX_outputs/mot20_test_yolox_x_final_submission/mot20_test_yolox_x_final_submission.zip
```

这些是评估结果压缩包，与 GitHub 源码发布包不同。结果包根目录直接放各序列的 `.txt` 文件，行格式为：

```text
frame,id,left,top,width,height,score,-1,-1,-1
```

测试配置通过 `skip_metrics=true` 跳过本地 GT 评估，并通过 `make_submission_zip=true` 生成结果包。测试集没有公开 GT，本地生成结果文件后应按照 MOTChallenge 当前评估入口的要求提交；服务器返回的指标需另行保存。

MOT20 终版配置使用 MOT20 检测权重和 MOT17 微调的 ReID 权重，分别需要 `bytetrack_x_mot20.tar` 与 `osnet_x1_0_mot17.pth`。

## 渲染已有结果

```bash
python tools/render_mot17_sequence_video.py --experiment-name mot17_train_yolox_x_occlusion_best --sequence MOT17-09-FRCNN --ann-json datasets/mot/annotations/train.json --dataset-root datasets/mot/train --start-frame 1
```

完整训练序列使用 `train.json`；使用 `val_half.json` 时，帧编号对应半集划分。
