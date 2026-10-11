# 第三方来源与署名

本项目基于现有开源代码扩展，以下上游工作不属于本毕业设计的原创贡献。原有源码中的作者和版权注释继续保留。

| 来源 | 在本项目中的用途 | 链接 |
| --- | --- | --- |
| ByteTrack | 两阶段关联、轨迹状态管理、基础工程 | https://github.com/ifzhang/ByteTrack |
| YOLOX | 检测模型、训练框架、数据加载和实验定义 | https://github.com/Megvii-BaseDetection/YOLOX |
| Torchreid / OSNet | 行人外观网络、预训练模型和训练损失 | https://github.com/KaiyangZhou/deep-person-reid |
| torchreid-pip | 本地 `torchreid==0.2.5` 使用的包布局 | https://github.com/goksenin-uav/torchreid-pip |
| pycocotools | 数据集标注读取、标准 COCO 检测评估 | https://github.com/cocodataset/cocoapi |
| motmetrics | CLEAR MOT 与身份指标评估 | https://github.com/cheind/py-motmetrics |
| OpenCV | 图像处理、光流、仿射估计和视频读写 | https://github.com/opencv/opencv |

ByteTrack 上游 README 还致谢了 FairMOT、TransTrack 和 JDE-Cpp 等项目；跟踪基础模块的代码来源随上游一并保留，相关文件中的现有注释没有被替换成本项目作者署名。

根目录 `LICENSE` 保留 ByteTrack 的 MIT 许可文本及 `Copyright (c) 2021 Yifu Zhang`。外部依赖、预训练模型和数据集的条款由各自发布方提供，不将根目录许可证解释为对这些外部资产的重新授权。

发布源码中没有打包 SORT、DeepSORT、MOTDT、第三方教程工程，检测 AP 评估使用依赖包提供的标准 COCO API，没有分发原版快速 COCO 评估器的 C++ 代码或本机编译产物。

可选 Ultralytics 后端通过 `requirements-optional.txt` 单独安装，其相关代码、模型和许可说明见其上游项目。默认毕业设计流程不依赖该后端。

系统结构图为毕业设计中整理的结构示意，跟踪案例为 MOT17 序列的结果展示，原始画面来源于 MOTChallenge 数据集；图片不替代数据集本身的获取与使用条款。
