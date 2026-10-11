# GitHub 发布说明

项目仓库：[lhy864/occlusion-aware-bytetrack](https://github.com/lhy864/occlusion-aware-bytetrack)。

建议仓库简介：`Pedestrian multi-object tracking based on ByteTrack, with OSNet ReID, long-occlusion recovery and camera motion compensation.`

建议主题：`multi-object-tracking`、`bytetrack`、`yolox`、`person-reidentification`、`computer-vision`、`graduation-project`。

## 发布源码

1. 解压 `bytetrack-occlusion-github-20261011.zip`，进入包含 `README.md`、`LICENSE` 和 `yolox/` 的目录。
2. 在 GitHub 创建空仓库，将该目录的内容提交到仓库根目录，避免在仓库内再套一层项目目录。
3. 保留 `LICENSE`、`THIRD_PARTY_NOTICES.md` 和源码中现有署名。
4. 权重、数据集和个人视频继续由 `.gitignore` 排除，运行产物不提交。

以下命令适用于将源码导入新的空仓库；已有仓库应先克隆，避免覆盖远程提交：

```bash
git init
git add .
git status --short
git commit -m "Initial public release of occlusion-aware ByteTrack project"
git branch -M main
git remote add origin https://github.com/lhy864/occlusion-aware-bytetrack.git
git push -u origin main
```

没有安装 Git 时，也可以通过 GitHub 的文件上传入口提交解压后的文件。源码 ZIP 适合作为下载附件，仓库根目录仍应包含实际源码文件，才能展示 README 和方便别人运行。

## 发布 ReID 权重

ReID 补充包 `bytetrack-occlusion-reid-weights-20261011.zip` 可放在仓库的 Release 附件中，使用说明里提供下载入口。权重不加入普通源码提交。

解压后将 `pretrained/osnet_x1_0_mot17.pth` 放到源码目录的 `pretrained/` 内，再从 ByteTrack 上游获取检测权重。该补充包不是完整模型组合，不包含约 0.8 GB 的 YOLOX-x 检测权重。

## 本次整理范围

- 保留当前毕业设计的核心跟踪实现、YOLOX 模型和终版配置。
- 新增中文 README、方法说明、数据准备、训练与发布说明。
- 将本机数据盘路径改为项目相对路径。
- 通过标准 COCO API 替代机器相关的快速 COCO C++ 扩展。
- 移除非主线比较跟踪器和相关入口，保留默认 ByteTrack 评估过程。
- 源码包不包含论文全文、个人信息材料、检测报告、日志、数据集、模型文件和个人视频。

发布包中的 `RELEASE_MANIFEST.json` 记录文件和 SHA-256，用于检查传输完整性。验证记录位于 `docs/VALIDATION.md`，ReID 补充包另有权重清单，源码与权重附件放在 [v0.1.0 Release](https://github.com/lhy864/occlusion-aware-bytetrack/releases/tag/v0.1.0) 中，账户凭据不随项目分发。
