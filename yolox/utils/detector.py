#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
检测器适配层。

这个文件的作用不是实现完整检测网络，而是把不同检测器的输出
统一成 ByteTrack 能直接消费的格式：
    [x1, y1, x2, y2, obj_score, cls_score, cls_id]

当前项目里真正重要的是：
1. 保留原始 YOLOX 推理链路，兼容原论文代码。
2. 新增 UltralyticsPredictor，让 Ultralytics YOLO 系列
   （YOLOv8 / YOLO11 / YOLO26 等）的输出都能无缝喂给 BYTETracker。
"""

import os.path as osp
from pathlib import Path

import cv2
import torch
from loguru import logger

from yolox.data.data_augment import preproc
from yolox.utils.boxes import postprocess


def _load_image(img):
    """
    统一读取图像输入。

    参数可能是两种形式：
    1. 字符串/Path：表示图片或视频帧文件路径，需要用 cv2.imread 读取。
    2. numpy.ndarray：表示 demo 中已经从视频流读出的原始帧。

    返回的 img_info 会被后续 predictor 和 tracker 共同使用，其中：
    - raw_img：原始 OpenCV BGR 图像，ReID 裁剪和可视化都需要它。
    - height/width：原图尺寸，tracker 会用它把检测框坐标映射回原图。
    """
    # demo/eval 入口既可能传文件路径，也可能直接传 OpenCV 读出的帧。
    if isinstance(img, (str, Path)):
        img_path = str(img)
        raw_img = cv2.imread(img_path)
        if raw_img is None:
            raise FileNotFoundError("failed to load image: {}".format(img_path))
        file_name = osp.basename(img_path)
    else:
        raw_img = img
        file_name = None

    height, width = raw_img.shape[:2]
    return {
        "id": 0,
        "file_name": file_name,
        "height": height,
        "width": width,
        "raw_img": raw_img,
    }


def _normalize_device(device):
    # Ultralytics 的 device 参数和 torch.device 的表示方式不完全一样，
    # 这里做一次统一，避免 "cuda:0" / 0 / "cpu" 混用时出错。
    if isinstance(device, torch.device):
        if device.type == "cuda":
            return device.index if device.index is not None else 0
        return "cpu"
    return device


def _extract_checkpoint_state_dict(checkpoint):
    """
    统一取出 checkpoint 里的模型权重字典。
    兼容两种常见格式：
    1. {"model": state_dict, ...}
    2. 直接就是 state_dict
    """
    if isinstance(checkpoint, dict) and "model" in checkpoint and isinstance(checkpoint["model"], dict):
        return checkpoint["model"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError("unsupported checkpoint format: {}".format(type(checkpoint)))


def load_yolox_detector_weights(model, ckpt_file, class_ids=None, map_location="cpu"):
    """
    加载 YOLOX 检测器权重，并兼容“80 类 COCO 预训练 -> 1 类行人检测头”的迁移。

    当前场景里，`pretrained/yolox_x.pth` 是 COCO 预训练权重，cls head 输出维度通常是 80；
    而 MOT 评估/演示用的 exp 一般把 `num_classes` 设成 1。这里做的事是：
    - 形状完全一致的层：正常加载
    - 如果发现当前模型是 1 类，但 checkpoint 的 `cls_preds.*` 是多类：
      取 `class_ids[0]` 对应那一类的分类头参数，映射到当前的单类 head

    对 COCO 来说，person 的类别 id 就是 0，所以默认能正确迁移出“人”这一类。
    """
    checkpoint = torch.load(ckpt_file, map_location=map_location)
    state_dict = _extract_checkpoint_state_dict(checkpoint)
    model_state = model.state_dict()
    compatible_state = {}
    loaded_keys = []
    remapped_keys = []
    skipped_keys = []
    selected_class_id = int(class_ids[0]) if class_ids else 0

    for raw_key, value in state_dict.items():
        key = raw_key[7:] if raw_key.startswith("module.") else raw_key
        target_value = model_state.get(key)
        if target_value is None:
            skipped_keys.append((raw_key, tuple(value.shape), "missing_in_model"))
            continue

        if target_value.shape == value.shape:
            compatible_state[key] = value
            loaded_keys.append(key)
            continue

        is_single_class_target = target_value.ndim >= 1 and target_value.shape[0] == 1
        is_multi_class_source = value.ndim >= 1 and value.shape[0] > 1
        same_tail_shape = target_value.shape[1:] == value.shape[1:]
        is_cls_head = ".cls_preds." in key

        if is_cls_head and is_single_class_target and is_multi_class_source and same_tail_shape:
            if selected_class_id >= value.shape[0]:
                skipped_keys.append((raw_key, tuple(value.shape), "class_id_out_of_range"))
                continue
            compatible_state[key] = value[selected_class_id : selected_class_id + 1].clone()
            remapped_keys.append((key, selected_class_id, tuple(value.shape), tuple(target_value.shape)))
            continue

        skipped_keys.append((raw_key, tuple(value.shape), tuple(target_value.shape)))

    missing_keys, unexpected_keys = model.load_state_dict(compatible_state, strict=False)
    logger.info(
        "Loaded YOLOX checkpoint {} with {} direct keys, {} remapped cls-head keys, {} skipped keys (missing {} / unexpected {} during load).",
        ckpt_file,
        len(loaded_keys),
        len(remapped_keys),
        len(skipped_keys),
        len(missing_keys),
        len(unexpected_keys),
    )
    if remapped_keys:
        logger.info(
            "Remapped YOLOX classification heads from source class {} into current single-class detector: {}",
            selected_class_id,
            ", ".join(key for key, *_ in remapped_keys),
        )
    return {
        "loaded_keys": loaded_keys,
        "remapped_keys": remapped_keys,
        "skipped_keys": skipped_keys,
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
    }


class YOLOXPredictor:
    """
    原始 YOLOX 检测器封装。

    这个类基本保留 ByteTrack 官方仓库的推理流程，用于兼容原论文复现。
    当前毕业设计的主要运行路径是 UltralyticsPredictor，但保留它可以方便后续做对比实验。
    """

    def __init__(self, model, exp, trt_file=None, decoder=None, device=torch.device("cpu"), fp16=False):
        # 这一支保留原始 ByteTrack 的 YOLOX 检测路径，主要服务于兼容原项目。
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))
            x = torch.ones((1, 3, exp.test_size[0], exp.test_size[1]), device=device)
            self.model(x)
            self.model = model_trt

    def inference(self, img, timer=None):
        """
        执行一次 YOLOX 推理。

        输出 outputs 的格式与原始 ByteTrack 一致，后续会直接送入 BYTETracker.update()。
        """
        # YOLOX 这条链路仍然沿用原仓库的 preproc + postprocess。
        img_info = _load_image(img)
        proc_img, ratio = preproc(img_info["raw_img"], self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio

        proc_img = torch.from_numpy(proc_img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            proc_img = proc_img.half()

        with torch.no_grad():
            if timer is not None:
                timer.tic()
            outputs = self.model(proc_img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)

        return outputs, img_info


class UltralyticsPredictor:
    """
    Ultralytics YOLO 检测器封装。

    本项目当前默认使用 pretrained/yolo26x.pt 替代原始 YOLOX 检测器。
    由于 Ultralytics YOLO 系列的输出结构与 YOLOX 不同，所以这里专门做格式适配。
    """

    def __init__(
        self,
        model_path,
        test_size,
        confthre,
        nmsthre,
        device=torch.device("cpu"),
        fp16=False,
        classes=None,
    ):
        # 这一支是本项目的核心改动之一：直接加载 Ultralytics 预训练权重做检测，
        # 从而绕过原项目里代价很高的 YOLOX 检测器训练。
        # 如果本地路径不存在，Ultralytics 会自动把官方权重下载到该位置。
        Path(model_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        from ultralytics import YOLO

        self.model = YOLO(model_path, verbose=False)
        self.test_size = test_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.classes = classes
        self.device = _normalize_device(device)
        self.fp16 = bool(fp16 and self.device != "cpu")

    def _to_bytetrack_format(self, result, ratio):
        """
        把 Ultralytics YOLO 系列的单帧输出转换成 ByteTrack 标准输入。

        Ultralytics 检测输出：
        - boxes.xyxy：原图坐标下的 [x1, y1, x2, y2]
        - boxes.conf：检测置信度
        - boxes.cls：类别 id

        ByteTrack 期望：
        - [x1, y1, x2, y2, obj_score, cls_score, cls_id]

        这里把 cls_score 置为 1，是因为 Ultralytics 的 conf 已经综合了目标和类别置信度；
        这样 tracker 里 scores = output[:,4] * output[:,5] 时不会再额外缩小分数。
        """
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return None

        xyxy = boxes.xyxy.detach().cpu().float()
        scores = boxes.conf.detach().cpu().float()
        classes = boxes.cls.detach().cpu().float()

        # ByteTrack 的 update() 会按照 test_size 和原图尺寸自己做一次反缩放，
        # 所以这里要先把 Ultralytics 的原图坐标重新投影到 "缩放后图像坐标系"。
        xyxy *= ratio
        cls_scores = torch.ones_like(scores)
        return torch.cat(
            (
                xyxy,
                scores.unsqueeze(1),
                cls_scores.unsqueeze(1),
                classes.unsqueeze(1),
            ),
            dim=1,
        )

    def inference(self, img, timer=None):
        """
        执行一次 Ultralytics YOLO 推理。

        返回值保持和 YOLOX predictor 一样的形式：
        - outputs：长度为 1 的 list，其中 outputs[0] 是当前帧检测框张量。
        - img_info：包含原图、原图尺寸和缩放比例。
        """
        # 和 YOLOX 不同，Ultralytics 内部自己完成了预处理和 NMS；
        # 我们只需要把最终 boxes 取出来并转成 ByteTrack 需要的格式。
        img_info = _load_image(img)
        ratio = min(
            self.test_size[0] / float(img_info["height"]),
            self.test_size[1] / float(img_info["width"]),
        )
        img_info["ratio"] = ratio

        with torch.no_grad():
            if timer is not None:
                timer.tic()
            results = self.model.predict(
                source=img_info["raw_img"],
                stream=False,
                verbose=False,
                conf=self.confthre,
                iou=self.nmsthre,
                imgsz=self.test_size,
                device=self.device,
                classes=self.classes,
                half=self.fp16,
            )

        return [self._to_bytetrack_format(results[0], ratio)], img_info


def build_detector(args, exp, device, model=None, trt_file=None, decoder=None):
    """
    根据 args.detector 构建检测器。

    这个函数是 demo_track.py 和 track.py 的共同入口：
    - args.detector == "ultralytics" / "yolov8"：不需要 YOLOX model，直接加载 args.yolo_model。
    - args.detector == "yolox"：需要外部先构建好 YOLOX model，再传进来。
    """
    # 统一的检测器工厂：
    # - detector=ultralytics / yolov8 时直接走 Ultralytics
    # - detector=yolox 时保留原始仓库的模型构建方式
    if args.detector in {"ultralytics", "yolov8"}:
        confthre = args.conf if args.conf is not None else exp.test_conf
        nmsthre = args.nms if args.nms is not None else exp.nmsthre
        return UltralyticsPredictor(
            model_path=args.yolo_model,
            test_size=exp.test_size,
            confthre=confthre,
            nmsthre=nmsthre,
            device=device,
            fp16=getattr(args, "fp16", False),
            classes=getattr(args, "classes", [0]),
        )

    if model is None:
        raise ValueError("YOLOX detector requires a loaded model.")

    return YOLOXPredictor(
        model=model,
        exp=exp,
        trt_file=trt_file,
        decoder=decoder,
        device=device,
        fp16=getattr(args, "fp16", False),
    )


# 兼容旧命名，避免其他文件或旧笔记里引用 YOLOv8Predictor 时立刻失效。
YOLOv8Predictor = UltralyticsPredictor
