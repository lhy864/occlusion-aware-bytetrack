#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
演示入口脚本。

这个脚本适合回答三个问题：
1. 输入一段视频后，系统完整推理链路怎么跑？
2. 现在到底是 YOLOX 在检测，还是 Ultralytics YOLO 在检测？
3. tracker 输出的可视化视频和 debug 日志会保存到哪里？
"""

import argparse
import json
import os
import os.path as osp
import sys
import time
from pathlib import Path

import cv2
import torch
from loguru import logger

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from yolox.exp import get_exp
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer
from yolox.utils import fuse_model, get_model_info
from yolox.utils.cli_config import apply_config_defaults
from yolox.utils.detector import build_detector, load_yolox_detector_weights
from yolox.utils.visualize import plot_tracking

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    """
    构建 demo 运行参数。

    这里的参数可以分成几类理解：
    - 输入输出类：demo/path/save_result/save_debug
    - 检测器类：detector/yolo_model/conf/nms/tsize
    - ByteTrack 基础类：track_thresh/track_buffer/match_thresh
    - 增强模块类：with_reid/with_gallery/with_temporal_matcher/with_anti_swap
    - 调试类：debug_association/debug_topk

    本项目支持 --config，所以大多数参数实际来自 configs/demo/*.json。
    """
    # 这个 parser 基本就是整套演示系统的“功能地图”。
    # 你以后想知道某个增强模块有没有接进 demo，先看这里最直观。
    parser = argparse.ArgumentParser("ByteTrack Demo")
    parser.add_argument("demo", nargs="?", default="image", help="demo type, eg. image, video and webcam")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("--path", default="./videos/input.mp4", help="path to images or video")
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument("--save_result", action="store_true", help="save the tracking result")
    parser.add_argument("--save-debug", dest="save_debug", default=False, action="store_true", help="save per-frame association debug records as jsonl")
    parser.add_argument("--debug-association", dest="debug_association", default=False, action="store_true", help="collect first-stage association debug information inside the tracker")
    parser.add_argument("--debug-ambiguous-only", dest="debug_ambiguous_only", default=False, action="store_true", help="only dump frames where ambiguous first-stage matches were detected")
    parser.add_argument("--debug-topk", default=3, type=int, help="top-k candidate detections exported for each track in debug jsonl")
    parser.add_argument("-f", "--exp_file", default=None, type=str, help="experiment description file")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="YOLOX checkpoint for eval")
    parser.add_argument(
        "--detector",
        default="yolox",
        choices=["ultralytics", "yolov8", "yolox"],
        help="detector backend used before ByteTrack",
    )
    parser.add_argument(
        "--yolo-model",
        default="pretrained/yolov8x.pt",
        type=str,
        help="optional Ultralytics model path, e.g. pretrained/yolov8x.pt",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        type=int,
        default=[0],
        help="class ids kept for Ultralytics YOLO, default is person=0",
    )
    parser.add_argument("--with-reid", dest="with_reid", default=False, action="store_true", help="enable appearance ReID for long-occlusion recovery")
    parser.add_argument("--reid-backbone", default="auto", type=str, help="Torchreid ReID backbone name, for example auto, osnet_x0_25, osnet_x0_75, osnet_x1_0, osnet_ain_x1_0 or osnet_ibn_x1_0")
    parser.add_argument("--reid-model", default="pretrained/osnet_x0_25_mot17.pth", type=str, help="ReID weight path")
    parser.add_argument("--reid-input-height", default=256, type=int, help="ReID crop height")
    parser.add_argument("--reid-input-width", default=128, type=int, help="ReID crop width")
    parser.add_argument("--appearance-thresh", default=0.25, type=float, help="appearance distance threshold for ReID matching")
    parser.add_argument("--reid-weight", default=0.5, type=float, help="weight of appearance cost in first-stage association")
    parser.add_argument("--reid-batch-size", default=32, type=int, help="batch size for ReID feature extraction")
    parser.add_argument("--with-gallery", dest="with_gallery", default=False, action="store_true", help="enable gallery and multi-prototype appearance memory")
    parser.add_argument("--gallery-size", default=20, type=int, help="stable gallery size kept in each track")
    parser.add_argument("--prototype-count", default=4, type=int, help="maximum number of appearance prototypes per track")
    parser.add_argument("--gallery-match-weight", default=0.4, type=float, help="weight of gallery appearance distance in ReID matching")
    parser.add_argument("--gallery-topk", default=3, type=int, help="top-k gallery matches averaged for appearance distance")
    parser.add_argument("--with-long-recovery", dest="with_long_recovery", default=False, action="store_true", help="enable a dedicated long-occlusion recovery stage for lost tracks")
    parser.add_argument("--long-track-buffer", default=90, type=int, help="frames to keep lost tracks when long recovery is enabled")
    parser.add_argument("--recent-lost-window", default=30, type=int, help="lost tracks newer than this still use the standard ByteTrack association")
    parser.add_argument("--lost-appearance-thresh", default=0.18, type=float, help="stricter appearance threshold for long-lost track recovery")
    parser.add_argument("--long-match-thresh", default=0.35, type=float, help="assignment threshold for long-lost recovery")
    parser.add_argument("--with-temporal-matcher", dest="with_temporal_matcher", default=False, action="store_true", help="enable Transformer-based temporal association for long-lost recovery")
    parser.add_argument("--temporal-model", default="pretrained/temporal_matcher_mot17.pth", type=str, help="temporal Transformer checkpoint path")
    parser.add_argument("--temporal-weight", default=0.5, type=float, help="weight of temporal association cost in long-lost recovery")
    parser.add_argument("--temporal-score-thresh", default=0.55, type=float, help="minimum Transformer match score for long-lost recovery")
    parser.add_argument("--with-temporal-ambiguity", dest="with_temporal_ambiguity", default=False, action="store_true", help="enable Transformer reranking for ambiguous first-stage matches")
    parser.add_argument("--temporal-ambiguity-weight", default=0.6, type=float, help="weight of temporal reranking in ambiguous first-stage matches")
    parser.add_argument("--temporal-ambiguity-score-thresh", default=0.45, type=float, help="minimum Transformer score required to trust ambiguous reranking")
    parser.add_argument("--temporal-ambiguity-score-margin", default=0.12, type=float, help="minimum top-1 vs top-2 Transformer score margin required before overriding an ambiguous first-stage match")
    parser.add_argument("--temporal-ambiguity-gap", default=0.05, type=float, help="top-2 cost gap below which a first-stage match is considered ambiguous")
    parser.add_argument("--temporal-ambiguity-cost-thresh", default=0.7, type=float, help="maximum base cost allowed for ambiguous temporal reranking")
    parser.add_argument("--temporal-ambiguity-override-margin", default=0.04, type=float, help="minimum fused-cost improvement required before temporal reranking can overturn the current best ambiguous candidate")
    parser.add_argument("--temporal-ambiguity-topk", default=2, type=int, help="number of ambiguous first-stage candidates reranked by the temporal matcher")
    parser.add_argument("--temporal-ambiguity-tracked-only", dest="temporal_ambiguity_tracked_only", default=False, action="store_true", help="apply ambiguous temporal reranking only to actively tracked trajectories")
    parser.add_argument("--temporal-proximity-center-thresh", default=1.2, type=float, help="normalized detection-center distance threshold for crowd-triggered temporal reranking")
    parser.add_argument("--temporal-proximity-iou-thresh", default=0.05, type=float, help="detection IoU threshold for crowd-triggered temporal reranking")
    parser.add_argument("--with-anti-swap", dest="with_anti_swap", default=False, action="store_true", help="enable close-contact pairwise anti-swap refinement after first-stage matching")
    parser.add_argument("--anti-swap-motion-weight", default=0.35, type=float, help="weight of motion continuity in pairwise anti-swap refinement")
    parser.add_argument("--anti-swap-order-weight", default=0.35, type=float, help="weight of relative-order consistency in pairwise anti-swap refinement")
    parser.add_argument("--anti-swap-margin", default=0.05, type=float, help="minimum pairwise total-cost improvement required before flipping a close-contact assignment")
    parser.add_argument("--anti-swap-topk", default=2, type=int, help="top-k detections considered for cross-candidate anti-swap checks")
    parser.add_argument("--anti-swap-track-center-thresh", default=1.25, type=float, help="normalized predicted-track center distance threshold for close-contact anti-swap checks")
    parser.add_argument("--anti-swap-det-center-thresh", default=1.35, type=float, help="normalized matched-detection center distance threshold for close-contact anti-swap checks")
    parser.add_argument("--anti-swap-det-iou-thresh", default=0.02, type=float, help="matched-detection IoU threshold used to trigger anti-swap refinement")
    parser.add_argument("--anti-swap-order-margin", default=0.12, type=float, help="minimum normalized separation required before left-right or top-bottom order is enforced")
    parser.add_argument("--temporal-history-len", default=8, type=int, help="minimum retained history length target for temporal association")
    parser.add_argument("--track-feature-history", default=50, type=int, help="feature history size kept in each track")
    parser.add_argument("--gallery-recent-count", default=3, type=int, help="number of recent appearance features included in gallery matching")
    parser.add_argument("--with-cmc", dest="with_cmc", default=False, action="store_true", help="enable global motion compensation before association")
    parser.add_argument("--cmc-downscale", default=2, type=int, help="downscale factor used by the global motion compensator")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run the detector, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=None, type=float, help="test confidence threshold")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test image size")
    parser.add_argument("--fps", default=30, type=int, help="frame rate")
    parser.add_argument("--fp16", dest="fp16", default=False, action="store_true", help="mixed precision")
    parser.add_argument("--fuse", dest="fuse", default=False, action="store_true", help="fuse conv and bn")
    parser.add_argument("--trt", dest="trt", default=False, action="store_true", help="use TensorRT model")
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh",
        type=float,
        default=1.6,
        help="filter boxes above this width/height ratio",
    )
    parser.add_argument("--min_box_area", type=float, default=10, help="filter out tiny boxes")
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test MOT20")
    return apply_config_defaults(parser)


def get_image_list(path):
    image_names = []
    for maindir, _, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1].lower()
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def dump_debug_record(debug_file, debug_record):
    # association_debug.jsonl 采用逐帧追加的方式写入，便于中途停止后继续排查。
    if debug_file is None or debug_record is None:
        return
    debug_file.write(json.dumps(debug_record, ensure_ascii=False) + "\n")
    debug_file.flush()


def image_demo(predictor, vis_folder, current_time, args):
    """
    图片或图片文件夹推理流程。

    它和视频流程的区别只是输入来源不同：
    - image_demo：一次处理一张图片或一个图片序列。
    - imageflow_demo：从视频流里逐帧读取。

    两者最终都会调用同一个 tracker.update()，因此跟踪逻辑完全一致。
    """
    # image 模式按图片列表逐帧跑，适合排查单张图或图像序列。
    if osp.isdir(args.path):
        files = get_image_list(args.path)
    else:
        files = [args.path]
    files.sort()

    tracker = BYTETracker(args, frame_rate=args.fps)
    timer = Timer()
    results = []
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    debug_file = None
    if getattr(args, "save_debug", False):
        debug_folder = osp.join(vis_folder, timestamp) if vis_folder else osp.join("YOLOX_outputs", "track_debug", timestamp)
        os.makedirs(debug_folder, exist_ok=True)
        debug_path = osp.join(debug_folder, "association_debug.jsonl")
        debug_file = open(debug_path, "w", encoding="utf-8")
        logger.info("debug save_path is {}", debug_path)

    try:
        for frame_id, img_path in enumerate(files, 1):
            outputs, img_info = predictor.inference(img_path, timer)

            online_targets = tracker.update(
                outputs[0], [img_info["height"], img_info["width"]], predictor.test_size, img=img_info["raw_img"]
            )
            dump_debug_record(debug_file, tracker.last_debug_frame)

            online_tlwhs = []
            online_ids = []

            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    results.append(
                        f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},"
                        f"{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                    )

            timer.toc()
            online_im = plot_tracking(
                img_info["raw_img"],
                online_tlwhs,
                online_ids,
                frame_id=frame_id,
                fps=1.0 / max(1e-5, timer.average_time),
            )

            if args.save_result:
                save_folder = osp.join(vis_folder, timestamp)
                os.makedirs(save_folder, exist_ok=True)
                cv2.imwrite(osp.join(save_folder, osp.basename(img_path)), online_im)

            if frame_id % 20 == 0:
                logger.info("Processing frame {} ({:.2f} fps)", frame_id, 1.0 / max(1e-5, timer.average_time))

            ch = cv2.waitKey(0)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
    finally:
        if debug_file is not None:
            debug_file.close()

    if args.save_result:
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, "w") as f:
            f.writelines(results)
        logger.info("save results to {}", res_file)


def imageflow_demo(predictor, vis_folder, current_time, args):
    """
    视频/摄像头推理流程。

    这是你平时跑 palace.mp4、test.mp4 时最常用的路径。
    每轮循环处理一帧，主要步骤是：
    1. cap.read() 读取原始帧。
    2. predictor.inference(frame) 做检测。
    3. tracker.update(...) 做跟踪。
    4. plot_tracking(...) 绘制 track id。
    5. 可选写视频、写 MOT txt、写 association_debug.jsonl。
    """
    # video / webcam 模式会持续读取视频帧，这也是最常用的演示路径。
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else args.fps

    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    if args.save_result:
        save_folder = osp.join(vis_folder, timestamp)
        os.makedirs(save_folder, exist_ok=True)
        if args.demo == "video":
            save_path = osp.join(save_folder, osp.basename(args.path))
        else:
            save_path = osp.join(save_folder, "camera.mp4")
        logger.info("video save_path is {}", save_path)
        vid_writer = cv2.VideoWriter(
            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
        )
    else:
        vid_writer = None
    debug_file = None
    if getattr(args, "save_debug", False):
        debug_folder = osp.join(vis_folder, timestamp) if vis_folder else osp.join("YOLOX_outputs", "track_debug", timestamp)
        os.makedirs(debug_folder, exist_ok=True)
        debug_path = osp.join(debug_folder, "association_debug.jsonl")
        debug_file = open(debug_path, "w", encoding="utf-8")
        logger.info("debug save_path is {}", debug_path)

    tracker = BYTETracker(args, frame_rate=args.fps)
    timer = Timer()
    frame_id = 0
    results = []

    try:
        while True:
            if frame_id % 20 == 0:
                logger.info("Processing frame {} ({:.2f} fps)", frame_id, 1.0 / max(1e-5, timer.average_time))

            ret_val, frame = cap.read()
            if not ret_val:
                break

            outputs, img_info = predictor.inference(frame, timer)

            online_targets = tracker.update(
                outputs[0], [img_info["height"], img_info["width"]], predictor.test_size, img=img_info["raw_img"]
            )
            dump_debug_record(debug_file, tracker.last_debug_frame)

            online_tlwhs = []
            online_ids = []

            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    results.append(
                        f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},"
                        f"{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                    )

            timer.toc()
            online_im = plot_tracking(
                img_info["raw_img"],
                online_tlwhs,
                online_ids,
                frame_id=frame_id + 1,
                fps=1.0 / max(1e-5, timer.average_time),
            )

            if vid_writer is not None:
                vid_writer.write(online_im)

            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
            frame_id += 1
    finally:
        if debug_file is not None:
            debug_file.close()

    cap.release()
    if vid_writer is not None:
        vid_writer.release()
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, "w") as f:
            f.writelines(results)
        logger.info("save results to {}", res_file)


def build_yolox_predictor(args, exp):
    """
    构建 YOLOX predictor。

    注意：只有 args.detector == "yolox" 时才会进入这里。
    终版配置使用 YOLOX；加载和融合信息会写入运行日志。
    """
    # 只有当 detector=yolox 时才会走到这里；
    # 如果用的是 Ultralytics，这一整段都不会参与推理。
    model = exp.get_model().to(args.device)
    logger.info("Model Summary: {}", get_model_info(model, exp.test_size))
    model.eval()

    if not args.trt:
        ckpt_file = args.ckpt or osp.join(osp.join(exp.output_dir, args.experiment_name), "best_ckpt.pth.tar")
        logger.info("loading checkpoint")
        load_yolox_detector_weights(model, ckpt_file, class_ids=getattr(args, "classes", [0]), map_location="cpu")
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("Fusing model...")
        model = fuse_model(model)

    if args.fp16:
        model = model.half()

    if args.trt:
        trt_file = osp.join(osp.join(exp.output_dir, args.experiment_name), "model_trt.pth")
        if not osp.exists(trt_file):
            raise FileNotFoundError("TensorRT model not found: {}".format(trt_file))
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
    else:
        trt_file = None
        decoder = None

    return build_detector(args, exp, device=args.device, model=model, trt_file=trt_file, decoder=decoder)


def main(exp, args):
    """
    demo 脚本主入口。

    这个函数负责把所有模块串起来：
    - 处理 device、conf、nms、tsize 等运行配置。
    - 根据 detector 参数选择 Ultralytics YOLO 或 YOLOX。
    - 根据 demo 类型选择图片流程或视频流程。

    如果你后面想画“系统运行流程图”，这个函数可以作为最高层伪代码。
    """
    # main() 负责做两件事：
    # 1. 根据参数初始化 detector 与 tracker
    # 2. 根据 demo 类型把输入送进 image/imageflow 两条推理路径
    if args.trt:
        args.device = "gpu"
    args.device = torch.device("cuda" if args.device == "gpu" and torch.cuda.is_available() else "cpu")
    if args.save_debug:
        args.debug_association = True

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    if not args.experiment_name:
        if args.detector in {"ultralytics", "yolov8"}:
            args.experiment_name = "{}_{}".format(exp.exp_name, Path(args.yolo_model).stem)
        else:
            args.experiment_name = exp.exp_name

    output_dir = osp.join(exp.output_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    vis_folder = None
    if args.save_result:
        vis_folder = osp.join(output_dir, "track_vis")
        os.makedirs(vis_folder, exist_ok=True)

    logger.info("Args: {}", args)
    if args.with_reid:
        logger.info("ReID enabled: backbone={}, weights={}", args.reid_backbone, args.reid_model)
    if args.with_long_recovery:
        logger.info(
            "Long recovery enabled: long_track_buffer={}, recent_lost_window={}, lost_appearance_thresh={}",
            args.long_track_buffer,
            args.recent_lost_window,
            args.lost_appearance_thresh,
        )
    if args.with_temporal_matcher:
        logger.info(
            "Temporal matcher enabled: model={}, temporal_weight={}, temporal_score_thresh={}",
            args.temporal_model,
            args.temporal_weight,
            args.temporal_score_thresh,
        )
    if args.with_temporal_ambiguity:
        logger.info(
            "Temporal ambiguity rerank enabled: weight={}, score_thresh={}, gap={}, topk={}, tracked_only={}",
            args.temporal_ambiguity_weight,
            args.temporal_ambiguity_score_thresh,
            args.temporal_ambiguity_gap,
            args.temporal_ambiguity_topk,
            args.temporal_ambiguity_tracked_only,
        )
    if args.with_anti_swap:
        logger.info(
            "Anti-swap refinement enabled: motion_weight={}, order_weight={}, margin={}, topk={}",
            args.anti_swap_motion_weight,
            args.anti_swap_order_weight,
            args.anti_swap_margin,
            args.anti_swap_topk,
        )
    if args.with_cmc:
        logger.info("Global motion compensation enabled with downscale={}", args.cmc_downscale)
    if args.debug_association:
        logger.info(
            "Association debug enabled: save_debug={}, ambiguous_only={}, topk={}",
            args.save_debug,
            args.debug_ambiguous_only,
            args.debug_topk,
        )

    if args.detector in {"ultralytics", "yolov8"}:
        # 可选 Ultralytics 检测后端。
        if args.ckpt:
            logger.warning("Ignoring `--ckpt` because the Ultralytics detector uses `--yolo-model`.")
        if args.trt:
            raise ValueError("TensorRT inference is only supported for YOLOX in demo_track.py.")
        if args.fuse:
            logger.warning("Ignoring `--fuse` because the Ultralytics detector manages its own inference graph.")
        predictor = build_detector(args, exp, device=args.device)
        logger.info(
            "Using Ultralytics detector weights (model={}): {}",
            Path(args.yolo_model).stem,
            args.yolo_model,
        )
    else:
        # 兼容原始 ByteTrack 的 YOLOX 检测路径。
        predictor = build_yolox_predictor(args, exp)

    current_time = time.localtime()
    if args.demo == "image":
        image_demo(predictor, vis_folder, current_time, args)
    elif args.demo == "video" or args.demo == "webcam":
        imageflow_demo(predictor, vis_folder, current_time, args)
    else:
        raise ValueError("Unsupported demo mode: {}".format(args.demo))


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    main(exp, args)
