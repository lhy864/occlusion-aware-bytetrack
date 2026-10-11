import argparse
import glob
import os
import random
import sys
import warnings
import zipfile
from collections import OrderedDict
from pathlib import Path

import numpy as np
import motmetrics as mm
import torch
import torch.backends.cudnn as cudnn
from loguru import logger
from torch.nn.parallel import DistributedDataParallel as DDP

# NumPy 2.0 删除了 np.asfarray，而当前 motmetrics 仍会调用它。
# 这里做一个轻量兼容补丁，避免评估在计算 MOTA/IDF1 时崩溃。
if not hasattr(np, "asfarray"):
    def _np_asfarray_compat(a, dtype=float):
        return np.asarray(a, dtype=dtype)

    np.asfarray = _np_asfarray_compat

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from yolox.core import launch
from yolox.evaluators import MOTEvaluator, UltralyticsMOTEvaluator
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, setup_logger
from yolox.utils.cli_config import apply_config_defaults
from yolox.utils.detector import build_detector, load_yolox_detector_weights


def make_parser():
    """
    构建标准数据集评估参数。

    它和 demo_track.py 的参数很像，但使用场景不同：
    - demo_track.py 面向单个视频或摄像头演示。
    - track.py 面向 MOT17/MOT20 这类标准数据集评估。

    因此这里没有 --path，而是通过 exp_file 中的数据集配置构造 dataloader。
    """
    # 评估脚本的参数基本和 demo 保持同构，
    # 区别在于这里的输入是标准 MOT dataloader，而不是单个视频文件。
    parser = argparse.ArgumentParser("ByteTrack Eval")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("--debug-association", dest="debug_association", default=False, action="store_true", help="collect first-stage association debug information inside the tracker")
    parser.add_argument("--debug-ambiguous-only", dest="debug_ambiguous_only", default=False, action="store_true", help="only keep debug records for ambiguous first-stage matches")
    parser.add_argument("--debug-topk", default=3, type=int, help="top-k candidate detections exported for each track in debug records")

    parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
    parser.add_argument("--dist-url", default=None, type=str, help="url used to set up distributed training")
    parser.add_argument("-b", "--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("-d", "--devices", default=None, type=int, help="device for training")
    parser.add_argument("--local_rank", default=0, type=int, help="local rank for dist training")
    parser.add_argument("--num_machines", default=1, type=int, help="num of node for training")
    parser.add_argument("--machine_rank", default=0, type=int, help="node rank for multi-node training")
    parser.add_argument("-f", "--exp_file", default=None, type=str, help="experiment description file")
    parser.add_argument("--fp16", dest="fp16", default=False, action="store_true", help="mixed precision")
    parser.add_argument("--fuse", dest="fuse", default=False, action="store_true", help="fuse conv and bn")
    parser.add_argument("--trt", dest="trt", default=False, action="store_true", help="use TensorRT model")
    parser.add_argument("--test", dest="test", default=False, action="store_true", help="evaluate on test-dev set")
    parser.add_argument("--speed", dest="speed", default=False, action="store_true", help="speed test only")
    parser.add_argument(
        "opts",
        help="modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

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
    parser.add_argument("--temporal-ambiguity-gap", default=0.05, type=float, help="top-2 cost gap below which a first-stage match is considered ambiguous")
    parser.add_argument("--temporal-ambiguity-cost-thresh", default=0.7, type=float, help="maximum base cost allowed for ambiguous temporal reranking")
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
    parser.add_argument("--conf", default=0.01, type=float, help="test confidence threshold")
    parser.add_argument("--nms", default=0.7, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test image size")
    parser.add_argument("--tsize-h", default=None, type=int, help="test image height override")
    parser.add_argument("--tsize-w", default=None, type=int, help="test image width override")
    parser.add_argument("--seed", default=None, type=int, help="eval seed")

    parser.add_argument("--track_thresh", type=float, default=0.6, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.9, help="matching threshold for tracking")
    parser.add_argument("--min-box-area", type=float, default=100, help="filter out tiny boxes")
    parser.add_argument(
        "--legacy-seq-tuning",
        default=True,
        type=lambda x: str(x).lower() in ("1", "true", "yes", "on"),
        help="whether to keep the original ByteTrack per-sequence threshold overrides for MOT17/MOT20 eval",
    )
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test MOT20")
    parser.add_argument("--skip-metrics", dest="skip_metrics", default=False, action="store_true", help="skip local MOT metrics, useful for official test sets without public ground truth")
    parser.add_argument("--make-submission-zip", dest="make_submission_zip", default=False, action="store_true", help="zip MOTChallenge result txt files for official submission")
    parser.add_argument("--submission-name", default=None, type=str, help="submission zip base name, defaults to experiment name")
    return apply_config_defaults(parser)


def make_submission_zip(results_folder, output_zip):
    result_files = sorted(
        f
        for f in glob.glob(os.path.join(results_folder, "*.txt"))
        if not os.path.basename(f).startswith("eval")
    )
    if not result_files:
        raise FileNotFoundError("No MOTChallenge result txt files found in {}".format(results_folder))

    os.makedirs(os.path.dirname(output_zip), exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for result_file in result_files:
            archive.write(result_file, arcname=os.path.basename(result_file))
    return output_zip, len(result_files)


def compare_dataframes(gts, ts):
    """
    将 ground truth 和 tracker 预测结果按序列名对齐。

    motmetrics 计算 MOTA/IDF1 等指标前，需要把每个视频序列的 GT
    和该序列对应的预测 txt 成对送入 compare_to_groundtruth。
    """
    # motmetrics 本地评估时，需要先把 GT 与预测结果按视频名对齐。
    accs = []
    names = []
    for k, tsacc in ts.items():
        if k in gts:
            logger.info("Comparing {}...", k)
            accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, "iou", distth=0.5))
            names.append(k)
        else:
            logger.warning("No ground truth for {}, skipping.", k)
    return accs, names


@logger.catch
def main(exp, args, num_gpu):
    """
    标准 MOT 评估主入口。

    运行流程：
    1. 初始化分布式/单卡环境。
    2. 构建 MOT dataloader。
    3. 根据 detector 选择 YOLOX 或 Ultralytics evaluator。
    4. evaluator 逐帧检测和跟踪。
    5. 输出 MOTChallenge 格式结果文件，并生成指标 summary。
    """
    # 这个入口负责“批量数据集评估”，而不是单视频演示。
    # 论文里要写 MOTA/IDF1 等指标时，主要依赖这条链路。
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn("You have chosen to seed testing. This will turn on the CUDNN deterministic setting.")

    if args.batch_size != 1:
        logger.warning("ByteTrack tracking eval expects `-b 1`; overriding {} to 1.", args.batch_size)
        args.batch_size = 1

    is_distributed = num_gpu > 1
    cudnn.benchmark = True

    rank = args.local_rank
    args.device = torch.device("cuda:{}".format(rank) if torch.cuda.is_available() else "cpu")
    file_name = os.path.join(exp.output_dir, args.experiment_name)
    if rank == 0:
        os.makedirs(file_name, exist_ok=True)

    results_folder = os.path.join(file_name, "track_results")
    os.makedirs(results_folder, exist_ok=True)

    setup_logger(file_name, distributed_rank=rank, filename="val_log.txt", mode="a")
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
    if args.with_cmc:
        logger.info("Global motion compensation enabled with downscale={}", args.cmc_downscale)

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize_h is not None or args.tsize_w is not None:
        if args.tsize_h is None or args.tsize_w is None:
            raise ValueError("Both --tsize-h and --tsize-w must be provided together.")
        exp.test_size = (args.tsize_h, args.tsize_w)
    elif args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    val_loader = exp.get_eval_loader(args.batch_size, is_distributed, args.test)
    # 当 detector=ultralytics / yolov8 时，使用自定义的 UltralyticsMOTEvaluator；
    # 否则保持原项目的 YOLOX evaluator。
    evaluator_cls = UltralyticsMOTEvaluator if args.detector in {"ultralytics", "yolov8"} else MOTEvaluator
    evaluator = evaluator_cls(
        args=args,
        dataloader=val_loader,
        img_size=exp.test_size,
        confthre=exp.test_conf,
        nmsthre=exp.nmsthre,
        num_classes=exp.num_classes,
    )

    if args.detector in {"ultralytics", "yolov8"}:
        device = torch.device("cuda:{}".format(rank) if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda.set_device(rank)

        if args.ckpt:
            logger.warning("Ignoring `--ckpt` because the Ultralytics detector uses `--yolo-model`.")
        if args.trt:
            raise ValueError("TensorRT evaluation is only supported for YOLOX in tools/track.py.")
        if args.fuse:
            logger.warning("Ignoring `--fuse` because the Ultralytics detector manages its own inference graph.")

        detector = build_detector(args, exp, device=device)
        logger.info(
            "Using Ultralytics detector weights (model={}): {}",
            Path(args.yolo_model).stem,
            args.yolo_model,
        )
        *_, summary = evaluator.evaluate(detector, distributed=is_distributed, result_folder=results_folder)
    else:
        model = exp.get_model()
        logger.info("Model Summary: {}", get_model_info(model, exp.test_size))

        torch.cuda.set_device(rank)
        model.cuda(rank)
        model.eval()

        if not args.speed and not args.trt:
            ckpt_file = args.ckpt or os.path.join(file_name, "best_ckpt.pth.tar")
            logger.info("loading checkpoint")
            loc = "cuda:{}".format(rank)
            load_yolox_detector_weights(model, ckpt_file, class_ids=getattr(args, "classes", [0]), map_location=loc)
            logger.info("loaded checkpoint done.")

        if is_distributed:
            model = DDP(model, device_ids=[rank])

        if args.fuse:
            logger.info("Fusing model...")
            model = fuse_model(model)

        if args.trt:
            if args.fuse or is_distributed or args.batch_size != 1:
                raise ValueError("TensorRT model does not support fuse, distributed inference, or batch size > 1.")
            trt_file = os.path.join(file_name, "model_trt.pth")
            if not os.path.exists(trt_file):
                raise FileNotFoundError("TensorRT model is not found: {}".format(trt_file))
            model.head.decode_in_inference = False
            decoder = model.head.decode_outputs
        else:
            trt_file = None
            decoder = None

        *_, summary = evaluator.evaluate(
            model, is_distributed, args.fp16, trt_file, decoder, exp.test_size, results_folder
        )

    logger.info("\n{}", summary)

    if args.make_submission_zip:
        zip_base = args.submission_name or "{}_submission".format(args.experiment_name)
        zip_path = os.path.join(file_name, "{}.zip".format(zip_base))
        zip_path, file_count = make_submission_zip(results_folder, zip_path)
        logger.info("Created MOTChallenge submission zip with {} files: {}", file_count, zip_path)
        print("Submission zip: {}".format(zip_path))

    if args.skip_metrics:
        logger.info("Skipping local metrics because skip_metrics=True.")
        return

    mm.lap.default_solver = "lap"

    gt_type = "_val_half" if exp.val_ann == "val_half.json" else ""
    if args.mot20:
        gtfiles = glob.glob(os.path.join("datasets/MOT20/train", "*/gt/gt{}.txt".format(gt_type)))
    else:
        gtfiles = glob.glob(os.path.join("datasets/mot/train", "*/gt/gt{}.txt".format(gt_type)))
    tsfiles = [
        f
        for f in glob.glob(os.path.join(results_folder, "*.txt"))
        if not os.path.basename(f).startswith("eval")
    ]

    logger.info("Found {} groundtruths and {} test files.", len(gtfiles), len(tsfiles))
    logger.info("Available LAP solvers {}", mm.lap.available_solvers)
    logger.info("Default LAP solver '{}'", mm.lap.default_solver)
    logger.info("Loading files.")

    gt = OrderedDict(
        [(Path(f).parts[-3], mm.io.loadtxt(f, fmt="mot15-2D", min_confidence=1)) for f in gtfiles]
    )
    ts = OrderedDict(
        [
            (os.path.splitext(Path(f).parts[-1])[0], mm.io.loadtxt(f, fmt="mot15-2D", min_confidence=-1))
            for f in tsfiles
        ]
    )

    mh = mm.metrics.create()
    accs, names = compare_dataframes(gt, ts)

    logger.info("Running metrics")
    metrics = [
        "recall",
        "precision",
        "num_unique_objects",
        "mostly_tracked",
        "partially_tracked",
        "mostly_lost",
        "num_false_positives",
        "num_misses",
        "num_switches",
        "num_fragmentations",
        "mota",
        "motp",
        "num_objects",
    ]
    summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
    div_dict = {
        "num_objects": ["num_false_positives", "num_misses", "num_switches", "num_fragmentations"],
        "num_unique_objects": ["mostly_tracked", "partially_tracked", "mostly_lost"],
    }
    for divisor in div_dict:
        for divided in div_dict[divisor]:
            summary[divided] = summary[divided] / summary[divisor]

    fmt = mh.formatters
    change_fmt_list = [
        "num_false_positives",
        "num_misses",
        "num_switches",
        "num_fragmentations",
        "mostly_tracked",
        "partially_tracked",
        "mostly_lost",
    ]
    for k in change_fmt_list:
        fmt[k] = fmt["mota"]
    short_summary_text = mm.io.render_summary(
        summary,
        formatters=fmt,
        namemap=mm.io.motchallenge_metric_names,
    )
    print(short_summary_text)
    logger.info("\n{}", short_summary_text)

    metrics = mm.metrics.motchallenge_metrics + ["num_objects"]
    summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
    full_summary_text = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names,
    )
    print(full_summary_text)
    logger.info("\n{}", full_summary_text)
    logger.info("Completed")


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        if args.detector in {"ultralytics", "yolov8"}:
            args.experiment_name = "{}_{}".format(exp.exp_name, Path(args.yolo_model).stem)
        else:
            args.experiment_name = exp.exp_name

    num_gpu = torch.cuda.device_count() if args.devices is None else args.devices
    assert num_gpu <= torch.cuda.device_count()

    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=args.dist_url,
        args=(exp, args, num_gpu),
    )
