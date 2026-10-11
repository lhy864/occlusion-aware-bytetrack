#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
把 MOT17 标注导出成 ReID 微调数据集。

目标是把原本用于跟踪的数据，重组织成：
1. 行人裁剪图
2. 身份标签 pid
3. metadata.jsonl

这样后面就可以直接用 OSNet 在 MOT 域上做外观微调。
"""

import argparse
import configparser
import json
import math
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
from loguru import logger

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from yolox.utils.cli_config import apply_config_defaults


def make_parser():
    """
    构建 ReID 数据集导出参数。

    重点参数：
    - mot_root：原始 MOT17 数据集路径。
    - output_root：导出的 ReID 数据保存位置。
    - split_mode：是否按 train/val half 切分。
    - min_visibility / min_box_*：过滤不可用行人框。
    """
    parser = argparse.ArgumentParser("Build MOT17 ReID fine-tuning dataset")
    parser.add_argument(
        "--mot-root",
        default="datasets/mot",
        type=str,
        help="root directory of MOT17",
    )
    parser.add_argument(
        "--output-root",
        default="datasets/reid_mot17",
        type=str,
        help="directory where the exported ReID dataset will be written",
    )
    parser.add_argument(
        "--sequence-suffix",
        default="FRCNN",
        type=str,
        help="only export sequences ending with this suffix to avoid MOT17 triple-counting",
    )
    parser.add_argument(
        "--split-mode",
        default="half",
        choices=["half", "all"],
        help="use train_half/val_half or the full train split",
    )
    parser.add_argument(
        "--min-visibility",
        default=0.2,
        type=float,
        help="minimum visibility for keeping a person crop",
    )
    parser.add_argument(
        "--min-box-width",
        default=16.0,
        type=float,
        help="minimum bbox width after filtering",
    )
    parser.add_argument(
        "--min-box-height",
        default=32.0,
        type=float,
        help="minimum bbox height after filtering",
    )
    parser.add_argument(
        "--min-box-area",
        default=1024.0,
        type=float,
        help="minimum bbox area after filtering",
    )
    parser.add_argument(
        "--bbox-padding",
        default=0.05,
        type=float,
        help="relative padding applied to each bbox before cropping",
    )
    parser.add_argument(
        "--frame-step",
        default=1,
        type=int,
        help="keep one frame every N frames inside each split",
    )
    parser.add_argument(
        "--min-samples-per-id",
        default=2,
        type=int,
        help="discard identities with fewer than this many samples inside a split",
    )
    parser.add_argument(
        "--max-neighbors",
        default=5,
        type=int,
        help="maximum nearby different identities recorded per sample",
    )
    parser.add_argument(
        "--neighbor-center-thresh",
        default=1.5,
        type=float,
        help="normalized center distance threshold for nearby different identities",
    )
    parser.add_argument(
        "--neighbor-iou-thresh",
        default=0.0,
        type=float,
        help="IoU threshold for nearby different identities",
    )
    parser.add_argument(
        "--max-sequences",
        default=0,
        type=int,
        help="optional cap on number of sequences processed, 0 means unlimited",
    )
    parser.add_argument(
        "--max-frames",
        default=0,
        type=int,
        help="optional cap on frames per exported split for smoke tests, 0 means unlimited",
    )
    parser.add_argument(
        "--jpeg-quality",
        default=95,
        type=int,
        help="jpeg quality for exported crops",
    )
    parser.add_argument(
        "--overwrite",
        default=False,
        action="store_true",
        help="overwrite the output directory if it already exists",
    )
    return apply_config_defaults(parser)


def parse_seqinfo(seq_dir):
    parser = configparser.ConfigParser()
    parser.read(os.path.join(seq_dir, "seqinfo.ini"))
    section = parser["Sequence"]
    return {
        "name": section.get("name", os.path.basename(seq_dir)),
        "image_dir": os.path.join(seq_dir, section.get("imDir", "img1")),
        "image_ext": section.get("imExt", ".jpg"),
        "seq_length": section.getint("seqLength"),
        "width": section.getint("imWidth"),
        "height": section.getint("imHeight"),
    }


def resolve_split_ranges(seq_length, split_mode):
    # half 模式下，训练集和验证集按时间顺序切开，
    # 可以避免同一时段同时出现在 train / val 中。
    if split_mode == "all":
        return {"train": (1, seq_length)}

    train_end = seq_length // 2 + 1
    val_start = train_end + 1
    return {
        "train": (1, train_end),
        "val": (val_start, seq_length),
    }


def clip_crop_box(tlwh, image_width, image_height, padding):
    # 在裁图前给框留一点边缘，上下文对 ReID 往往有帮助。
    x, y, w, h = tlwh
    pad_w = w * padding
    pad_h = h * padding
    x1 = max(int(math.floor(x - pad_w)), 0)
    y1 = max(int(math.floor(y - pad_h)), 0)
    x2 = min(int(math.ceil(x + w + pad_w)), image_width - 1)
    y2 = min(int(math.ceil(y + h + pad_h)), image_height - 1)
    return x1, y1, x2, y2


def compute_iou(a, b):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(inter_x2 - inter_x1, 0.0)
    inter_h = max(inter_y2 - inter_y1, 0.0)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    union_area = aw * ah + bw * bh - inter_area
    if union_area <= 0:
        return 0.0
    return float(inter_area / union_area)


def center_distance_norm(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    acx = ax + aw / 2.0
    acy = ay + ah / 2.0
    bcx = bx + bw / 2.0
    bcy = by + bh / 2.0
    dist = math.hypot(acx - bcx, acy - bcy)
    scale = max(max(ah, bh), 1.0)
    return float(dist / scale)


def discover_sequences(mot_root, sequence_suffix, max_sequences):
    train_dir = os.path.join(mot_root, "train")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError("MOT17 train directory not found: {}".format(train_dir))

    sequence_names = sorted(
        [
            name
            for name in os.listdir(train_dir)
            if os.path.isdir(os.path.join(train_dir, name))
            and (not sequence_suffix or name.endswith(sequence_suffix))
        ]
    )
    if max_sequences > 0:
        sequence_names = sequence_names[:max_sequences]
    if not sequence_names:
        raise RuntimeError("No sequences matched under {}".format(train_dir))
    return train_dir, sequence_names


def collect_raw_samples(args):
    """
    从 MOT17 标注中收集待导出的行人样本元信息。

    这个阶段只读取 gt.txt 和 seqinfo.ini，不写图片。
    每个样本会记录：
    - 来自哪个序列、哪一帧、哪个 track id
    - 原始 tlwh 框
    - 该样本属于 train 还是 val
    - 同帧附近的不同 ID，用于后续 hard negative 分析
    """
    # 第一步只做“收集样本元信息”，还不真正落盘图片。
    train_dir, sequence_names = discover_sequences(args.mot_root, args.sequence_suffix, args.max_sequences)
    split_raw_samples = defaultdict(list)
    split_pid_counts = defaultdict(Counter)
    global_pid_map = {}
    sequence_split_ranges = {}

    for sequence_name in sequence_names:
        sequence_dir = os.path.join(train_dir, sequence_name)
        seqinfo = parse_seqinfo(sequence_dir)
        split_ranges = resolve_split_ranges(seqinfo["seq_length"], args.split_mode)
        sequence_split_ranges[sequence_name] = split_ranges
        gt_path = os.path.join(sequence_dir, "gt", "gt.txt")

        samples_by_split_and_frame = defaultdict(lambda: defaultdict(list))
        with open(gt_path, "r") as handle:
            for line in handle:
                fields = line.strip().split(",")
                if len(fields) < 9:
                    continue

                frame_id = int(fields[0])
                source_track_id = int(fields[1])
                tlwh = [float(fields[2]), float(fields[3]), float(fields[4]), float(fields[5])]
                conf = int(float(fields[6]))
                cls = int(float(fields[7]))
                visibility = float(fields[8])

                if conf != 1 or cls != 1:
                    continue
                if visibility < args.min_visibility:
                    continue
                if tlwh[2] < args.min_box_width or tlwh[3] < args.min_box_height:
                    continue
                if tlwh[2] * tlwh[3] < args.min_box_area:
                    continue

                for split_name, (start_frame, end_frame) in split_ranges.items():
                    if frame_id < start_frame or frame_id > end_frame:
                        continue
                    if args.max_frames > 0 and frame_id > start_frame + args.max_frames - 1:
                        continue
                    if args.frame_step > 1 and ((frame_id - start_frame) % args.frame_step != 0):
                        continue

                    pid_key = "{}:{}".format(sequence_name, source_track_id)
                    global_pid = global_pid_map.setdefault(pid_key, len(global_pid_map) + 1)
                    image_name = "{:06d}{}".format(frame_id, seqinfo["image_ext"])
                    image_path = os.path.join(seqinfo["image_dir"], image_name)
                    sample = {
                        "split": split_name,
                        "pid_key": pid_key,
                        "global_pid": global_pid,
                        "sequence_name": sequence_name,
                        "source_track_id": source_track_id,
                        "source_frame_id": frame_id,
                        "split_frame_id": frame_id - start_frame + 1,
                        "bbox_tlwh": tlwh,
                        "visibility": visibility,
                        "image_width": seqinfo["width"],
                        "image_height": seqinfo["height"],
                        "source_image_path": image_path,
                        "source_image_rel": os.path.relpath(image_path, args.mot_root).replace("\\", "/"),
                    }
                    samples_by_split_and_frame[split_name][frame_id].append(sample)
                    break

        for split_name, frames in samples_by_split_and_frame.items():
            for frame_id, frame_samples in frames.items():
                for sample in frame_samples:
                    neighbor_records = []
                    for other in frame_samples:
                        if other["pid_key"] == sample["pid_key"]:
                            continue
                        iou = compute_iou(sample["bbox_tlwh"], other["bbox_tlwh"])
                        center_norm = center_distance_norm(sample["bbox_tlwh"], other["bbox_tlwh"])
                        if iou > args.neighbor_iou_thresh or center_norm < args.neighbor_center_thresh:
                            neighbor_records.append(
                                {
                                    "pid_key": other["pid_key"],
                                    "global_pid": other["global_pid"],
                                    "iou": round(iou, 6),
                                    "center_distance_norm": round(center_norm, 6),
                                }
                            )
                    neighbor_records.sort(key=lambda item: (item["center_distance_norm"], -item["iou"], item["pid_key"]))
                    sample["nearby_candidates"] = neighbor_records[: args.max_neighbors]
                    split_raw_samples[split_name].append(sample)
                    split_pid_counts[split_name][sample["pid_key"]] += 1

    return split_raw_samples, split_pid_counts, global_pid_map, sequence_split_ranges


def materialize_dataset(args, split_raw_samples, split_pid_counts, global_pid_map, sequence_split_ranges):
    """
    将收集到的样本真正写成 ReID 数据集。

    输出内容包括：
    - train/images 和 val/images：裁剪后的行人图片。
    - metadata.jsonl：每张裁剪图对应的源序列、帧号、身份等信息。
    - label_maps.json：原始 MOT track id 到连续训练标签的映射。
    - stats.json：样本数量、身份数量等统计信息。
    """
    # 第二步才真正把 crop 图和 metadata 写到磁盘上。
    output_root = Path(args.output_root)
    if output_root.exists():
        existing_entries = list(output_root.iterdir())
        if existing_entries and not args.overwrite:
            raise FileExistsError(
                "Output directory is not empty: {}. Use --overwrite to rebuild it.".format(output_root)
            )
        if args.overwrite:
            shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    kept_pid_maps = {}
    global_identities = defaultdict(lambda: {"splits": set(), "sequence_name": None, "source_track_id": None})
    split_written_samples = {}
    split_stats = {}

    for split_name, raw_samples in split_raw_samples.items():
        kept_pid_keys = sorted(
            [pid_key for pid_key, count in split_pid_counts[split_name].items() if count >= args.min_samples_per_id]
        )
        split_pid_map = {pid_key: idx + 1 for idx, pid_key in enumerate(kept_pid_keys)}
        kept_pid_maps[split_name] = split_pid_map

        split_dir = output_root / split_name
        images_root = split_dir / "images"
        images_root.mkdir(parents=True, exist_ok=True)
        metadata_path = split_dir / "metadata.jsonl"
        written_samples = []

        kept_samples = [sample for sample in raw_samples if sample["pid_key"] in split_pid_map]
        kept_samples_by_image = defaultdict(list)
        for sample in kept_samples:
            kept_samples_by_image[sample["source_image_path"]].append(sample)

        logger.info(
            "Exporting split `{}` with {} samples and {} identities",
            split_name,
            len(kept_samples),
            len(split_pid_map),
        )
        sample_id = 0
        with metadata_path.open("w", encoding="utf-8") as metadata_file:
            for source_image_path in sorted(kept_samples_by_image.keys()):
                image = cv2.imread(source_image_path)
                if image is None:
                    raise FileNotFoundError("failed to load image: {}".format(source_image_path))

                for sample in sorted(
                    kept_samples_by_image[source_image_path],
                    key=lambda item: (item["source_frame_id"], item["source_track_id"]),
                ):
                    crop_box = clip_crop_box(
                        sample["bbox_tlwh"],
                        image_width=sample["image_width"],
                        image_height=sample["image_height"],
                        padding=args.bbox_padding,
                    )
                    x1, y1, x2, y2 = crop_box
                    if x2 <= x1 or y2 <= y1:
                        continue
                    crop = image[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    sample_id += 1
                    split_pid = split_pid_map[sample["pid_key"]]
                    identity_dir = images_root / "{:06d}".format(split_pid)
                    identity_dir.mkdir(parents=True, exist_ok=True)
                    export_name = "{}_f{:06d}_tid{:06d}.jpg".format(
                        sample["sequence_name"],
                        sample["source_frame_id"],
                        sample["source_track_id"],
                    )
                    export_path = identity_dir / export_name
                    cv2.imwrite(
                        str(export_path),
                        crop,
                        [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)],
                    )

                    filtered_neighbors = []
                    for neighbor in sample["nearby_candidates"]:
                        neighbor_split_pid = split_pid_map.get(neighbor["pid_key"])
                        if neighbor_split_pid is None:
                            continue
                        filtered_neighbors.append(
                            {
                                "pid_key": neighbor["pid_key"],
                                "global_pid": neighbor["global_pid"],
                                "split_pid": neighbor_split_pid,
                                "iou": neighbor["iou"],
                                "center_distance_norm": neighbor["center_distance_norm"],
                            }
                        )

                    metadata_record = {
                        "sample_id": sample_id,
                        "split": split_name,
                        "pid_key": sample["pid_key"],
                        "global_pid": sample["global_pid"],
                        "split_pid": split_pid,
                        "sequence_name": sample["sequence_name"],
                        "source_track_id": sample["source_track_id"],
                        "source_frame_id": sample["source_frame_id"],
                        "split_frame_id": sample["split_frame_id"],
                        "source_image_rel": sample["source_image_rel"],
                        "export_image_rel": export_path.relative_to(output_root).as_posix(),
                        "bbox_tlwh": [round(v, 3) for v in sample["bbox_tlwh"]],
                        "crop_tlbr": [int(v) for v in crop_box],
                        "visibility": round(float(sample["visibility"]), 6),
                        "image_width": int(sample["image_width"]),
                        "image_height": int(sample["image_height"]),
                        "area": round(float(sample["bbox_tlwh"][2] * sample["bbox_tlwh"][3]), 3),
                        "nearby_different_ids": filtered_neighbors,
                    }
                    metadata_file.write(json.dumps(metadata_record, ensure_ascii=False) + "\n")
                    written_samples.append(metadata_record)

                    identity_info = global_identities[sample["pid_key"]]
                    identity_info["splits"].add(split_name)
                    identity_info["sequence_name"] = sample["sequence_name"]
                    identity_info["source_track_id"] = sample["source_track_id"]

        split_written_samples[split_name] = written_samples

        samples_per_id = Counter(record["split_pid"] for record in written_samples)
        areas = [record["area"] for record in written_samples]
        visibilities = [record["visibility"] for record in written_samples]
        split_stats[split_name] = {
            "num_samples": len(written_samples),
            "num_identities": len(samples_per_id),
            "num_sequences": len(set(record["sequence_name"] for record in written_samples)),
            "samples_per_identity": {
                "min": min(samples_per_id.values()) if samples_per_id else 0,
                "max": max(samples_per_id.values()) if samples_per_id else 0,
                "mean": round(sum(samples_per_id.values()) / max(len(samples_per_id), 1), 4),
            },
            "visibility": {
                "min": round(min(visibilities), 6) if visibilities else 0.0,
                "max": round(max(visibilities), 6) if visibilities else 0.0,
                "mean": round(sum(visibilities) / max(len(visibilities), 1), 6),
            },
            "area": {
                "min": round(min(areas), 3) if areas else 0.0,
                "max": round(max(areas), 3) if areas else 0.0,
                "mean": round(sum(areas) / max(len(areas), 1), 3),
            },
        }

    label_maps = {
        split_name: {
            "num_identities": len(pid_map),
            "pid_key_to_split_pid": pid_map,
        }
        for split_name, pid_map in kept_pid_maps.items()
    }
    label_map_path = output_root / "label_maps.json"
    label_map_path.write_text(json.dumps(label_maps, indent=2, ensure_ascii=False), encoding="utf-8")

    dataset_info = {
        "mot_root": os.path.abspath(args.mot_root),
        "output_root": str(output_root.resolve()),
        "sequence_suffix": args.sequence_suffix,
        "split_mode": args.split_mode,
        "sequence_split_ranges": {
            sequence_name: {
                split_name: {"start_frame": start_frame, "end_frame": end_frame}
                for split_name, (start_frame, end_frame) in split_ranges.items()
            }
            for sequence_name, split_ranges in sequence_split_ranges.items()
        },
        "global_identities": [
            {
                "pid_key": pid_key,
                "global_pid": global_pid_map[pid_key],
                "sequence_name": info["sequence_name"],
                "source_track_id": info["source_track_id"],
                "splits": sorted(info["splits"]),
            }
            for pid_key, info in sorted(global_identities.items(), key=lambda item: global_pid_map[item[0]])
        ],
    }
    dataset_info_path = output_root / "dataset_info.json"
    dataset_info_path.write_text(json.dumps(dataset_info, indent=2, ensure_ascii=False), encoding="utf-8")

    per_sequence_stats = defaultdict(lambda: defaultdict(lambda: {"samples": 0, "identities": set()}))
    for split_name, written_samples in split_written_samples.items():
        for record in written_samples:
            seq_stats = per_sequence_stats[record["sequence_name"]][split_name]
            seq_stats["samples"] += 1
            seq_stats["identities"].add(record["pid_key"])
    per_sequence_stats = {
        sequence_name: {
            split_name: {
                "num_samples": stats["samples"],
                "num_identities": len(stats["identities"]),
            }
            for split_name, stats in split_stats_dict.items()
        }
        for sequence_name, split_stats_dict in per_sequence_stats.items()
    }

    stats = {
        "args": vars(args),
        "splits": split_stats,
        "per_sequence": per_sequence_stats,
    }
    stats_path = output_root / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    return split_stats


def main():
    args = make_parser().parse_args()
    logger.info("Args: {}", args)
    split_raw_samples, split_pid_counts, global_pid_map, sequence_split_ranges = collect_raw_samples(args)
    split_stats = materialize_dataset(
        args,
        split_raw_samples=split_raw_samples,
        split_pid_counts=split_pid_counts,
        global_pid_map=global_pid_map,
        sequence_split_ranges=sequence_split_ranges,
    )
    logger.info("Finished building MOT17 ReID dataset at {}", Path(args.output_root).resolve())
    for split_name, stats in split_stats.items():
        logger.info(
            "Split `{}`: samples={}, identities={}, mean_samples_per_id={}",
            split_name,
            stats["num_samples"],
            stats["num_identities"],
            stats["samples_per_identity"]["mean"],
        )


if __name__ == "__main__":
    main()
