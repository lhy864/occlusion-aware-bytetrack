#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
构建 temporal matcher 的训练样本。

核心思路是把 MOT17 的轨迹切成：
1. 历史轨迹片段
2. 正样本候选框
3. 负样本候选框

再把它们整理成 Transformer 训练时需要的固定长度张量。
"""

import argparse
import configparser
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from loguru import logger

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from yolox.tracker.reid_extractor import OSNetReIDExtractor
from yolox.tracker.temporal_matcher import build_temporal_pair_inputs


def make_parser():
    parser = argparse.ArgumentParser("Build temporal matcher dataset")
    parser.add_argument(
        "--mot-root",
        default="datasets/mot",
        type=str,
        help="root directory of MOT17",
    )
    parser.add_argument(
        "--output",
        default="datasets/temporal_matcher_mot17_train.pt",
        type=str,
        help="path to save the generated training samples",
    )
    parser.add_argument(
        "--reid-model",
        default="pretrained/osnet_x0_25_msmt17.pt",
        type=str,
        help="OSNet ReID checkpoint used to extract appearance features",
    )
    parser.add_argument("--device", default="cpu", type=str, help="device for feature extraction")
    parser.add_argument("--reid-batch-size", default=64, type=int, help="batch size for OSNet feature extraction")
    parser.add_argument(
        "--sequence-suffix",
        default="FRCNN",
        type=str,
        help="only use sequences ending with this suffix to avoid triple-counting MOT17",
    )
    parser.add_argument("--history-len", default=8, type=int, help="number of history steps kept for each sample")
    parser.add_argument("--min-history", default=3, type=int, help="minimum valid history observations")
    parser.add_argument("--min-gap", default=8, type=int, help="minimum frame gap between history tail and candidate")
    parser.add_argument("--max-gap", default=120, type=int, help="maximum frame gap between history tail and candidate")
    parser.add_argument("--negatives-per-positive", default=1, type=int, help="hard negatives sampled for each positive pair")
    parser.add_argument("--min-visibility", default=0.2, type=float, help="minimum MOT visibility kept for sample generation")
    parser.add_argument("--max-positives", default=0, type=int, help="optional cap on total positive samples, 0 means unlimited")
    parser.add_argument("--max-sequences", default=0, type=int, help="optional cap on how many sequences to process, 0 means unlimited")
    parser.add_argument("--max-frames", default=0, type=int, help="optional cap on frames per sequence for quick smoke tests")
    return parser


def parse_seqinfo(seq_dir):
    parser = configparser.ConfigParser()
    parser.read(os.path.join(seq_dir, "seqinfo.ini"))
    section = parser["Sequence"]
    return {
        "name": section.get("name", os.path.basename(seq_dir)),
        "width": section.getint("imWidth"),
        "height": section.getint("imHeight"),
        "image_dir": os.path.join(seq_dir, section.get("imDir")),
        "image_ext": section.get("imExt"),
    }


def load_annotations(seq_dir, min_visibility):
    """
    读取单个 MOT17 序列的 GT 标注。

    这里会同时建立两个索引：
    - annotations_by_frame：按帧组织，方便在同一帧里寻找不同 ID 的负样本。
    - annotations_by_track：按轨迹 ID 组织，方便抽取一条轨迹的历史序列。
    """
    annotations_by_frame = defaultdict(list)
    annotations_by_track = defaultdict(list)
    gt_path = os.path.join(seq_dir, "gt", "gt.txt")
    with open(gt_path, "r") as handle:
        for line in handle:
            fields = line.strip().split(",")
            if len(fields) < 9:
                continue
            frame_id = int(fields[0])
            track_id = int(fields[1])
            tlwh = np.asarray([float(fields[2]), float(fields[3]), float(fields[4]), float(fields[5])], dtype=np.float32)
            conf = int(float(fields[6]))
            cls = int(float(fields[7]))
            visibility = float(fields[8])
            if conf != 1 or cls != 1 or visibility < min_visibility:
                continue

            annotation = {
                "frame_id": frame_id,
                "track_id": track_id,
                "tlwh": tlwh,
                "visibility": visibility,
                "feat": None,
            }
            annotations_by_frame[frame_id].append(annotation)
            annotations_by_track[track_id].append(annotation)
    for track_annotations in annotations_by_track.values():
        track_annotations.sort(key=lambda ann: ann["frame_id"])
    return annotations_by_frame, annotations_by_track


def trim_annotations(annotations_by_frame, annotations_by_track, max_frames):
    if max_frames <= 0:
        return annotations_by_frame, annotations_by_track

    kept_frames = set(sorted(annotations_by_frame.keys())[:max_frames])
    trimmed_by_frame = defaultdict(list)
    for frame_id in sorted(kept_frames):
        trimmed_by_frame[frame_id] = annotations_by_frame[frame_id]

    trimmed_by_track = defaultdict(list)
    for track_id, annotations in annotations_by_track.items():
        kept_annotations = [ann for ann in annotations if ann["frame_id"] in kept_frames]
        if kept_annotations:
            trimmed_by_track[track_id] = kept_annotations
    return trimmed_by_frame, trimmed_by_track


def extract_sequence_features(seq_meta, annotations_by_frame, extractor):
    """
    对一个序列中所有有效 GT 框提取 OSNet 特征。

    temporal matcher 的训练样本不是直接使用图像，而是使用 ReID embedding。
    因此在真正构造正负样本前，需要先把每个标注框的 feat 字段补齐。
    """
    # 先逐帧跑 OSNet，把每个 GT 行人框的外观特征补齐。
    for frame_id in sorted(annotations_by_frame.keys()):
        image_path = os.path.join(seq_meta["image_dir"], "{:06d}{}".format(frame_id, seq_meta["image_ext"]))
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError("failed to read frame: {}".format(image_path))
        frame_annotations = annotations_by_frame[frame_id]
        tlbrs = [np.r_[ann["tlwh"][:2], ann["tlwh"][:2] + ann["tlwh"][2:]] for ann in frame_annotations]
        features = extractor(image, tlbrs)
        for annotation, feature in zip(frame_annotations, features):
            annotation["feat"] = feature.astype(np.float32)


def select_hard_negatives(frame_annotations, positive_track_id, history_mean, limit):
    """
    在同一帧中选择 hard negative。

    hard negative 指“不是同一 ID，但外观特征和当前历史轨迹很像”的人。
    这类样本对解决近距离相似行人 ID 互换更有帮助。
    """
    # 负样本优先选“看起来很像但实际不是同一人”的 nearby hard negatives。
    if limit <= 0:
        return []

    negatives = []
    for annotation in frame_annotations:
        if annotation["track_id"] == positive_track_id:
            continue
        feat = annotation.get("feat")
        if feat is None or np.linalg.norm(feat) == 0:
            continue
        similarity = float(np.dot(history_mean, feat / max(np.linalg.norm(feat), 1e-6)))
        negatives.append((similarity, annotation))
    negatives.sort(key=lambda item: item[0], reverse=True)
    return [annotation for _, annotation in negatives[:limit]]


def build_samples(args):
    # 这里同时完成：
    # 1. 逐序列提 ReID 特征
    # 2. 生成正/负时序匹配样本
    # 3. 保存成可直接 torch.load 的训练张量
    mot_train_dir = os.path.join(args.mot_root, "train")
    if not os.path.isdir(mot_train_dir):
        raise FileNotFoundError("MOT train directory not found: {}".format(mot_train_dir))

    sequence_names = sorted(
        [
            name
            for name in os.listdir(mot_train_dir)
            if os.path.isdir(os.path.join(mot_train_dir, name))
            and (not args.sequence_suffix or name.endswith(args.sequence_suffix))
        ]
    )
    if args.max_sequences > 0:
        sequence_names = sequence_names[: args.max_sequences]
    if not sequence_names:
        raise RuntimeError("no MOT sequences matched under {}".format(mot_train_dir))

    extractor = OSNetReIDExtractor(
        model_path=args.reid_model,
        device=args.device,
        batch_size=args.reid_batch_size,
    )

    history_feat_list = []
    history_geom_list = []
    history_mask_list = []
    candidate_feat_list = []
    candidate_geom_list = []
    label_list = []
    sequence_list = []
    gap_list = []
    positive_count = 0

    for sequence_name in sequence_names:
        seq_dir = os.path.join(mot_train_dir, sequence_name)
        seq_meta = parse_seqinfo(seq_dir)
        logger.info("Processing sequence {}", sequence_name)
        annotations_by_frame, annotations_by_track = load_annotations(seq_dir, min_visibility=args.min_visibility)
        annotations_by_frame, annotations_by_track = trim_annotations(
            annotations_by_frame,
            annotations_by_track,
            max_frames=args.max_frames,
        )
        extract_sequence_features(seq_meta, annotations_by_frame, extractor)

        for track_id, track_annotations in annotations_by_track.items():
            valid_annotations = [
                ann for ann in track_annotations if ann["feat"] is not None and np.linalg.norm(ann["feat"]) > 0
            ]
            if len(valid_annotations) < args.min_history + 1:
                continue

            for candidate_index in range(args.min_history, len(valid_annotations)):
                candidate = valid_annotations[candidate_index]
                valid_history_ends = [
                    history_end
                    for history_end in range(candidate_index)
                    if history_end + 1 >= args.min_history
                    and args.min_gap <= candidate["frame_id"] - valid_annotations[history_end]["frame_id"] <= args.max_gap
                ]
                if not valid_history_ends:
                    continue

                history_end = valid_history_ends[-1]
                history_annotations = valid_annotations[max(0, history_end - args.history_len + 1) : history_end + 1]
                if len(history_annotations) < args.min_history:
                    continue

                # 正样本：同一个 track_id 的历史片段与未来候选框。
                sample = build_temporal_pair_inputs(
                    history_features=[ann["feat"] for ann in history_annotations],
                    history_boxes=[ann["tlwh"] for ann in history_annotations],
                    history_frames=[ann["frame_id"] for ann in history_annotations],
                    candidate_feature=candidate["feat"],
                    candidate_tlwh=candidate["tlwh"],
                    candidate_frame=candidate["frame_id"],
                    image_size=(seq_meta["height"], seq_meta["width"]),
                    max_history=args.history_len,
                    max_dt=args.max_gap,
                    min_history=args.min_history,
                )
                if sample is None:
                    continue

                # 用历史特征均值代表这条轨迹的近期外观，再去同帧里找最像它的其他人。
                history_mean = np.mean(np.stack([ann["feat"] for ann in history_annotations], axis=0), axis=0)
                history_mean = history_mean / max(np.linalg.norm(history_mean), 1e-6)
                negatives = select_hard_negatives(
                    annotations_by_frame[candidate["frame_id"]],
                    positive_track_id=track_id,
                    history_mean=history_mean,
                    limit=args.negatives_per_positive,
                )

                history_feat_list.append(sample["history_feat"].astype(np.float32))
                history_geom_list.append(sample["history_geom"].astype(np.float32))
                history_mask_list.append(sample["history_mask"].astype(bool))
                candidate_feat_list.append(sample["candidate_feat"].astype(np.float32))
                candidate_geom_list.append(sample["candidate_geom"].astype(np.float32))
                label_list.append(1.0)
                sequence_list.append(sequence_name)
                gap_list.append(candidate["frame_id"] - history_annotations[-1]["frame_id"])
                positive_count += 1

                # 负样本：候选框来自其他 track_id，用 label=0 训练模型识别“不是同一人”。
                for negative in negatives:
                    negative_sample = build_temporal_pair_inputs(
                        history_features=[ann["feat"] for ann in history_annotations],
                        history_boxes=[ann["tlwh"] for ann in history_annotations],
                        history_frames=[ann["frame_id"] for ann in history_annotations],
                        candidate_feature=negative["feat"],
                        candidate_tlwh=negative["tlwh"],
                        candidate_frame=candidate["frame_id"],
                        image_size=(seq_meta["height"], seq_meta["width"]),
                        max_history=args.history_len,
                        max_dt=args.max_gap,
                        min_history=args.min_history,
                    )
                    if negative_sample is None:
                        continue
                    history_feat_list.append(negative_sample["history_feat"].astype(np.float32))
                    history_geom_list.append(negative_sample["history_geom"].astype(np.float32))
                    history_mask_list.append(negative_sample["history_mask"].astype(bool))
                    candidate_feat_list.append(negative_sample["candidate_feat"].astype(np.float32))
                    candidate_geom_list.append(negative_sample["candidate_geom"].astype(np.float32))
                    label_list.append(0.0)
                    sequence_list.append(sequence_name)
                    gap_list.append(candidate["frame_id"] - history_annotations[-1]["frame_id"])

                if args.max_positives > 0 and positive_count >= args.max_positives:
                    logger.info("Reached max positive sample cap: {}", args.max_positives)
                    break

            if args.max_positives > 0 and positive_count >= args.max_positives:
                break

        if args.max_positives > 0 and positive_count >= args.max_positives:
            break

    if not history_feat_list:
        raise RuntimeError("no temporal training samples were generated")

    dataset = {
        "history_feat": torch.as_tensor(np.asarray(history_feat_list), dtype=torch.float32),
        "history_geom": torch.as_tensor(np.asarray(history_geom_list), dtype=torch.float32),
        "history_mask": torch.as_tensor(np.asarray(history_mask_list), dtype=torch.bool),
        "candidate_feat": torch.as_tensor(np.asarray(candidate_feat_list), dtype=torch.float32),
        "candidate_geom": torch.as_tensor(np.asarray(candidate_geom_list), dtype=torch.float32),
        "labels": torch.as_tensor(np.asarray(label_list), dtype=torch.float32),
        "sequence_names": sequence_list,
        "gaps": torch.as_tensor(np.asarray(gap_list), dtype=torch.int64),
        "config": {
            "history_len": args.history_len,
            "min_history": args.min_history,
            "max_gap": args.max_gap,
            "min_gap": args.min_gap,
            "sequence_suffix": args.sequence_suffix,
            "min_visibility": args.min_visibility,
        },
    }
    return dataset


def main():
    args = make_parser().parse_args()
    logger.info("Args: {}", args)
    dataset = build_samples(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, output_path)
    positives = int(dataset["labels"].sum().item())
    negatives = int(dataset["labels"].numel() - positives)
    logger.info("Saved temporal matcher dataset to {}", output_path)
    logger.info(
        "Samples: total={}, positives={}, negatives={}",
        int(dataset["labels"].numel()),
        positives,
        negatives,
    )


if __name__ == "__main__":
    main()
