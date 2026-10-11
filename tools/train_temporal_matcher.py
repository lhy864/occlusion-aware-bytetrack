#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
训练阶段三/阶段七使用的时序关联 Transformer。

这个脚本训练出的不是新的检测器，也不是新的跟踪器主体，
而是一个“轨迹-候选框匹配打分器”。
"""

import argparse
import math
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader, Dataset, random_split

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from yolox.tracker.temporal_matcher import TemporalAssociationTransformer


class TemporalPairDataset(Dataset):
    """
    temporal matcher 的张量数据集。

    每个样本都是一个二分类样本：
    - history_* 表示某条轨迹过去 K 次观测。
    - candidate_* 表示当前候选检测框。
    - label=1 表示同一 ID，label=0 表示不同 ID。
    """

    def __init__(self, data):
        # 训练数据已经在 build_temporal_matcher_dataset.py 中整理成张量，
        # 这里直接按字段读取即可。
        self.history_feat = data["history_feat"].float()
        self.history_geom = data["history_geom"].float()
        self.history_mask = data["history_mask"].bool()
        self.candidate_feat = data["candidate_feat"].float()
        self.candidate_geom = data["candidate_geom"].float()
        self.labels = data["labels"].float()

    def __len__(self):
        return int(self.labels.shape[0])

    def __getitem__(self, index):
        return {
            "history_feat": self.history_feat[index],
            "history_geom": self.history_geom[index],
            "history_mask": self.history_mask[index],
            "candidate_feat": self.candidate_feat[index],
            "candidate_geom": self.candidate_geom[index],
            "label": self.labels[index],
        }


def make_parser():
    parser = argparse.ArgumentParser("Train temporal matcher")
    parser.add_argument(
        "--samples",
        default="datasets/temporal_matcher_mot17_train.pt",
        type=str,
        help="temporal training samples generated from MOT17",
    )
    parser.add_argument(
        "--output",
        default="pretrained/temporal_matcher_mot17.pth",
        type=str,
        help="checkpoint path for the trained temporal matcher",
    )
    parser.add_argument("--device", default="cuda", type=str, help="training device")
    parser.add_argument("--batch-size", default=256, type=int, help="training batch size")
    parser.add_argument("--epochs", default=15, type=int, help="number of training epochs")
    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
    parser.add_argument("--weight-decay", default=1e-4, type=float, help="weight decay")
    parser.add_argument("--val-ratio", default=0.1, type=float, help="validation ratio")
    parser.add_argument("--seed", default=42, type=int, help="random seed")
    parser.add_argument("--d-model", default=256, type=int, help="Transformer hidden size")
    parser.add_argument("--nhead", default=8, type=int, help="number of attention heads")
    parser.add_argument("--num-layers", default=2, type=int, help="number of Transformer encoder layers")
    parser.add_argument("--dim-feedforward", default=512, type=int, help="feedforward hidden size")
    parser.add_argument("--dropout", default=0.1, type=float, help="dropout ratio")
    return parser


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch, device):
    """
    将一个 batch 的所有张量移动到目标设备。

    因为 batch 是 dict 格式，所以不能直接 batch.to(device)，需要逐字段移动。
    """
    return {key: value.to(device) for key, value in batch.items()}


def run_epoch(model, loader, optimizer, criterion, device, train):
    """
    运行一个训练或验证 epoch。

    当 train=True：
    - 反向传播并更新参数。

    当 train=False：
    - 只前向推理，用于计算验证 loss / accuracy / precision / recall / f1。
    """
    # 这是标准的二分类训练：
    # label=1 表示候选框属于这条轨迹，label=0 表示不属于。
    model.train(train)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_positive = 0
    total_pred_positive = 0
    total_true_positive = 0

    for batch in loader:
        batch = move_batch(batch, device)
        logits = model(
            batch["history_feat"],
            batch["history_geom"],
            batch["history_mask"],
            batch["candidate_feat"],
            batch["candidate_geom"],
        )
        loss = criterion(logits, batch["label"])

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        labels = batch["label"]

        total_loss += float(loss.item()) * labels.numel()
        total_correct += int((preds == labels).sum().item())
        total_samples += int(labels.numel())
        total_positive += int(labels.sum().item())
        total_pred_positive += int(preds.sum().item())
        total_true_positive += int(((preds == 1.0) & (labels == 1.0)).sum().item())

    avg_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)
    precision = total_true_positive / max(total_pred_positive, 1)
    recall = total_true_positive / max(total_positive, 1)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    args = make_parser().parse_args()
    logger.info("Args: {}", args)
    set_seed(args.seed)

    data = torch.load(args.samples, map_location="cpu")
    dataset = TemporalPairDataset(data)
    if len(dataset) < 10:
        raise RuntimeError("temporal dataset is too small: {}".format(len(dataset)))

    val_size = max(1, int(math.ceil(len(dataset) * args.val_ratio)))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    feat_dim = int(data["history_feat"].shape[-1])
    max_history = int(data["history_feat"].shape[1])
    max_dt = int(data.get("config", {}).get("max_gap", max(30, max_history)))
    min_history = int(data.get("config", {}).get("min_history", 3))

    # 模型结构参数一部分来自命令行，一部分来自样本张量本身的尺寸。
    model = TemporalAssociationTransformer(
        feat_dim=feat_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_history=max_history,
        max_dt=max_dt,
        min_history=min_history,
    ).to(device)

    labels = dataset.labels
    positive_count = float(labels.sum().item())
    negative_count = float(labels.numel() - positive_count)
    pos_weight = torch.tensor(
        [negative_count / max(positive_count, 1.0)],
        dtype=torch.float32,
        device=device,
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1 = -1.0
    best_metrics = None
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, criterion, device, train=True)
        val_metrics = run_epoch(model, val_loader, optimizer, criterion, device, train=False)
        scheduler.step()

        logger.info(
            "Epoch {}/{} | train loss {:.4f} acc {:.4f} f1 {:.4f} | val loss {:.4f} acc {:.4f} f1 {:.4f}",
            epoch,
            args.epochs,
            train_metrics["loss"],
            train_metrics["accuracy"],
            train_metrics["f1"],
            val_metrics["loss"],
            val_metrics["accuracy"],
            val_metrics["f1"],
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_metrics = {
                "train": train_metrics,
                "val": val_metrics,
                "epoch": epoch,
            }
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": model.get_config(),
                    "metrics": best_metrics,
                },
                output_path,
            )
            logger.info("Saved best checkpoint to {}", output_path)

    logger.info("Training finished. Best metrics: {}", best_metrics)


if __name__ == "__main__":
    main()
