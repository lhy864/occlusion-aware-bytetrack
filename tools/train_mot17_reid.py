#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
OSNet 在 MOT17 ReID 数据集上的微调脚本。

训练目标是让外观特征更适合当前跟踪任务里的行人外观区分，
尤其是近距离遮挡、重现和 ID 保持这些场景。
"""

import argparse
import json
import math
import random
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from loguru import logger
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

warnings.filterwarnings(
    "ignore",
    message="Cython evaluation .* unavailable, now use python evaluation\\.",
    module=r"torchreid\.reid\.metrics\.rank",
)

from torchreid import losses
from torchreid.reid.models import build_model

from yolox.tracker.reid_extractor import _unwrap_reid_state_dict, _ensure_reid_checkpoint
from yolox.utils.cli_config import apply_config_defaults


def make_parser():
    parser = argparse.ArgumentParser("Fine-tune OSNet on MOT17 ReID crops")
    parser.add_argument(
        "--data-root",
        default="datasets/reid_mot17",
        type=str,
        help="root directory produced by tools/build_mot17_reid_dataset.py",
    )
    parser.add_argument(
        "--train-split",
        default="train",
        type=str,
        help="dataset split used for training",
    )
    parser.add_argument(
        "--val-split",
        default="val",
        type=str,
        help="dataset split used for validation",
    )
    parser.add_argument(
        "--pretrained",
        default="pretrained/osnet_x0_25_msmt17.pt",
        type=str,
        help="source checkpoint used to initialize the OSNet backbone",
    )
    parser.add_argument(
        "--output",
        default="pretrained/osnet_x0_25_mot17.pth",
        type=str,
        help="inference-ready fine-tuned weights path",
    )
    parser.add_argument(
        "--work-dir",
        default="",
        type=str,
        help="directory for logs and training checkpoints, defaults to <output_stem>_workdir",
    )
    parser.add_argument(
        "--backbone",
        default="osnet_x0_25",
        type=str,
        help="torchreid backbone name",
    )
    parser.add_argument(
        "--image-height",
        default=256,
        type=int,
        help="input image height",
    )
    parser.add_argument(
        "--image-width",
        default=128,
        type=int,
        help="input image width",
    )
    parser.add_argument(
        "--batch-size",
        default=64,
        type=int,
        help="training batch size",
    )
    parser.add_argument(
        "--instances-per-identity",
        default=4,
        type=int,
        help="number of instances per identity inside one batch",
    )
    parser.add_argument(
        "--epochs",
        default=40,
        type=int,
        help="number of fine-tuning epochs",
    )
    parser.add_argument(
        "--lr",
        default=3e-4,
        type=float,
        help="classifier learning rate",
    )
    parser.add_argument(
        "--backbone-lr-mult",
        default=0.1,
        type=float,
        help="backbone learning-rate multiplier relative to --lr",
    )
    parser.add_argument(
        "--weight-decay",
        default=5e-4,
        type=float,
        help="optimizer weight decay",
    )
    parser.add_argument(
        "--margin",
        default=0.3,
        type=float,
        help="triplet loss margin",
    )
    parser.add_argument(
        "--label-smoothing",
        default=0.1,
        type=float,
        help="cross-entropy label smoothing epsilon",
    )
    parser.add_argument(
        "--ce-weight",
        default=1.0,
        type=float,
        help="weight for cross-entropy loss",
    )
    parser.add_argument(
        "--triplet-weight",
        default=1.0,
        type=float,
        help="weight for triplet loss",
    )
    parser.add_argument(
        "--num-workers",
        default=4,
        type=int,
        help="dataloader worker count",
    )
    parser.add_argument(
        "--eval-every",
        default=1,
        type=int,
        help="run retrieval validation every N epochs",
    )
    parser.add_argument(
        "--skip-val",
        default=False,
        action="store_true",
        help="skip validation entirely and only save last-epoch checkpoints",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="random seed",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        help="training device, for example cuda, cuda:0 or cpu",
    )
    parser.add_argument(
        "--amp",
        default=False,
        action="store_true",
        help="enable torch.cuda.amp when using CUDA",
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        help="optional training checkpoint produced by this script",
    )
    parser.add_argument(
        "--eval-only",
        default=False,
        action="store_true",
        help="skip training and only evaluate the provided weights",
    )
    parser.add_argument(
        "--max-train-samples",
        default=0,
        type=int,
        help="optional cap on training samples for smoke tests",
    )
    parser.add_argument(
        "--max-val-samples",
        default=0,
        type=int,
        help="optional cap on validation samples for smoke tests",
    )
    return apply_config_defaults(parser)


def normalize_device(device):
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_jsonl(path):
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


class MOT17ReIDDataset(Dataset):
    """
    MOT17 ReID 裁剪图数据集。

    每个样本返回：
    - image：裁剪后的行人图，经过 resize、增强和 normalize。
    - label：当前 split 内连续的身份标签。
    - record：metadata 中的原始信息，验证和误差分析时会用到。
    """

    def __init__(self, data_root, split, image_height, image_width, training, max_samples=0):
        self.data_root = Path(data_root)
        self.split = split
        self.training = training
        metadata_path = self.data_root / split / "metadata.jsonl"
        if not metadata_path.is_file():
            raise FileNotFoundError("metadata file not found: {}".format(metadata_path))

        records = load_jsonl(metadata_path)
        if max_samples > 0:
            records = records[:max_samples]
        if not records:
            raise RuntimeError("No records found for split `{}` under {}".format(split, self.data_root))

        self.records = records
        self.labels = [int(record["split_pid"]) - 1 for record in self.records]
        self.num_classes = max(self.labels) + 1
        self.pid_to_indices = defaultdict(list)
        for index, label in enumerate(self.labels):
            self.pid_to_indices[label].append(index)

        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        resize = transforms.Resize((image_height, image_width), interpolation=InterpolationMode.BILINEAR)
        if training:
            self.transform = transforms.Compose(
                [
                    resize,
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.03),
                    transforms.ToTensor(),
                    normalize,
                    transforms.RandomErasing(p=0.4, scale=(0.02, 0.2), ratio=(0.3, 3.3), value="random"),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    resize,
                    transforms.ToTensor(),
                    normalize,
                ]
            )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image_path = self.data_root / record["export_image_rel"]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), self.labels[index], index


class IdentityBatchSampler(Sampler):
    """
    ReID 常用的身份采样器。

    一个 batch 不是随机抽 batch_size 张图，而是：
    - 先抽若干个身份。
    - 每个身份抽 instances_per_identity 张图。

    这样做是为了让 triplet loss 在一个 batch 内更容易找到正样本和负样本。
    """

    def __init__(self, dataset, batch_size, instances_per_identity, seed=42):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if instances_per_identity <= 0:
            raise ValueError("instances_per_identity must be positive")
        if batch_size % instances_per_identity != 0:
            raise ValueError("batch_size must be divisible by instances_per_identity")

        self.dataset = dataset
        self.batch_size = batch_size
        self.instances_per_identity = instances_per_identity
        self.identities_per_batch = batch_size // instances_per_identity
        self.pid_list = sorted(dataset.pid_to_indices.keys())
        if not self.pid_list:
            raise RuntimeError("IdentityBatchSampler received an empty pid list")
        self.num_batches = max(1, math.ceil(len(dataset) / float(batch_size)))
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.num_batches * self.batch_size

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        indices = []
        shuffled_pids = self.pid_list[:]
        rng.shuffle(shuffled_pids)
        pid_cursor = 0

        for _ in range(self.num_batches):
            if len(self.pid_list) >= self.identities_per_batch:
                if pid_cursor + self.identities_per_batch > len(shuffled_pids):
                    shuffled_pids = self.pid_list[:]
                    rng.shuffle(shuffled_pids)
                    pid_cursor = 0
                batch_pids = shuffled_pids[pid_cursor : pid_cursor + self.identities_per_batch]
                pid_cursor += self.identities_per_batch
            else:
                batch_pids = [rng.choice(self.pid_list) for _ in range(self.identities_per_batch)]

            for pid in batch_pids:
                candidate_indices = self.dataset.pid_to_indices[pid]
                if len(candidate_indices) >= self.instances_per_identity:
                    sampled_indices = rng.sample(candidate_indices, self.instances_per_identity)
                else:
                    sampled_indices = [rng.choice(candidate_indices) for _ in range(self.instances_per_identity)]
                indices.extend(sampled_indices)

        return iter(indices)


def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


def strip_state_dict_prefix(state_dict):
    normalized = {}
    for key, value in state_dict.items():
        normalized_key = key[7:] if key.startswith("module.") else key
        normalized[normalized_key] = value
    return normalized


def load_partial_pretrained(model, checkpoint_path):
    resolved_checkpoint_path = _ensure_reid_checkpoint(checkpoint_path)
    checkpoint = torch.load(resolved_checkpoint_path, map_location="cpu")
    state_dict = _unwrap_reid_state_dict(checkpoint, str(resolved_checkpoint_path))
    model_state = model.state_dict()
    compatible_state = {}
    skipped = []

    for key, value in state_dict.items():
        if key.startswith("classifier."):
            skipped.append(key)
            continue
        if key not in model_state:
            skipped.append(key)
            continue
        if model_state[key].shape != value.shape:
            skipped.append(key)
            continue
        compatible_state[key] = value

    missing, unexpected = model.load_state_dict(compatible_state, strict=False)
    return {
        "resolved_checkpoint_path": str(resolved_checkpoint_path),
        "loaded_keys": len(compatible_state),
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "skipped_keys": len(skipped),
    }


def build_optimizer(model, args):
    backbone_params = []
    classifier_params = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("classifier."):
            classifier_params.append(parameter)
        else:
            backbone_params.append(parameter)

    parameter_groups = []
    if backbone_params:
        parameter_groups.append({"params": backbone_params, "lr": args.lr * args.backbone_lr_mult})
    if classifier_params:
        parameter_groups.append({"params": classifier_params, "lr": args.lr})

    return torch.optim.AdamW(parameter_groups, lr=args.lr, weight_decay=args.weight_decay)


def build_dataloaders(args, device):
    """
    构建训练/验证 dataloader。

    如果 args.skip_val=True，则只构建训练集，适合显存/内存较紧张时使用。
    之前 RTX4060 训练时卡在大规模验证，这个开关就是为了解决那个问题。
    """
    train_dataset = MOT17ReIDDataset(
        data_root=args.data_root,
        split=args.train_split,
        image_height=args.image_height,
        image_width=args.image_width,
        training=True,
        max_samples=args.max_train_samples,
    )
    train_sampler = IdentityBatchSampler(
        dataset=train_dataset,
        batch_size=args.batch_size,
        instances_per_identity=args.instances_per_identity,
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        worker_init_fn=worker_init_fn,
    )
    if args.skip_val:
        val_dataset = None
        val_loader = None
    else:
        val_dataset = MOT17ReIDDataset(
            data_root=args.data_root,
            split=args.val_split,
            image_height=args.image_height,
            image_width=args.image_width,
            training=False,
            max_samples=args.max_val_samples,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=max(args.batch_size, 1),
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            worker_init_fn=worker_init_fn,
        )
    return train_dataset, val_dataset, train_sampler, train_loader, val_loader


def build_model_for_training(args, num_classes, device):
    # 训练时需要把分类头替换成当前数据集的身份数，
    # 再尽量从预训练权重中加载 backbone 和可复用层。
    model = build_model(
        name=args.backbone,
        num_classes=num_classes,
        loss="triplet",
        pretrained=False,
        use_gpu=device.type == "cuda",
    )
    init_info = load_partial_pretrained(model, args.pretrained)
    logger.info(
        "Initialized `{}` from {} with {} matching keys (skipped {}, missing {})",
        args.backbone,
        init_info["resolved_checkpoint_path"],
        init_info["loaded_keys"],
        init_info["skipped_keys"],
        init_info["missing_keys"],
    )
    model.to(device)
    return model


def train_one_epoch(model, train_loader, optimizer, ce_criterion, triplet_criterion, scaler, device, epoch, args):
    # 一个 epoch 内同时优化分类损失和 triplet 损失。
    model.train()
    if hasattr(train_loader.sampler, "set_epoch"):
        train_loader.sampler.set_epoch(epoch)

    total_loss = 0.0
    total_ce = 0.0
    total_triplet = 0.0
    total_correct = 0
    total_samples = 0
    optimizer_steps = 0
    autocast_enabled = scaler is not None and scaler.is_enabled()

    for images, labels, _ in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=autocast_enabled):
            outputs = model(images)
            if not isinstance(outputs, tuple) or len(outputs) != 2:
                raise RuntimeError("Expected (logits, features) from training-time OSNet forward pass")
            logits, features = outputs
            loss_ce = ce_criterion(logits, labels)

        # triplet hard mining 对数值更敏感，这里统一回到 fp32 计算。
        loss_triplet = triplet_criterion(features.float(), labels)
        loss = args.ce_weight * loss_ce + args.triplet_weight * loss_triplet

        if scaler is not None:
            scale_before = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= scale_before:
                optimizer._opt_called = True
                optimizer_steps += 1
        else:
            loss.backward()
            optimizer.step()
            optimizer_steps += 1

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_ce += float(loss_ce.item()) * batch_size
        total_triplet += float(loss_triplet.item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_samples += batch_size

    return {
        "loss": total_loss / max(total_samples, 1),
        "loss_ce": total_ce / max(total_samples, 1),
        "loss_triplet": total_triplet / max(total_samples, 1),
        "acc": 100.0 * total_correct / max(total_samples, 1),
        "optimizer_steps": optimizer_steps,
    }


def extract_embeddings(model, data_loader, device):
    # 验证时只抽特征，不需要分类 logits。
    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels, _ in data_loader:
            images = images.to(device, non_blocking=True)
            features = model(images)
            features = F.normalize(features, dim=1)
            all_features.append(features.cpu())
            all_labels.append(labels.cpu())

    if not all_features:
        return torch.zeros((0, 512), dtype=torch.float32), torch.zeros((0,), dtype=torch.long)
    return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)


def compute_retrieval_metrics(features, labels, records):
    # 这里评估的是 ReID 检索质量，而不是跟踪指标。
    # top1 / mAP 越高，说明“同一身份在特征空间里更容易被拉近”。
    if features.numel() == 0 or labels.numel() == 0:
        return {
            "num_queries": 0,
            "top1": 0.0,
            "mAP": 0.0,
            "mean_best_positive_sim": 0.0,
            "mean_nearest_negative_sim": 0.0,
            "mean_similarity_margin": 0.0,
            "mean_nearby_negative_sim": 0.0,
        }

    similarity = torch.matmul(features, features.t())
    label_array = labels.numpy()
    query_count = 0
    ap_values = []
    top1_values = []
    best_positive_sims = []
    nearest_negative_sims = []
    similarity_margins = []

    for index in range(len(records)):
        same_identity = label_array == label_array[index]
        same_identity[index] = False
        if not same_identity.any():
            continue

        valid_mask = np.ones(len(records), dtype=bool)
        valid_mask[index] = False
        scores = similarity[index][valid_mask]
        matches = torch.from_numpy(same_identity[valid_mask])
        ranking = torch.argsort(scores, descending=True)
        sorted_matches = matches[ranking]

        query_count += 1
        top1_values.append(float(sorted_matches[0].item()))

        positive_scores = scores[matches]
        negative_scores = scores[~matches]
        if positive_scores.numel() > 0:
            best_positive_sims.append(float(positive_scores.max().item()))
        if negative_scores.numel() > 0:
            nearest_negative_sims.append(float(negative_scores.max().item()))
        if positive_scores.numel() > 0 and negative_scores.numel() > 0:
            similarity_margins.append(float(positive_scores.max().item() - negative_scores.max().item()))

        hit_count = 0
        precision_sum = 0.0
        for rank, is_match in enumerate(sorted_matches.tolist(), start=1):
            if is_match:
                hit_count += 1
                precision_sum += hit_count / rank
        ap_values.append(precision_sum / max(int(matches.sum().item()), 1))

    lookup = {}
    for idx, record in enumerate(records):
        lookup[(record["sequence_name"], record["source_frame_id"], int(record["split_pid"]))] = idx
    nearby_negative_sims = []
    for idx, record in enumerate(records):
        for neighbor in record.get("nearby_different_ids", []):
            key = (record["sequence_name"], record["source_frame_id"], int(neighbor["split_pid"]))
            other_idx = lookup.get(key)
            if other_idx is None:
                continue
            nearby_negative_sims.append(float(similarity[idx, other_idx].item()))

    return {
        "num_queries": query_count,
        "top1": 100.0 * float(np.mean(top1_values)) if top1_values else 0.0,
        "mAP": 100.0 * float(np.mean(ap_values)) if ap_values else 0.0,
        "mean_best_positive_sim": float(np.mean(best_positive_sims)) if best_positive_sims else 0.0,
        "mean_nearest_negative_sim": float(np.mean(nearest_negative_sims)) if nearest_negative_sims else 0.0,
        "mean_similarity_margin": float(np.mean(similarity_margins)) if similarity_margins else 0.0,
        "mean_nearby_negative_sim": float(np.mean(nearby_negative_sims)) if nearby_negative_sims else 0.0,
    }


def validate(model, val_loader, val_dataset, device):
    features, labels = extract_embeddings(model, val_loader, device)
    return compute_retrieval_metrics(features, labels, val_dataset.records)


def checkpoint_score(metrics):
    return metrics["mAP"] + 0.2 * metrics["top1"] + 10.0 * metrics["mean_similarity_margin"]


def save_training_state(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    best_metrics,
    history,
    args,
):
    state = {
        "epoch": epoch,
        "state_dict": strip_state_dict_prefix(model.state_dict()),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "best_metrics": best_metrics,
        "history": history,
        "args": vars(args),
    }
    torch.save(state, path)


def load_training_state(path, model, optimizer, scheduler, scaler, device):
    checkpoint = torch.load(path, map_location=device)
    state_dict = _unwrap_reid_state_dict(checkpoint, path)
    model.load_state_dict(state_dict, strict=True)
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    return {
        "epoch": int(checkpoint.get("epoch", -1)),
        "best_metrics": checkpoint.get("best_metrics", {}),
        "history": checkpoint.get("history", []),
    }


def ensure_work_dir(args):
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.work_dir:
        work_dir = Path(args.work_dir)
    else:
        work_dir = output_path.parent / "{}_workdir".format(output_path.stem)
    work_dir.mkdir(parents=True, exist_ok=True)
    return output_path, work_dir


def main():
    args = make_parser().parse_args()
    if args.eval_only and args.skip_val:
        raise ValueError("--eval-only cannot be used together with --skip-val")
    device = normalize_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")
    output_path, work_dir = ensure_work_dir(args)
    history_path = work_dir / "history.json"
    best_ckpt_path = work_dir / "best.ckpt"
    last_ckpt_path = work_dir / "last.ckpt"
    metrics_path = work_dir / "best_metrics.json"

    set_random_seed(args.seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    logger.info("Args: {}", args)
    train_dataset, val_dataset, train_sampler, train_loader, val_loader = build_dataloaders(args, device)
    logger.info(
        "Train split `{}`: {} samples, {} identities{}",
        args.train_split,
        len(train_dataset),
        train_dataset.num_classes,
        (
            " | Val split `{}`: {} samples, {} identities".format(
                args.val_split,
                len(val_dataset),
                val_dataset.num_classes,
            )
            if val_dataset is not None
            else " | Validation skipped"
        ),
    )

    model = build_model_for_training(args, train_dataset.num_classes, device)
    optimizer = build_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    ce_criterion = losses.CrossEntropyLoss(
        num_classes=train_dataset.num_classes,
        eps=args.label_smoothing,
        use_gpu=device.type == "cuda",
        label_smooth=args.label_smoothing > 0,
    )
    triplet_criterion = losses.TripletLoss(margin=args.margin)
    scaler = (
        torch.amp.GradScaler(device=device.type, enabled=True, init_scale=1024.0)
        if use_amp
        else None
    )

    start_epoch = 0
    best_metrics = {}
    history = []
    if args.resume:
        logger.info("Resuming training state from {}", args.resume)
        resume_state = load_training_state(args.resume, model, optimizer, scheduler, scaler, device)
        start_epoch = resume_state["epoch"] + 1
        best_metrics = resume_state["best_metrics"]
        history = resume_state["history"]

    if args.eval_only:
        eval_metrics = validate(model, val_loader, val_dataset, device)
        logger.info("Eval only metrics: {}", eval_metrics)
        metrics_path.write_text(json.dumps(eval_metrics, indent=2), encoding="utf-8")
        return

    best_score = checkpoint_score(best_metrics) if best_metrics else float("-inf")

    for epoch in range(start_epoch, args.epochs):
        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            ce_criterion=ce_criterion,
            triplet_criterion=triplet_criterion,
            scaler=scaler,
            device=device,
            epoch=epoch,
            args=args,
        )

        epoch_record = {
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
        }
        logger.info(
            "Epoch {}/{} | train_loss={:.4f} ce={:.4f} tri={:.4f} acc={:.2f}",
            epoch + 1,
            args.epochs,
            train_metrics["loss"],
            train_metrics["loss_ce"],
            train_metrics["loss_triplet"],
            train_metrics["acc"],
        )

        if train_metrics["optimizer_steps"] > 0:
            scheduler.step()
        else:
            logger.warning("Epoch {} completed without an optimizer step, scheduler step skipped", epoch + 1)

        run_validation = (
            not args.skip_val
            and ((epoch + 1) % max(args.eval_every, 1) == 0 or epoch + 1 == args.epochs)
        )
        if run_validation:
            val_metrics = validate(model, val_loader, val_dataset, device)
            epoch_record["val"] = val_metrics
            logger.info(
                "Epoch {}/{} | val_top1={:.2f} val_mAP={:.2f} margin={:.4f} nearby_neg={:.4f}",
                epoch + 1,
                args.epochs,
                val_metrics["top1"],
                val_metrics["mAP"],
                val_metrics["mean_similarity_margin"],
                val_metrics["mean_nearby_negative_sim"],
            )
            current_score = checkpoint_score(val_metrics)
            if current_score > best_score:
                best_score = current_score
                best_metrics = dict(val_metrics)
                torch.save(strip_state_dict_prefix(model.state_dict()), output_path)
                save_training_state(
                    best_ckpt_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    best_metrics=best_metrics,
                    history=history + [epoch_record],
                    args=args,
                )
                metrics_path.write_text(json.dumps(best_metrics, indent=2), encoding="utf-8")
                logger.info("New best checkpoint saved to {}", output_path)
        else:
            torch.save(strip_state_dict_prefix(model.state_dict()), output_path)
            logger.info("Validation skipped, updated latest inference weights at {}", output_path)

        history.append(epoch_record)
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        save_training_state(
            last_ckpt_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_metrics=best_metrics,
            history=history,
            args=args,
        )

    if not output_path.is_file():
        torch.save(strip_state_dict_prefix(model.state_dict()), output_path)
        if args.skip_val:
            logger.warning("Validation was skipped, saved last epoch weights to {}", output_path)
        else:
            logger.warning("No validation checkpoint beat the baseline, saved last epoch weights to {}", output_path)

    logger.info("Finished training. Best inference weights: {}", output_path.resolve())
    if best_metrics:
        logger.info("Best metrics: {}", best_metrics)


if __name__ == "__main__":
    main()
