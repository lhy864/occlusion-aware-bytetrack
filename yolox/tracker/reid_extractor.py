#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
ReID 外观特征提取器。

当前保持在 Torchreid/OSNet 路线，但不再写死 `osnet_x0_25`。
也就是说，后续可以直接切换到：
- osnet_x0_5
- osnet_x0_75
- osnet_x1_0
- osnet_ain_x1_0
- osnet_ibn_x1_0

而不需要再改 tracker 主体逻辑。
"""

from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms


_REID_EXTRACTOR_CACHE = {}

_OFFICIAL_REID_CHECKPOINTS = {
    "osnet_x1_0_msmt17_combineall.pth": {
        "url": "https://drive.google.com/uc?id=1IosIFlLiulGIjwW3H8uMRmx3MzPwf86x",
        "description": "official torchreid osnet_x1_0 MSMT17(combineall) checkpoint",
    },
    "osnet_ain_x1_0_msmt17.pth": {
        "url": "https://drive.google.com/uc?id=1SigwBE6mPdqiJMqhuIY4aqC7--5CsMal",
        "description": "official torchreid osnet_ain_x1_0 MSMT17 checkpoint",
    },
    "osnet_x0_25_msmt17_combineall.pth": {
        "url": "https://drive.google.com/uc?id=1Kkx2zW89jq_NETu4u42CFZTMVD5Hwm6e",
        "description": "official torchreid osnet_x0_25 MSMT17(combineall) checkpoint",
    },
}


def _normalize_device(device):
    if isinstance(device, torch.device):
        return device
    if isinstance(device, str):
        if device == "cpu":
            return torch.device("cpu")
        if device.startswith("cuda"):
            return torch.device(device)
    if isinstance(device, int):
        return torch.device("cuda:{}".format(device))
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _unwrap_reid_state_dict(checkpoint, checkpoint_path):
    if isinstance(checkpoint, OrderedDict):
        state_dict = checkpoint
    elif isinstance(checkpoint, dict):
        state_dict = None
        for key in ("state_dict", "model_state_dict", "model", "net_dict"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, (dict, OrderedDict)):
                state_dict = candidate
                break
        if state_dict is None and checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
            state_dict = checkpoint
        if state_dict is None:
            raise ValueError("Unsupported ReID checkpoint format: {}".format(type(checkpoint)))
    else:
        raise ValueError("Unsupported ReID checkpoint format: {}".format(type(checkpoint)))

    normalized_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            continue
        normalized_key = key[7:] if key.startswith("module.") else key
        normalized_state_dict[normalized_key] = value

    if "classifier.weight" not in normalized_state_dict:
        raise KeyError("`classifier.weight` not found in ReID checkpoint: {}".format(checkpoint_path))
    return normalized_state_dict


def _ensure_reid_checkpoint(model_path):
    path = Path(model_path)
    if path.exists():
        return path

    meta = _OFFICIAL_REID_CHECKPOINTS.get(path.name)
    if meta is None:
        raise FileNotFoundError(
            "ReID checkpoint not found: {}. If you want automatic download, please use one of the known "
            "official filenames: {}.".format(
                model_path,
                ", ".join(sorted(_OFFICIAL_REID_CHECKPOINTS.keys())),
            )
        )

    try:
        import gdown
    except ImportError as exc:
        raise ImportError(
            "Missing `gdown`, cannot auto-download ReID checkpoint: {}.".format(model_path)
        ) from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    print("Auto-downloading {} -> {}".format(meta["description"], path))
    downloaded = gdown.download(meta["url"], str(path), quiet=False)
    if not downloaded or not path.exists():
        raise FileNotFoundError("Automatic download failed for ReID checkpoint: {}".format(model_path))
    return path


class BaseReIDExtractor:
    backend_name = "torchreid"

    def __init__(self, model_path, device=None, image_size=(256, 128), batch_size=32):
        self.model_path = model_path
        self.device = _normalize_device(device)
        self.image_size = tuple(image_size)
        self.batch_size = int(batch_size)
        self.model, self.feature_dim, self.model_info = self._build_model(model_path)
        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def _build_model(self, model_path):
        raise NotImplementedError

    def _clip_tlbr(self, tlbr, image_shape):
        height, width = image_shape[:2]
        x1, y1, x2, y2 = tlbr
        x1 = max(int(np.floor(x1)), 0)
        y1 = max(int(np.floor(y1)), 0)
        x2 = min(int(np.ceil(x2)), width - 1)
        y2 = min(int(np.ceil(y2)), height - 1)
        return x1, y1, x2, y2

    def _preprocess(self, image, tlbrs):
        crops = []
        valid = []
        input_h, input_w = self.image_size

        for tlbr in tlbrs:
            x1, y1, x2, y2 = self._clip_tlbr(tlbr, image.shape)
            if x2 <= x1 or y2 <= y1:
                valid.append(False)
                crops.append(None)
                continue

            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                valid.append(False)
                crops.append(None)
                continue

            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop = cv2.resize(crop.astype(np.float32) / 255.0, (input_w, input_h))
            crops.append(self.transform(crop))
            valid.append(True)

        return crops, valid

    def __call__(self, image, tlbrs):
        if image is None or len(tlbrs) == 0:
            return np.zeros((0, self.feature_dim), dtype=np.float32)

        crops, valid = self._preprocess(image, tlbrs)
        features = np.zeros((len(tlbrs), self.feature_dim), dtype=np.float32)

        valid_indices = [idx for idx, is_valid in enumerate(valid) if is_valid]
        if not valid_indices:
            return features

        with torch.no_grad():
            for start in range(0, len(valid_indices), self.batch_size):
                end = start + self.batch_size
                batch_indices = valid_indices[start:end]
                batch = torch.stack([crops[idx] for idx in batch_indices], dim=0).to(self.device)
                batch_features = self.model(batch)
                batch_features = torch.nn.functional.normalize(batch_features, dim=1)
                features[batch_indices] = batch_features.cpu().numpy().astype(np.float32)

        return features


class TorchreidReIDExtractor(BaseReIDExtractor):
    def __init__(self, model_path, backbone="osnet_x0_25", device=None, image_size=(256, 128), batch_size=32):
        self.backbone = backbone
        super().__init__(model_path=model_path, device=device, image_size=image_size, batch_size=batch_size)

    def _build_model(self, model_path):
        from torchreid.reid.models import build_model

        resolved_model_path = _ensure_reid_checkpoint(model_path)
        checkpoint = torch.load(resolved_model_path, map_location="cpu")
        state_dict = _unwrap_reid_state_dict(checkpoint, str(resolved_model_path))
        num_classes = int(state_dict["classifier.weight"].shape[0])
        feature_dim = int(state_dict["classifier.weight"].shape[1])
        model = build_model(
            name=self.backbone,
            num_classes=num_classes,
            loss="softmax",
            pretrained=False,
            use_gpu=self.device.type == "cuda",
        )
        model.load_state_dict(state_dict, strict=True)
        model.to(self.device)
        return model, feature_dim, {"backbone": self.backbone, "resolved_model_path": str(resolved_model_path)}


class OSNetReIDExtractor(TorchreidReIDExtractor):
    """
    兼容旧脚本的类名。
    现在本质上是“任意 Torchreid backbone 提取器”的 OSNet 默认别名。
    """

    def __init__(self, model_path, device=None, image_size=(256, 128), batch_size=32, backbone="osnet_x0_25"):
        super().__init__(
            model_path=model_path,
            backbone=backbone,
            device=device,
            image_size=image_size,
            batch_size=batch_size,
        )


def infer_reid_backend(args):
    explicit = str(getattr(args, "reid_backbone", "auto") or "auto").strip().lower()
    if explicit != "auto":
        if explicit.startswith("transreid"):
            raise ValueError(
                "TransReID support has been rolled back. Please choose an OSNet-style backbone such as "
                "`osnet_x1_0`, `osnet_ain_x1_0` or `osnet_ibn_x1_0`."
            )
        return explicit

    model_path = str(getattr(args, "reid_model", "") or "").lower()
    if "transreid" in model_path:
        raise ValueError(
            "Current project has rolled back to the OSNet route, but the configured `reid_model` still points "
            "to a TransReID checkpoint. Please switch it to an OSNet checkpoint."
        )
    return "osnet_x0_25"


def get_reid_input_size(args):
    if hasattr(args, "reid_input_height") and hasattr(args, "reid_input_width"):
        return (int(getattr(args, "reid_input_height")), int(getattr(args, "reid_input_width")))
    return tuple(getattr(args, "reid_input_size", (256, 128)))


def build_reid_extractor(args, device=None):
    return TorchreidReIDExtractor(
        model_path=getattr(args, "reid_model", None),
        backbone=infer_reid_backend(args),
        device=device if device is not None else getattr(args, "device", "cpu"),
        image_size=get_reid_input_size(args),
        batch_size=getattr(args, "reid_batch_size", 32),
    )


def get_reid_extractor(args, device=None):
    if not getattr(args, "with_reid", False):
        return None

    image_size = get_reid_input_size(args)
    key = (
        infer_reid_backend(args),
        getattr(args, "reid_model", None),
        str(_normalize_device(device if device is not None else getattr(args, "device", "cpu"))),
        tuple(image_size),
        getattr(args, "reid_batch_size", 32),
    )
    extractor = _REID_EXTRACTOR_CACHE.get(key)
    if extractor is None:
        extractor = build_reid_extractor(args, device=key[2])
        _REID_EXTRACTOR_CACHE[key] = extractor
    return extractor
