# encoding: utf-8

import os

import torch

from exps.example.mot.yolox_x_mix_mot20_ch import Exp as BaseExp
from yolox.data import get_yolox_datadir


class Exp(BaseExp):
    def __init__(self):
        super().__init__()
        self.exp_name = "yolox_x_mot20_train_full"
        self.val_ann = "train.json"

    def get_eval_loader(self, batch_size, is_distributed, testdev=False):
        from yolox.data import MOTDataset, ValTransform

        valdataset = MOTDataset(
            data_dir=os.path.join(get_yolox_datadir(), "MOT20"),
            json_file=self.val_ann,
            img_size=self.test_size,
            name="train",
            preproc=ValTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        )

        if is_distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(valdataset, shuffle=False)
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
            "batch_size": batch_size,
        }
        return torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)
