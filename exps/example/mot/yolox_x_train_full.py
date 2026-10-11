# encoding: utf-8

from exps.example.mot.yolox_x_ablation import Exp as BaseExp


class Exp(BaseExp):
    def __init__(self):
        super().__init__()
        self.exp_name = "yolox_x_train_full"
        self.val_ann = "train.json"
