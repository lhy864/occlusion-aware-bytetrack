#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Ultralytics 检测器对应的 MOT 评估器。

原始 ByteTrack 的 evaluator 默认假设检测器是 YOLOX 模型，
这里单独派生一个子类，把“读取数据 -> 调 Ultralytics YOLO 推理 -> tracker.update -> 写结果”
这条链路接起来。
"""

import itertools
import os
import time
from collections import defaultdict

import torch
from tqdm import tqdm

from yolox.evaluators.mot_evaluator import MOTEvaluator, write_results
from yolox.tracker.byte_tracker import BYTETracker
from yolox.utils import gather, is_main_process, synchronize, time_synchronized


class UltralyticsMOTEvaluator(MOTEvaluator):
    """
    Ultralytics YOLO 版本的 MOT 评估器。

    原始 MOTEvaluator 假设输入是 YOLOX dataloader 预处理后的 tensor。
    但 UltralyticsPredictor 更适合直接接收图片路径或原始图像。
    所以这里重写 evaluate()，在 MOT dataloader 中取出图片路径后，
    交给 UltralyticsPredictor.inference() 完成检测。
    """

    def evaluate(self, detector, distributed=False, result_folder=None):
        """
        执行完整 MOT 评估。

        主要输出有两类：
        1. data_list：检测评估相关结果，沿用原 evaluator 的接口。
        2. result_folder/*.txt：MOTChallenge 格式跟踪结果，后续用于计算 MOTA/IDF1。
        """
        # 这里延续原始 MOTEvaluator 的整体流程，
        # 但 detector.inference() 改成了直接接收图像路径并调用 Ultralytics YOLO。
        data_list = []
        results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1
        on_cuda = torch.cuda.is_available()

        tracker = BYTETracker(self.args)
        ori_thresh = self.args.track_thresh
        ori_buffer = self.args.track_buffer

        for cur_iter, (_, _, info_imgs, ids) in enumerate(progress_bar(self.dataloader)):
            with torch.no_grad():
                frame_id = info_imgs[2].item()
                video_id = info_imgs[3].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split("/")[0]

                self._apply_sequence_tuning(video_name, ori_thresh, ori_buffer)

                if video_name not in video_names:
                    video_names[video_id] = video_name

                # 每个视频从第一帧开始时都要重置一次 tracker，避免不同视频状态串掉。
                if frame_id == 1:
                    tracker = BYTETracker(self.args)
                    if len(results) != 0:
                        result_filename = os.path.join(
                            result_folder, "{}.txt".format(video_names[video_id - 1])
                        )
                        write_results(result_filename, results)
                        results = []

                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                dataset = self.dataloader.dataset
                image_path = os.path.join(dataset.data_dir, dataset.name, img_file_name[0])
                outputs, img_info = detector.inference(image_path)

                if is_time_record:
                    infer_end = time_synchronized() if on_cuda else time.time()
                    inference_time += infer_end - start

                output_results = self.convert_to_coco_format(outputs, info_imgs, ids)
                data_list.extend(output_results)

                # 只有 ReID / CMC 开启时才需要原图；否则没必要重复保留。
                raw_img = img_info["raw_img"] if (self.args.with_reid or self.args.with_cmc) else None
                online_targets = tracker.update(outputs[0], info_imgs, self.img_size, img=raw_img)

                online_tlwhs = []
                online_ids = []
                online_scores = []

                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > 1.6
                    if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)

                results.append((frame_id, online_tlwhs, online_ids, online_scores))

                if is_time_record:
                    track_end = time_synchronized() if on_cuda else time.time()
                    track_time += track_end - infer_end

                if cur_iter == len(self.dataloader) - 1:
                    result_filename = os.path.join(
                        result_folder, "{}.txt".format(video_names[video_id])
                    )
                    write_results(result_filename, results)

        statistics = torch.tensor(
            [inference_time, track_time, n_samples],
            device="cuda" if on_cuda else "cpu",
            dtype=torch.float32,
        )
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results


# 兼容旧命名，避免其他文件仍按 YOLOv8MOTEvaluator 引用时失效。
YOLOv8MOTEvaluator = UltralyticsMOTEvaluator
