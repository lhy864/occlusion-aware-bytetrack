#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Render a MOT17 sequence and a MOTChallenge-format txt file into an mp4 video.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
from loguru import logger

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yolox.utils.visualize import plot_tracking


def make_parser():
    parser = argparse.ArgumentParser("Render MOT17 Tracking Video")
    parser.add_argument("--experiment-name", default=None, type=str, help="experiment name under YOLOX_outputs/<exp>/track_results")
    parser.add_argument("--result-file", default=None, type=str, help="explicit MOTChallenge-format txt path")
    parser.add_argument("--sequence", required=True, type=str, help="sequence name, e.g. MOT17-02-FRCNN")
    parser.add_argument("--ann-json", default="datasets/mot/annotations/val_half.json", type=str, help="annotation json used to map frame ids to image files")
    parser.add_argument("--dataset-root", default="datasets/mot/train", type=str, help="MOT17 train root")
    parser.add_argument("--fps", default=0, type=float, help="video fps override, use seqinfo.ini when <= 0")
    parser.add_argument("--output", default=None, type=str, help="output mp4 path")
    parser.add_argument("--start-frame", default=1, type=int, help="first split-relative frame id to render")
    parser.add_argument("--end-frame", default=0, type=int, help="last split-relative frame id to render, 0 means until the end")
    return parser


def resolve_result_file(args) -> Path:
    if args.result_file:
        path = (ROOT / args.result_file).resolve() if not Path(args.result_file).is_absolute() else Path(args.result_file)
        if not path.is_file():
            raise FileNotFoundError(f"Result file not found: {path}")
        return path

    if not args.experiment_name:
        raise ValueError("Either --result-file or --experiment-name must be provided.")

    path = ROOT / "YOLOX_outputs" / args.experiment_name / "track_results" / f"{args.sequence}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Track result file not found: {path}")
    return path


def load_frame_map(ann_json: Path, dataset_root: Path, sequence: str) -> dict[int, Path]:
    if not ann_json.is_file():
        raise FileNotFoundError(f"Annotation json not found: {ann_json}")
    data = json.loads(ann_json.read_text(encoding="utf-8"))
    frame_map: dict[int, Path] = {}
    for image in data.get("images", []):
        file_name = str(image["file_name"])
        if file_name.split("/")[0] != sequence:
            continue
        frame_id = int(image["frame_id"])
        frame_map[frame_id] = dataset_root / Path(file_name)
    if not frame_map:
        raise ValueError(f"No frames for sequence `{sequence}` were found in {ann_json}")
    return dict(sorted(frame_map.items()))


def load_tracks(result_file: Path) -> dict[int, list[tuple[list[float], int]]]:
    tracks: dict[int, list[tuple[list[float], int]]] = defaultdict(list)
    with result_file.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            fields = line.split(",")
            if len(fields) < 6:
                continue
            frame_id = int(float(fields[0]))
            track_id = int(float(fields[1]))
            tlwh = [
                float(fields[2]),
                float(fields[3]),
                float(fields[4]),
                float(fields[5]),
            ]
            tracks[frame_id].append((tlwh, track_id))
    return tracks


def read_seqinfo(seq_dir: Path) -> tuple[int, int, float]:
    seqinfo = seq_dir / "seqinfo.ini"
    if not seqinfo.is_file():
        raise FileNotFoundError(f"seqinfo.ini not found: {seqinfo}")

    width = height = None
    fps = None
    with seqinfo.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith("imWidth="):
                width = int(line.split("=", 1)[1])
            elif line.startswith("imHeight="):
                height = int(line.split("=", 1)[1])
            elif line.startswith("frameRate="):
                fps = float(line.split("=", 1)[1])

    if width is None or height is None or fps is None:
        raise ValueError(f"Failed to parse image size or frameRate from {seqinfo}")
    return width, height, fps


def default_output_path(result_file: Path, experiment_name: str | None, sequence: str) -> Path:
    if experiment_name:
        out_dir = ROOT / "YOLOX_outputs" / experiment_name / "mot17_vis"
    else:
        out_dir = result_file.parent / "mot17_vis"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{sequence}.mp4"


def main():
    args = make_parser().parse_args()

    result_file = resolve_result_file(args)
    ann_json = (ROOT / args.ann_json).resolve() if not Path(args.ann_json).is_absolute() else Path(args.ann_json)
    dataset_root = (ROOT / args.dataset_root).resolve() if not Path(args.dataset_root).is_absolute() else Path(args.dataset_root)
    sequence_dir = dataset_root / args.sequence
    frame_map = load_frame_map(ann_json, dataset_root, args.sequence)
    tracks = load_tracks(result_file)

    width, height, default_fps = read_seqinfo(sequence_dir)
    fps = args.fps if args.fps and args.fps > 0 else default_fps
    output_path = (
        (ROOT / args.output).resolve() if args.output and not Path(args.output).is_absolute()
        else Path(args.output) if args.output
        else default_output_path(result_file, args.experiment_name, args.sequence)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    valid_frame_ids = [fid for fid in frame_map.keys() if fid >= args.start_frame and (args.end_frame <= 0 or fid <= args.end_frame)]
    if not valid_frame_ids:
        raise ValueError("No frames matched the requested frame range.")

    logger.info("Rendering sequence {} from {} to {}", args.sequence, valid_frame_ids[0], valid_frame_ids[-1])
    logger.info("Using result file: {}", result_file)
    logger.info("Output video: {}", output_path)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (int(width), int(height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output_path}")

    try:
        for idx, frame_id in enumerate(valid_frame_ids, 1):
            image_path = frame_map[frame_id]
            image = cv2.imread(str(image_path))
            if image is None:
                raise FileNotFoundError(f"Failed to load frame image: {image_path}")

            frame_tracks = tracks.get(frame_id, [])
            tlwhs = [item[0] for item in frame_tracks]
            track_ids = [item[1] for item in frame_tracks]
            rendered = plot_tracking(image, tlwhs, track_ids, frame_id=frame_id, fps=float(fps))
            writer.write(rendered)

            if idx % 100 == 0 or idx == len(valid_frame_ids):
                logger.info("Rendered {}/{} frames", idx, len(valid_frame_ids))
    finally:
        writer.release()

    logger.info("Done: {}", output_path)


if __name__ == "__main__":
    main()
