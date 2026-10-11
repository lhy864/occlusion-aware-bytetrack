"""Check a source release, optionally running a short end-to-end inference."""

import argparse
import ast
import json
import os
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IGNORED_DIRS = {"__pycache__", ".git", ".venv", "venv", "build", "dist", "outputs", "YOLOX_outputs"}
FORBIDDEN_SUFFIXES = {
    ".pth", ".pt", ".tar", ".onnx", ".engine", ".pyd", ".dll", ".so",
    ".zip", ".mp4", ".avi", ".mov", ".mkv", ".pdf", ".docx", ".pyc",
}
ENTRYPOINTS = {
    "demo": "tools/demo_track.py",
    "eval": "tools/track.py",
    "build_mot17_reid_dataset": "tools/build_mot17_reid_dataset.py",
    "train_osnet": "tools/train_mot17_reid.py",
}


def source_files(prepared_runtime=False):
    return sorted(
        path for path in ROOT.rglob("*")
        if path.is_file() and not IGNORED_DIRS.intersection(path.relative_to(ROOT).parts)
        and not any(part.endswith(".egg-info") for part in path.relative_to(ROOT).parts)
        and not (prepared_runtime and path.relative_to(ROOT).parts[0] in {"datasets", "pretrained", "videos"}
                 and path.name != "README.md")
    )


def parser_destinations(path):
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    destinations = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument":
            continue
        explicit = next((item.value for item in node.keywords if item.arg == "dest"), None)
        if isinstance(explicit, ast.Constant):
            destinations.add(explicit.value)
            continue
        flags = [item.value for item in node.args if isinstance(item, ast.Constant)
                 and isinstance(item.value, str)]
        if flags:
            flag = next((value for value in flags if value.startswith("--")), flags[0])
            destinations.add(flag.lstrip("-").replace("-", "_"))
    return destinations


def static_checks(prepared_runtime=False):
    failures = []
    files = source_files(prepared_runtime)
    python_files = [path for path in files if path.suffix == ".py"]
    for path in python_files:
        try:
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        except (SyntaxError, UnicodeError) as error:
            failures.append("Invalid Python: {} ({})".format(path.relative_to(ROOT), error))

    configs = sorted((ROOT / "configs").rglob("*.json"))
    for path in configs:
        try:
            config = json.loads(path.read_text(encoding="utf-8-sig"))
            category = path.parent.name
            if category in {"demo", "eval"}:
                entry = ENTRYPOINTS[category]
            elif path.stem == "build_mot17_reid_dataset":
                entry = ENTRYPOINTS["build_mot17_reid_dataset"]
            else:
                entry = ENTRYPOINTS["train_osnet"]
            unknown = set(config) - parser_destinations(ROOT / entry)
            if unknown:
                failures.append("Unknown config keys: {} {}".format(path.relative_to(ROOT), sorted(unknown)))
            if "exp_file" in config and not (ROOT / config["exp_file"]).is_file():
                failures.append("Missing experiment: {}".format(config["exp_file"]))
            if config.get("with_long_recovery") and not config.get("with_reid"):
                failures.append("Long recovery requires ReID: {}".format(path.relative_to(ROOT)))
        except (ValueError, TypeError, OSError) as error:
            failures.append("Invalid config: {} ({})".format(path.relative_to(ROOT), error))

    secret_patterns = [
        r"(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}",
        r"AKIA[0-9A-Z]{16}",
        r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    ]
    local_path_pattern = r"(?<![A-Za-z])[A-Za-z]:[\\/]"
    for path in files:
        relative = path.relative_to(ROOT)
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            failures.append("Unexpected release asset: {}".format(relative))
        if path.stat().st_size > 50 * 1024 * 1024:
            failures.append("Oversized source file: {}".format(relative))
        if path.suffix in {".py", ".json", ".md", ".toml", ".txt"}:
            content = path.read_text(encoding="utf-8-sig")
            if re.search(local_path_pattern, content):
                failures.append("Machine-specific drive path: {}".format(relative))
            if any(re.search(pattern, content) for pattern in secret_patterns):
                failures.append("Possible credential: {}".format(relative))
            if path.suffix == ".md":
                for target in re.findall(r"\]\(([^)]+)\)", content):
                    target = target.split("#", 1)[0]
                    if not target or "://" in target or target.startswith("mailto:"):
                        continue
                    if not (path.parent / target).exists():
                        failures.append("Broken document link: {} -> {}".format(relative, target))

    if not (ROOT / "LICENSE").is_file() or not (ROOT / "THIRD_PARTY_NOTICES.md").is_file():
        failures.append("Missing license or attribution")
    if failures:
        raise RuntimeError("\n".join(failures))
    return {"files": len(files), "python_files": len(python_files), "configs": len(configs)}


def runtime_check(options):
    sys.path.insert(0, str(ROOT))
    import cv2
    import torch
    from tools import demo_track
    from yolox.exp import get_exp
    from yolox.tracker.byte_tracker import BYTETracker
    from yolox.tracking_utils.timer import Timer
    from yolox.utils.detector import build_detector, load_yolox_detector_weights
    from yolox.utils.visualize import plot_tracking

    weights_root = options.weights_root.resolve() if options.weights_root else ROOT / "pretrained"
    video = options.video.resolve() if options.video else ROOT / "videos/input.mp4"
    detector_weights = weights_root / "bytetrack_x_mot17.pth.tar"
    reid_weights = weights_root / "osnet_x1_0_mot17.pth"
    for required in (video, detector_weights, reid_weights):
        if not required.is_file():
            raise FileNotFoundError("Runtime input missing: {}".format(required))

    argv = ["--config", "configs/demo/mytest_yolox_x_occlusion_best.json",
            "--ckpt", str(detector_weights), "--reid-model", str(reid_weights), "--path", str(video)]
    previous_argv = sys.argv
    try:
        sys.argv = ["demo_track.py"] + argv
        args = demo_track.make_parser().parse_args(argv)
    finally:
        sys.argv = previous_argv

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.fp16 = args.fp16 and args.device.type == "cuda"
    exp = get_exp(args.exp_file, args.name)
    exp.test_size = (640, 640)
    exp.test_conf = args.conf
    exp.nmsthre = args.nms
    model = exp.get_model().to(args.device).eval()
    load_yolox_detector_weights(model, args.ckpt, map_location="cpu")
    if args.fp16:
        model = model.half()
    detector = build_detector(args, exp, device=args.device, model=model)
    tracker = BYTETracker(args, frame_rate=args.fps)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError("Could not open runtime video")
    frames = []
    last_image = None
    try:
        with torch.no_grad():
            for frame_id in range(1, 4):
                success, frame = cap.read()
                if not success:
                    raise RuntimeError("Runtime video must contain at least three frames")
                outputs, info = detector.inference(frame, Timer())
                detections = outputs[0]
                tracks = tracker.update(detections, [info["height"], info["width"]], detector.test_size,
                                        img=info["raw_img"])
                frames.append({"frame": frame_id, "detections": 0 if detections is None else len(detections),
                               "tracks": len(tracks)})
                last_image = plot_tracking(info["raw_img"], [track.tlwh for track in tracks],
                                           [track.track_id for track in tracks], frame_id=frame_id)
    finally:
        cap.release()

    if not any(item["detections"] for item in frames) or not any(item["tracks"] for item in frames):
        raise RuntimeError("Runtime produced no detections or tracks")
    if options.preview:
        options.preview.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(options.preview), last_image):
            raise RuntimeError("Could not write runtime preview")
    return {"device": str(args.device), "resolution": [640, 640], "frames": frames,
            "reid": args.with_reid, "long_recovery": args.with_long_recovery, "cmc": args.with_cmc,
            "inference_module": str(Path(demo_track.__file__).relative_to(ROOT))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", action="store_true", help="also infer the first three frames")
    parser.add_argument("--video", type=Path, help="video for optional runtime verification")
    parser.add_argument("--weights-root", type=Path, help="directory containing detector and ReID weights")
    parser.add_argument("--preview", type=Path, help="optional last-frame tracking image")
    options = parser.parse_args()
    os.chdir(ROOT)
    result = {"static": static_checks(prepared_runtime=options.runtime)}
    if options.runtime:
        result["runtime"] = runtime_check(options)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
