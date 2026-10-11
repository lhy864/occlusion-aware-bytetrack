#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import argparse
import json
from pathlib import Path


def _load_json_config(config_path):
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError("Config file not found: {}".format(path))

    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    if not isinstance(config, dict):
        raise ValueError("Config file must contain a JSON object: {}".format(path))
    return config, path


def apply_config_defaults(parser, argv=None):
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", default=None, type=str)
    bootstrap_args, _ = bootstrap.parse_known_args(argv)

    parser.add_argument(
        "--config",
        default=bootstrap_args.config,
        type=str,
        help="path to a JSON config file with CLI defaults",
    )
    if not bootstrap_args.config:
        return parser

    config, path = _load_json_config(bootstrap_args.config)
    config.pop("config", None)
    known_dests = {action.dest for action in parser._actions if action.dest != argparse.SUPPRESS}
    unknown_keys = sorted(key for key in config.keys() if key not in known_dests)
    if unknown_keys:
        raise KeyError("Unknown config keys in {}: {}".format(path, ", ".join(unknown_keys)))

    parser.set_defaults(**config)
    return parser

