#!/usr/bin/env python3
"""Detect rosbag files that fall below a minimal size threshold."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List

MINIMAL_SIZE_MB = 40.0
DEFAULT_EXTENSIONS = (".mcap")


def iter_rosbag_files(paths: Iterable[Path], extensions: Iterable[str]) -> Iterable[Path]:
    for root in paths:
        if root.is_file():
            if root.suffix in extensions:
                yield root
            continue
        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                if Path(filename).suffix in extensions:
                    yield Path(dirpath, filename)


def find_small_bags(paths: Iterable[Path], min_size_mb: float, extensions: Iterable[str]) -> List[Path]:
    small_bags: List[Path] = []
    for bag in iter_rosbag_files(paths, extensions):
        size_mb = bag.stat().st_size / (1024 * 1024)
        if size_mb < min_size_mb:
            small_bags.append(bag)
            print(f"{size_mb:8.2f} MB  {bag}")
    return small_bags


def main() -> None:
    parser = argparse.ArgumentParser(description="List rosbag files smaller than the minimal size threshold.")
    parser.add_argument(
        "paths",
        nargs="*",
        default=[Path.cwd()],
        type=Path,
        help="Directory or file paths to inspect (defaults to current working directory).",
    )
    parser.add_argument(
        "--min-size-mb",
        type=float,
        default=MINIMAL_SIZE_MB,
        help=f"Minimal rosbag size in MB (default: {MINIMAL_SIZE_MB}).",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=list(DEFAULT_EXTENSIONS),
        help="File extensions treated as rosbag files.",
    )
    args = parser.parse_args()

    small_bags = find_small_bags(args.paths, args.min_size_mb, tuple(args.extensions))

    if not small_bags:
        print("No rosbag files under the minimal size were found.")


if __name__ == "__main__":
    main()

