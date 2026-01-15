#!/usr/bin/env python3
"""Detect rosbag files that fall below a minimal size threshold."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List

MINIMAL_SIZE_MB = 60
DEFAULT_EXTENSIONS = ".mcap"


def iter_rosbag_files(paths: Iterable[Path], extensions: Iterable[str]) -> Iterable[Path]:
    for root in paths:
        if root.is_file():
            if root.suffix == extensions:
                yield root
            continue
        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                if Path(filename).suffix == extensions:
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

    small_bags = find_small_bags([Path('/mnt/dataset/storage/')],
                                 MINIMAL_SIZE_MB,
                                 DEFAULT_EXTENSIONS)

    if not small_bags:
        print("No rosbag files under the minimal size were found.")


if __name__ == "__main__":
    main()

