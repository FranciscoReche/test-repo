#!/usr/bin/env python
"""Rebuild master dataset and rerun temporal validation after adding new events."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config.settings import load_settings
from src.pipeline import build_master_dataset, run_modeling

parser = argparse.ArgumentParser(description="Recalibrate models and reports with the current raw data.")
parser.add_argument("--config", default="project/src/config/default.yaml")
args = parser.parse_args()
settings = load_settings(args.config)
df = build_master_dataset(settings)
run_modeling(settings, df)
