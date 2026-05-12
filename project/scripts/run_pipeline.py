#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import run_pipeline

parser = argparse.ArgumentParser(description="Run the Polymarket earnings research pipeline.")
parser.add_argument("--config", default="project/src/config/default.yaml", help="Path to YAML config.")
args = parser.parse_args()
run_pipeline(args.config)
