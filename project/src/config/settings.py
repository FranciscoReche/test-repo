"""Configuration loading for reproducible research runs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Settings:
    """Thin wrapper around the YAML configuration dictionary."""

    raw: dict[str, Any]

    @property
    def paths(self) -> dict[str, str]:
        return self.raw.get("paths", {})

    @property
    def quality(self) -> dict[str, Any]:
        return self.raw.get("quality", {})

    @property
    def modeling(self) -> dict[str, Any]:
        return self.raw.get("modeling", {})

    @property
    def backtest(self) -> dict[str, Any]:
        return self.raw.get("backtest", {})

    @property
    def cutoffs(self) -> dict[str, Any]:
        return self.raw.get("cutoffs", {})

    def path(self, key: str) -> Path:
        if key not in self.paths:
            raise KeyError(f"Missing path setting: {key}")
        return Path(self.paths[key])


def load_settings(path: str | Path = "project/src/config/default.yaml") -> Settings:
    """Load YAML settings and create output directories."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    settings = Settings(data)
    for key in ("raw_dir", "processed_dir", "reports_dir", "artifacts_dir"):
        settings.path(key).mkdir(parents=True, exist_ok=True)
    return settings
