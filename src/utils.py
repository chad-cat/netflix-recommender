"""Shared utilities: config loading, RNG seeding, logging, IO helpers."""
from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def get_logger(name: str = "netflix_rec") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


LOG = get_logger()


def set_seed(seed: int = 42) -> None:
    """Seed every RNG we use so runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    """Load the YAML config. Falls back to a JSON read if PyYAML is absent."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    if yaml is not None:
        return yaml.safe_load(text)
    # Minimal fallback: only works if the file happens to be JSON.
    return json.loads(text)


def ensure_dirs(config: Dict[str, Any]) -> None:
    for key, path in config.get("paths", {}).items():
        os.makedirs(path, exist_ok=True)


def save_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)
    LOG.info("Saved JSON -> %s", path)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    return str(o)


@dataclass
class Timer:
    """Tiny context-manager timer for reporting training/eval cost."""
    label: str = "task"
    seconds: float = 0.0

    def __enter__(self):
        import time
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        import time
        self.seconds = time.perf_counter() - self._start
        LOG.info("%s finished in %.2fs", self.label, self.seconds)
