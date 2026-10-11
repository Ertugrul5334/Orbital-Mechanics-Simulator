"""Shared setup for the example scripts (lets them run without installing the package)."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib

if not os.environ.get("SHOW_PLOTS"):
    matplotlib.use("Agg")   # save figures to files; set SHOW_PLOTS=1 for windows

IMAGES = os.path.join(ROOT, "docs", "images")
os.makedirs(IMAGES, exist_ok=True)


def img(name: str) -> str:
    return os.path.join(IMAGES, name)
