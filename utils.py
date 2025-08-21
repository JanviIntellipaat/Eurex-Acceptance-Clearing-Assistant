
from __future__ import annotations
from pathlib import Path

def ensure_dirs(*paths):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)

def readable_bytes(n: int) -> str:
    step = 1024.0
    units = ["B","KB","MB","GB","TB"]
    i = 0
    num = float(n)
    while num >= step and i < len(units)-1:
        num /= step; i += 1
    return f"{num:.1f} {units[i]}"
