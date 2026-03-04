from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_csv_row(path: Path, row: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    write_header = not path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8", newline="") as f:
        if not rows:
            f.write("")
            return
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
