from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "prep_config.yaml"))
    args = ap.parse_args()
    from prep.stages.check_inputs import run

    return run(config_path=args.config)


if __name__ == "__main__":
    raise SystemExit(main())
