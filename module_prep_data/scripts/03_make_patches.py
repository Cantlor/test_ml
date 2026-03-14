from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "prep_config.yaml"))
    ap.add_argument("--n", type=int, default=None, help="target patches per dataset override")
    ap.add_argument("--seed", type=int, default=None, help="seed override")
    args = ap.parse_args()
    from prep.stages.make_patches import run

    return run(config_path=args.config, n_override=args.n, seed_override=args.seed)


if __name__ == "__main__":
    raise SystemExit(main())
