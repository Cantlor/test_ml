from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "prep_config.yaml"))
    ap.add_argument(
        "--patches_manifest",
        default=None,
        help="explicit path to patches_manifest.json (strict source of truth)",
    )
    # Backward-compatible aliases:
    ap.add_argument("--patches_all", default=None, help="deprecated alias; resolved as <dir>/../patches_manifest.json")
    ap.add_argument("--out_prep_data", default=None, help="override export root (legacy mode)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--train", type=float, default=None)
    ap.add_argument("--val", type=float, default=None)
    ap.add_argument("--test", type=float, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    from prep.stages.split_dataset import run

    return run(
        config_path=args.config,
        patches_manifest_override=args.patches_manifest,
        patches_all_override=args.patches_all,
        out_prep_data_override=args.out_prep_data,
        seed_override=args.seed,
        train_override=args.train,
        val_override=args.val,
        test_override=args.test,
        overwrite=bool(args.overwrite),
    )


if __name__ == "__main__":
    raise SystemExit(main())
