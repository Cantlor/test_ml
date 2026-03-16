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
    ap.add_argument(
        "--dry-run-budget",
        action="store_true",
        help="preview patch budget per dataset without writing patch artifacts",
    )
    ap.add_argument(
        "--budget-report-out",
        default=None,
        help="optional path for dry-run budget JSON report",
    )
    args = ap.parse_args()
    from prep.stages.make_patches import run

    return run(
        config_path=args.config,
        n_override=args.n,
        seed_override=args.seed,
        dry_run_budget=bool(args.dry_run_budget),
        budget_report_out=args.budget_report_out,
    )


if __name__ == "__main__":
    raise SystemExit(main())
