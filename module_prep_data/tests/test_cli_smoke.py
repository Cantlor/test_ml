from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_stage_scripts_have_working_help():
    root = Path(__file__).resolve().parents[1]
    scripts = [
        root / "scripts" / "01_check_inputs.py",
        root / "scripts" / "02_clip_to_aoi.py",
        root / "scripts" / "03_make_patches.py",
        root / "scripts" / "04_split_dataset.py",
    ]
    for script in scripts:
        cp = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=False,
        )
        assert cp.returncode == 0, f"{script} --help failed: {cp.stderr}"
        assert "usage:" in cp.stdout.lower()
