from __future__ import annotations

from pathlib import Path



def module_root() -> Path:
    return Path(__file__).resolve().parents[1]



def project_root() -> Path:
    return module_root().parents[0]



def default_runs_root() -> Path:
    return (project_root() / "output_data" / "module_net_train" / "runs").resolve()
