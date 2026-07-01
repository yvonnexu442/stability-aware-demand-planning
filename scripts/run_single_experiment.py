"""Run one current experiment module through a stable command-line entry point."""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from utils.logging_utils import setup_logger


SCRIPT_BY_EXPERIMENT: Dict[str, str] = {
    "dataco": "scripts/profile_dataco.py",
    "favorita": "scripts/run_favorita_minimal_pipeline.py",
    "m5": "scripts/run_m5_robustness_pipeline.py",
    "walmart": "scripts/run_walmart_robustness_pipeline.py",
    "switch_budget": "scripts/run_switch_budget_sensitivity.py",
    "thesis_quantification": "scripts/run_thesis_quantification.py",
}


def parse_args() -> Tuple[argparse.Namespace, List[str]]:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run one operational-planning experiment module.")
    parser.add_argument("experiment", choices=sorted(SCRIPT_BY_EXPERIMENT))
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--run-mode", default="quick")
    parser.add_argument("--dataset", choices=["m5", "walmart"], default=None, help="Dataset for switch_budget sensitivity.")
    parser.add_argument("--max-series", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved command without running it.")
    return parser.parse_known_args()


def main() -> None:
    """Resolve and run the requested experiment script."""
    args, extra_args = parse_args()
    logger = setup_logger("run_single_experiment")
    command = build_command(args, extra_args)
    logger.info("Resolved command: %s", " ".join(command))
    if args.dry_run:
        return
    subprocess.run(command, cwd=Path.cwd(), env=pythonpath_env(), check=True)


def build_command(args: argparse.Namespace, extra_args: List[str]) -> List[str]:
    """Return the concrete Python command for one experiment."""
    script = SCRIPT_BY_EXPERIMENT[args.experiment]
    command = [sys.executable, script]
    if args.experiment not in {"dataco", "thesis_quantification"}:
        command.extend(["--config", args.config, "--run-mode", args.run_mode])
    if args.experiment == "switch_budget" and args.dataset is not None:
        command.extend(["--dataset", args.dataset])
    if args.max_series is not None and args.experiment not in {"dataco", "thesis_quantification"}:
        command.extend(["--max-series", str(args.max_series)])
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]
    command.extend(extra_args)
    return command


def pythonpath_env() -> Dict[str, str]:
    """Return an environment with src and scripts import paths available."""
    env = dict(os.environ)
    paths = ["src", "scripts"]
    existing = env.get("PYTHONPATH", "")
    if existing:
        paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


if __name__ == "__main__":
    main()
