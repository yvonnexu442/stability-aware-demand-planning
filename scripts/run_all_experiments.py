"""Convenience wrapper for running current experiment modules.

Dataset-specific scripts remain the canonical entry points:
``profile_dataco.py``, ``run_favorita_minimal_pipeline.py``,
``run_m5_robustness_pipeline.py``, ``run_walmart_robustness_pipeline.py``,
``run_switch_budget_sensitivity.py``, and ``run_thesis_quantification.py``.
This wrapper is maintained for reproducible batch execution.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence

from utils.logging_utils import setup_logger


DEFAULT_EXPERIMENTS = ["dataco", "favorita", "m5", "walmart"]
SCRIPT_BY_EXPERIMENT = {
    "dataco": "scripts/profile_dataco.py",
    "favorita": "scripts/run_favorita_minimal_pipeline.py",
    "m5": "scripts/run_m5_robustness_pipeline.py",
    "walmart": "scripts/run_walmart_robustness_pipeline.py",
    "switch_budget": "scripts/run_switch_budget_sensitivity.py",
    "thesis_quantification": "scripts/run_thesis_quantification.py",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run current experiment modules for the planning-stability paper.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--run-mode", default="quick")
    parser.add_argument("--experiments", nargs="+", choices=sorted(SCRIPT_BY_EXPERIMENT), default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--max-series", type=int, default=None)
    parser.add_argument("--include-switch-budget", action="store_true", help="Append Walmart switch-budget sensitivity to the run list.")
    parser.add_argument("--include-thesis-quantification", action="store_true", help="Append thesis-level summary table generation to the run list.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue running later modules after a failure.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    return parser.parse_args()


def main() -> None:
    """Run configured experiment modules in sequence."""
    args = parse_args()
    logger = setup_logger("run_all_experiments")
    experiments = list(args.experiments)
    if args.include_switch_budget and "switch_budget" not in experiments:
        experiments.append("switch_budget")
    if args.include_thesis_quantification and "thesis_quantification" not in experiments:
        experiments.append("thesis_quantification")

    commands = [build_command(experiment, args) for experiment in experiments]
    for command in commands:
        logger.info("Resolved command: %s", " ".join(command))
    if args.dry_run:
        return

    failures = []
    for experiment, command in zip(experiments, commands):
        try:
            logger.info("Running experiment module: %s", experiment)
            subprocess.run(command, cwd=Path.cwd(), env=pythonpath_env(), check=True)
        except subprocess.CalledProcessError as error:
            failures.append((experiment, error.returncode))
            logger.error("Experiment module %s failed with exit code %s.", experiment, error.returncode)
            if not args.continue_on_error:
                raise

    if failures:
        formatted = ", ".join("{}:{}".format(name, code) for name, code in failures)
        raise SystemExit("One or more experiment modules failed: {}".format(formatted))


def build_command(experiment: str, args: argparse.Namespace) -> List[str]:
    """Return the concrete command for one experiment module."""
    command = [sys.executable, SCRIPT_BY_EXPERIMENT[experiment]]
    if experiment not in {"dataco", "thesis_quantification"}:
        command.extend(["--config", args.config, "--run-mode", args.run_mode])
    if args.max_series is not None and experiment not in {"dataco", "thesis_quantification"}:
        command.extend(["--max-series", str(args.max_series)])
    if experiment == "switch_budget":
        command.extend(["--dataset", "walmart"])
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
