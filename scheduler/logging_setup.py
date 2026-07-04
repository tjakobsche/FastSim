# scheduler/logging_setup.py
from __future__ import annotations

import logging
from pathlib import Path
from dataclasses import dataclass

@dataclass(frozen=True)
class RunLogs:
    print_log: logging.Logger
    sreport_log: logging.Logger
    print_log_fp: Path
    power_log_fp: Path
    log_dir: Path

def _make_file_logger(name: str, fp: Path) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Prevent duplicate handlers if code is called multiple times in same process
    for h in list(logger.handlers):
        logger.removeHandler(h)

    handler = logging.FileHandler(fp)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    return logger

def setup_run_logs(results_filepath: str | Path, log_dir: str | Path | None = None) -> RunLogs:
    results_path = Path(results_filepath)
    base_name = results_path.stem or "run"

    if log_dir is None:
        if results_path.name:
            # Put logs next to output by default
            log_dir_path = results_path.resolve().parent / "logs"
        else:
            # No --output given; keep logs under the current directory
            log_dir_path = Path("logs").resolve()
    else:
        log_dir_path = Path(log_dir).expanduser().resolve()

    log_dir_path.mkdir(parents=True, exist_ok=True)

    print_fp = log_dir_path / f"logfile_{base_name}.log"
    sreport_fp = log_dir_path / f"sreport_{base_name}.log"
    power_fp = log_dir_path / f"{base_name}_power_log.csv"

    print_log = _make_file_logger(f"fastsim.print.{base_name}", print_fp)
    sreport_log = _make_file_logger(f"fastsim.sreport.{base_name}", sreport_fp)

    # Header line for sreport-style output
    sreport_log.info("time,running_nodes,down_nodes,planned_nodes,idle_nodes")

    return RunLogs(print_log=print_log, sreport_log=sreport_log, print_log_fp=print_fp, power_log_fp=power_fp, log_dir=log_dir_path)
