#!/usr/bin/env python3
"""Run unattended real-data validation jobs for Pitcher Twin.

This runner creates no data. It fails if the requested real Statcast cache is
missing, then runs the current random and temporal split smoke tests and writes
an overnight status file.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def write_blocker(output_dir: Path, message: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "BLOCKED.md").write_text(f"# Blocked\n\n{message}\n")


def run_command(command: list[str], log_path: Path) -> dict:
    started_at = iso_now()
    with log_path.open("w") as log_file:
        proc = subprocess.run(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return {
        "command": command,
        "log": str(log_path),
        "started_at": started_at,
        "finished_at": iso_now(),
        "returncode": proc.returncode,
        "ok": proc.returncode == 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/overnight"))
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--samples", type=int, default=800)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    smoke_script = project_root / "scripts" / "real_data_smoke_test.py"
    summary_script = project_root / "scripts" / "summarize_overnight_results.py"
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    status = {
        "started_at": iso_now(),
        "finished_at": None,
        "ok": False,
        "data_path": str(args.data),
        "output_dir": str(output_dir),
        "commands": [],
        "artifacts": [],
    }

    if not args.data.exists():
        message = (
            f"Real Statcast data file is missing: `{args.data}`.\n\n"
            "Run a real data fetch or pass an existing real Statcast cache with `--data`."
        )
        write_blocker(output_dir, message)
        status["finished_at"] = iso_now()
        status["blocker"] = message
        write_json(output_dir / "run_status.json", status)
        return 2

    if not smoke_script.exists():
        message = f"Missing required smoke-test script: `{smoke_script}`."
        write_blocker(output_dir, message)
        status["finished_at"] = iso_now()
        status["blocker"] = message
        write_json(output_dir / "run_status.json", status)
        return 2

    for split in ("random", "temporal"):
        report_path = output_dir / f"{split}_split_report.json"
        log_path = output_dir / f"{split}_split.log"
        command = [
            sys.executable,
            str(smoke_script),
            "--data",
            str(args.data),
            "--top",
            str(args.top),
            "--repeats",
            str(args.repeats),
            "--samples",
            str(args.samples),
            "--split",
            split,
            "--json-out",
            str(report_path),
        ]
        result = run_command(command, log_path)
        status["commands"].append(result)
        status["artifacts"].extend([str(report_path), str(log_path)])
        if not result["ok"]:
            message = (
                f"The `{split}` split validation failed. See `{log_path}`.\n\n"
                f"Command: `{' '.join(command)}`"
            )
            write_blocker(output_dir, message)
            status["finished_at"] = iso_now()
            status["blocker"] = message
            write_json(output_dir / "run_status.json", status)
            return result["returncode"] or 1

    if summary_script.exists():
        command = [
            sys.executable,
            str(summary_script),
            "--input-dir",
            str(output_dir),
            "--output",
            str(output_dir / "morning_report.md"),
        ]
        result = run_command(command, output_dir / "summary.log")
        status["commands"].append(result)
        status["artifacts"].extend(
            [str(output_dir / "morning_report.md"), str(output_dir / "summary.log")]
        )
        if not result["ok"]:
            message = (
                f"Summary generation failed. See `{output_dir / 'summary.log'}`.\n\n"
                f"Command: `{' '.join(command)}`"
            )
            write_blocker(output_dir, message)
            status["finished_at"] = iso_now()
            status["blocker"] = message
            write_json(output_dir / "run_status.json", status)
            return result["returncode"] or 1

    status["ok"] = True
    status["finished_at"] = iso_now()
    write_json(output_dir / "run_status.json", status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

