#!/usr/bin/env python3
"""Summarize a Cosmic Ray session and compute the mutation score.

Cosmic Ray's ``dump`` command emits one JSON array per line of the form
``[mutation_spec, work_result]``.  This script tolerates both that streaming
format and the raw SQLite ``work_results`` table so it keeps working if the
CLI layout changes.

Examples:
    cosmic-ray exec cosmic-ray.toml mutation-session.sqlite
    python scripts/mutation_report.py mutation-session.sqlite
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
from collections import Counter
from pathlib import Path

COUNTED_OUTCOMES = ("killed", "survived", "timeout", "incompetent", "error")


def read_dump(session: Path) -> list[dict[str, object]]:
    """Return work-result dictionaries for a session."""
    proc = subprocess.run(
        [sys.executable, "-m", "cosmic_ray.cli", "dump", str(session)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 0 and proc.stdout.strip():
        results: list[dict[str, object]] = []
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, list) and len(payload) >= 2 and isinstance(payload[1], dict):
                results.append(payload[1])
            elif isinstance(payload, dict):
                results.append(payload)
        if results:
            return results

    conn = sqlite3.connect(str(session))
    conn.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in conn.execute("SELECT * FROM work_results")]
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path, help="Cosmic Ray session SQLite file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("reports/mutation.json"),
        help="Where to write the JSON report",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Exit non-zero when the mutation score is below this percentage",
    )
    args = parser.parse_args()

    if not args.session.exists():
        parser.error(f"session does not exist: {args.session}")

    results = read_dump(args.session)
    counts: Counter[str] = Counter()
    diff_by_outcome: dict[str, str] = {}
    for result in results:
        outcome = str(result.get("test_outcome", "unknown")).lower()
        if outcome not in COUNTED_OUTCOMES:
            outcome = "unknown"
        counts[outcome] += 1
        if outcome in {"survived", "timeout", "incompetent", "error"} and outcome not in diff_by_outcome:
            diff_by_outcome[outcome] = str(result.get("diff") or result.get("output") or "")[:2000]

    total = sum(counts.values())
    killed = counts["killed"]
    score = (killed / total * 100.0) if total else 0.0
    report = {
        "session": str(args.session),
        "total": total,
        "counts": dict(counts),
        "mutation_score_percent": round(score, 2),
        "sample_failures": diff_by_outcome,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"{args.output}: total={total} killed={killed} "
        f"survived={counts['survived']} score={score:.2f}%"
    )

    if args.min_score is not None and score < args.min_score:
        print(f"mutation score {score:.2f}% < required {args.min_score:.2f}%", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
