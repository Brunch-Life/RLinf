# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
"""Live progress monitor for ``examples/embodiment/collect_real_data.py``.

Runs in a **separate terminal**. Tails the collector's log file and surfaces
success count plus latest keyboard / success / discard events as a live tqdm
bar. Zero cooperation required from the collector — just parses log lines.

Why this exists instead of a bar inside the collector:

- The collector runs as a Ray worker, so its stdout is batched (~500 ms) by
  Ray's log monitor, which breaks tqdm's ``\\r`` refresh when the batched
  chunks arrive.
- The driver's stdout is typically piped through ``tee`` in the launch
  script, so printing from the driver also won't refresh in place.
- A monitor in its own TTY dodges both issues — tqdm writes to a real
  terminal that no one else is rewriting.

Typical two-terminal usage (on the same node that runs the collector):

    # terminal 1 — launch (stdout gets tee'd to a log file)
    bash collect_data.sh 2>&1 | tee run_embodiment.log

    # terminal 2 — live bar
    python toolkits/realworld_check/collect_monitor.py run_embodiment.log

The monitor waits for the log file to exist, so it can be started either
before or after the collector.
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Iterator, Optional

from tqdm import tqdm

# Matches both "Total: N/M" (success line) and "Total success: N/M" (discard line).
_TOTAL_RE = re.compile(r"Total(?:\s+success)?:\s*(\d+)\s*/\s*(\d+)")
_KB_RE = re.compile(r"\[keyboard\]\s+(\S+)")
_SUCCESS_RE = re.compile(r"Success\s*\(reward=([-\d.]+)")


def _follow(path: Path, poll_s: float) -> Iterator[str]:
    """Yield new lines appended to ``path``. Waits for the file to appear."""
    while not path.exists():
        time.sleep(poll_s)
    with path.open("r", errors="replace") as f:
        f.seek(0, 2)  # start at EOF so existing content is not re-played
        while True:
            line = f.readline()
            if not line:
                time.sleep(poll_s)
                continue
            yield line.rstrip("\n")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Live progress monitor for collect_real_data.py.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "log_path",
        type=Path,
        help="Path to the collector's tee'd log file (stdout of launch script).",
    )
    ap.add_argument(
        "--poll",
        type=float,
        default=0.5,
        help="Tail poll interval in seconds (default: 0.5).",
    )
    args = ap.parse_args()

    pbar: Optional[tqdm] = None
    last_saved = 0
    last_event = ""
    last_reward: Optional[str] = None

    try:
        for line in _follow(args.log_path, args.poll):
            m_tot = _TOTAL_RE.search(line)
            if m_tot:
                saved = int(m_tot.group(1))
                target = int(m_tot.group(2))
                if pbar is None:
                    pbar = tqdm(
                        total=target,
                        dynamic_ncols=True,
                        desc="collect",
                        unit="ep",
                        leave=True,
                    )
                if saved > last_saved:
                    pbar.update(saved - last_saved)
                    last_saved = saved
                if saved >= target:
                    pbar.refresh()
                    break

            m_kb = _KB_RE.search(line)
            if m_kb:
                last_event = m_kb.group(1)

            m_succ = _SUCCESS_RE.search(line)
            if m_succ:
                last_reward = m_succ.group(1)

            if pbar is not None:
                post: dict[str, str] = {}
                if last_event:
                    post["last_event"] = last_event
                if last_reward is not None:
                    post["last_reward"] = last_reward
                if post:
                    pbar.set_postfix(post, refresh=False)
                    pbar.refresh()
    except KeyboardInterrupt:
        pass
    finally:
        if pbar is not None:
            pbar.close()


if __name__ == "__main__":
    main()
