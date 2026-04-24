# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Lightweight JSONL diagnostic logger for realworld eval debugging.

Each process writes to ``<session_dir>/<name>.<pid>.jsonl`` so concurrent
Ray workers don't stomp each other. Session dir resolves from
``$RLINF_DIAG_DIR`` if set (wire it in the launch script so env-worker
and policy share one dir), else ``<repo>/logs/diag/<timestamp>/``.

Disabled by ``RLINF_DIAG_DISABLE=1``. Never raises — diagnostics must
not kill the eval loop.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_lock = threading.Lock()
_session_dir: Path | None = None


def _resolve_dir() -> Path:
    global _session_dir
    if _session_dir is not None:
        return _session_dir
    override = os.environ.get("RLINF_DIAG_DIR")
    if override:
        d = Path(override)
    else:
        # Ray workers don't inherit driver env vars; the launch script
        # also writes a ``logs/diag/current`` symlink so workers can find
        # the active session without any per-actor env plumbing.
        link = _REPO_ROOT / "logs" / "diag" / "current"
        if link.exists() or link.is_symlink():
            d = link.resolve()
        else:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            d = _REPO_ROOT / "logs" / "diag" / stamp
    d.mkdir(parents=True, exist_ok=True)
    _session_dir = d
    return d


def _json_default(o: Any) -> Any:
    import numpy as np

    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    try:
        import torch

        if torch.is_tensor(o):
            return o.detach().cpu().tolist()
    except ImportError:
        pass
    return str(o)


def log_jsonl(name: str, record: dict[str, Any]) -> None:
    """Append one JSON line to ``<session_dir>/<name>.<pid>.jsonl``."""
    if os.environ.get("RLINF_DIAG_DISABLE") == "1":
        return
    try:
        path = _resolve_dir() / f"{name}.{os.getpid()}.jsonl"
        line = json.dumps(record, default=_json_default)
        with _lock:
            with path.open("a") as f:
                f.write(line + "\n")
    except Exception:
        pass
