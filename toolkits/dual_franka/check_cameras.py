#!/usr/bin/env python3
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
"""Camera health check for dual-Franka eval/collect.

Verifies the three cameras expected by ``realworld_eval_dual_franka.yaml``
are (a) enumerated, (b) negotiated at USB 3.x, and (c) actually streaming
one frame. Also lists any processes holding /dev/video* — useful when
a previous Ray worker leaked a V4L2 handle and ``cap.read()`` times out.

After all three PASS, opens a 3-pane preview window streaming live frames
from all cameras for ``--stream-secs`` seconds (default 60). Press ``q``
to exit early. When ``DISPLAY`` is unset (typical SSH without -X), the
script auto-detects the local physical console session via ``who`` +
``XAUTHORITY`` probing so the window appears on the monitor wired to the
machine. Use ``--no-stream`` to skip the preview entirely.

Run:
    python toolkits/dual_franka/check_cameras.py
    python toolkits/dual_franka/check_cameras.py --stream-secs 30
    python toolkits/dual_franka/check_cameras.py --no-stream

Exit code 0 only if all three PASS; otherwise 1.
"""

from __future__ import annotations

import argparse
import getpass
import glob
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Expected hardware (matches the active eval yaml). Edit if the rig changes.
EXPECTED_REALSENSE_SERIAL = "941322072906"
EXPECTED_LUMOS = {
    "left  (1242)": "usb-XVisio_Technology_XVisio_vSLAM_250801DR48FB26001242-video-index0",
    "right (1173)": "usb-XVisio_Technology_XVisio_vSLAM_250801DR48FB26001173-video-index0",
}

XVISIO_VID = "040e"


def find_usb_speed(vid: str, serial_substr: str = "") -> str | None:
    for d in glob.glob("/sys/bus/usb/devices/*/idVendor"):
        try:
            if open(d).read().strip() != vid:
                continue
        except OSError:
            continue
        dev = Path(d).parent
        sn_file = dev / "serial"
        sn = sn_file.read_text().strip() if sn_file.exists() else ""
        if serial_substr and serial_substr not in sn:
            continue
        speed = dev / "speed"
        if speed.exists():
            return speed.read_text().strip() + "M"
    return None


def check_realsense() -> dict:
    r: dict = {"name": "RealSense", "status": "FAIL", "details": []}
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        r["details"].append(f"pyrealsense2 import failed: {exc}")
        return r

    devs = list(rs.context().query_devices())
    if not devs:
        r["details"].append("no RealSense device on the bus")
        return r

    matched = None
    for d in devs:
        sn = d.get_info(rs.camera_info.serial_number)
        r["details"].append(
            f"found serial={sn} "
            f"usb={d.get_info(rs.camera_info.usb_type_descriptor)} "
            f"fw={d.get_info(rs.camera_info.firmware_version)}"
        )
        if sn == EXPECTED_REALSENSE_SERIAL:
            matched = d
    if matched is None:
        r["details"].append(f"expected serial {EXPECTED_REALSENSE_SERIAL} NOT present")
        return r

    usb_descriptor = matched.get_info(rs.camera_info.usb_type_descriptor)

    try:
        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(EXPECTED_REALSENSE_SERIAL)
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
        pipe.start(cfg)
        frames = pipe.wait_for_frames(timeout_ms=3000)
        color = frames.get_color_frame()
        pipe.stop()
        if not color:
            r["details"].append("pipeline started but no color frame in 3s")
            return r
        r["details"].append("color stream: 1 frame OK")
    except Exception as exc:
        r["details"].append(f"pipeline error: {exc}")
        return r

    if usb_descriptor.startswith("3"):
        r["status"] = "PASS"
    else:
        r["status"] = "WARN"
        r["details"].append(
            f"USB descriptor {usb_descriptor} — fell back from 3.x; "
            "swap cable or move to a root USB-3 port"
        )
    return r


def check_lumos(label: str, by_id: str) -> dict:
    r: dict = {"name": f"Lumos {label}", "status": "FAIL", "details": []}
    by_path = Path("/dev/v4l/by-id") / by_id
    if not by_path.exists():
        r["details"].append("device path missing — not plugged in or USB link dead")
        return r
    r["details"].append(f"{by_path.name} -> {os.path.realpath(by_path)}")

    # /sys USB speed (XVisio vendor 040e)
    serial_substr = by_id.split("vSLAM_")[1].split("-")[0] if "vSLAM_" in by_id else ""
    speed = find_usb_speed(XVISIO_VID, serial_substr)
    speed_warn = False
    if speed:
        r["details"].append(f"USB speed: {speed}")
        if not speed.startswith("5000"):
            speed_warn = True
            r["details"].append("⚠ link fell back below 5000M (USB 2.0)")

    # Functional read test — reuse the streaming opener so cold-start cases
    # (first invocation after boot, where the driver hasn't been configured
    # for YU12 1280x1280 yet) use the same double-open + warmup as streaming.
    # Without this we trip the V4L2 select() 10s timeout on the very first run.
    try:
        reader, closer = _open_lumos(by_id)
    except Exception as exc:
        r["details"].append(f"open/warmup failed: {exc}")
        return r
    try:
        frame = reader()
    finally:
        closer()
    if frame is None:
        r["details"].append(
            "opened + warmed up but reader returned None "
            "(driver in inconsistent state — try re-plugging or another USB port)"
        )
        return r

    r["details"].append(f"read frame OK shape={frame.shape}")
    r["status"] = "WARN" if speed_warn else "PASS"
    return r


def show_video_holders() -> None:
    print("\n=== /dev/video* holders (fuser) ===")
    nodes = sorted(glob.glob("/dev/video*"))
    if not nodes:
        print("  (no /dev/video* nodes)")
        return
    any_holder = False
    for n in nodes:
        try:
            res = subprocess.run(
                ["fuser", "-v", n],
                capture_output=True,
                text=True,
                timeout=2,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            print(f"  {n}: fuser error: {exc}")
            continue
        text = (res.stderr or "").strip()
        if text and "PID" in text:
            any_holder = True
            print(text)
    if not any_holder:
        print("  (no processes hold any /dev/video*)")


def _can_connect_display(disp: str, auth: str | None) -> bool:
    """Probe an X display via xdpyinfo. Returns True if connectable."""
    if not shutil.which("xdpyinfo"):
        return False
    env = os.environ.copy()
    env["DISPLAY"] = disp
    if auth:
        env["XAUTHORITY"] = auth
    else:
        env.pop("XAUTHORITY", None)
    try:
        r = subprocess.run(["xdpyinfo"], env=env, capture_output=True, timeout=2)
        return r.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def _resolve_display() -> tuple[str | None, str | None]:
    """Find DISPLAY+XAUTHORITY for the local physical console.

    Order:
      1. Existing DISPLAY (e.g. SSH ``-X``, or already on console).
      2. ``who`` parses for ``(:N)`` of currently-logged-in console sessions,
         tried with several common ``XAUTHORITY`` paths.
      3. Falls back to ``:0`` and ``:1`` with default auth.

    Returns ``(None, None)`` if no display can be opened.
    """
    cur_disp = os.environ.get("DISPLAY")
    cur_auth = os.environ.get("XAUTHORITY")
    if cur_disp and _can_connect_display(cur_disp, cur_auth):
        return cur_disp, cur_auth

    uid = os.getuid()
    user = getpass.getuser()
    home = os.path.expanduser("~")

    # Discover candidate :N from `who` (covers GDM/SDDM/lightdm console sessions)
    display_candidates: list[str] = []
    try:
        out = subprocess.check_output(["who"], text=True, timeout=2)
        for line in out.splitlines():
            m = re.search(r"\((:[0-9]+)\)", line)
            if m and m.group(1) not in display_candidates:
                display_candidates.append(m.group(1))
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    for fallback in (":0", ":1"):
        if fallback not in display_candidates:
            display_candidates.append(fallback)

    # Candidate XAUTHORITY files (most-specific first)
    auth_candidates: list[str | None] = [None]  # try unset first
    for path in [
        cur_auth,
        f"{home}/.Xauthority",
        f"/run/user/{uid}/gdm/Xauthority",
        f"/var/run/lightdm/{user}/xauthority",
    ]:
        if path and os.path.exists(path) and path not in auth_candidates:
            auth_candidates.append(path)
    # GNOME-on-Wayland XWayland cookies live as random suffixes
    for path in glob.glob(f"/run/user/{uid}/.mutter-Xwaylandauth.*"):
        if path not in auth_candidates:
            auth_candidates.append(path)
    for path in glob.glob(f"/run/user/{uid}/gdm/Xauthority"):
        if path not in auth_candidates:
            auth_candidates.append(path)

    for disp in display_candidates:
        for auth in auth_candidates:
            if _can_connect_display(disp, auth):
                return disp, auth
    return None, None


class _CameraThread:
    """Background pump: opens a camera, keeps `latest` updated, isolates errors.

    A bad camera never blocks main-thread display; it just shows a status pane.
    """

    # If the pump thread doesn't push a fresh frame within this many seconds,
    # `get()` treats `_latest` as stale and returns None so the display shows
    # NO SIGNAL — covers the case where V4L2 read() blocks in C land and the
    # pump itself is stuck (no exception, no None return, no status change).
    STALE_THRESHOLD_S = 2.0

    def __init__(self, label: str, opener):
        import threading

        self.label = label
        self._opener = opener
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._latest = None  # numpy frame or None
        self._frames = 0
        self._last_frame_time = 0.0  # monotonic seconds; 0 means "no frame yet"
        self.status = "opening…"
        self._thread = threading.Thread(
            target=self._run, name=f"cam:{label}", daemon=True
        )

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2.0)

    def get(self):
        with self._lock:
            latest, status, count = self._latest, self.status, self._frames
            last_t = self._last_frame_time
        if latest is not None and last_t > 0.0:
            age = time.monotonic() - last_t
            if age > self.STALE_THRESHOLD_S:
                return None, f"STALE {age:.1f}s (pump hung)", count
        return latest, status, count

    def _run(self):
        try:
            reader, closer = self._opener()
        except Exception as exc:
            self.status = f"open failed: {exc}"
            return
        try:
            self.status = "live"
            consecutive_fail = 0
            while not self._stop.is_set():
                try:
                    frame = reader()
                except Exception as exc:
                    self.status = f"read err: {exc}"
                    time.sleep(0.2)
                    continue
                if frame is None:
                    consecutive_fail += 1
                    if consecutive_fail > 30:
                        self.status = "no frames"
                    time.sleep(0.01)
                    continue
                consecutive_fail = 0
                with self._lock:
                    self._latest = frame
                    self._frames += 1
                    self._last_frame_time = time.monotonic()
                    self.status = "live"
        finally:
            try:
                closer()
            except Exception:
                pass


def _open_realsense():
    import numpy as np
    import pyrealsense2 as rs

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(EXPECTED_REALSENSE_SERIAL)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
    pipe.start(cfg)

    def reader():
        # poll = non-blocking; returns empty frameset until a frame arrives
        ok, fs = pipe.try_wait_for_frames(timeout_ms=100)
        if not ok:
            return None
        color = fs.get_color_frame()
        if not color:
            return None
        return np.asanyarray(color.get_data()).copy()

    def closer():
        pipe.stop()

    return reader, closer


def _open_lumos(by_id: str):
    """Open an XVisio vSLAM lumos camera at its only reliable mode (1280x1280
    YU12), and return a reader that yields BGR frames.

    Mirrors ``rlinf.envs.realworld.common.camera.lumos_camera.LumosCamera``,
    plus a built-in "double-open" + warmup. Cold-start STREAMON can lose the
    USB isoch bandwidth race; the driver retains the format on close, so a
    second open after a failed warmup almost always succeeds.
    """
    import cv2
    import numpy as np

    path = f"/dev/v4l/by-id/{by_id}"
    native_w, native_h = 1280, 1280
    buf_h = native_h * 3 // 2  # I420 packed plane height

    def _configure(c):
        c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YU12"))
        c.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        c.set(cv2.CAP_PROP_FRAME_WIDTH, native_w)
        c.set(cv2.CAP_PROP_FRAME_HEIGHT, native_h)
        c.set(cv2.CAP_PROP_FPS, 15)
        try:
            c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

    def _attempt():
        c = cv2.VideoCapture(path, cv2.CAP_V4L2)
        if not c.isOpened():
            c.release()
            return None
        _configure(c)
        # Warmup: discard up to 8 frames waiting for one whose buffer size
        # matches the I420 plane shape. A bad STREAMON returns short or junk
        # buffers; reshape failure here is the most reliable detector.
        for _ in range(8):
            ok, raw = c.read()
            if ok and raw is not None:
                try:
                    np.ascontiguousarray(raw).reshape(buf_h, native_w)
                    return c
                except ValueError:
                    pass
            time.sleep(0.05)
        c.release()
        return None

    cap = _attempt()
    if cap is None:
        # Cold-start STREAMON often fails the first time; the driver retains
        # the format so the second attempt typically succeeds immediately.
        time.sleep(0.2)
        cap = _attempt()
    if cap is None:
        raise RuntimeError(f"lumos warmup failed after 2 attempts: {path}")

    def reader():
        ok, raw = cap.read()
        if not ok or raw is None:
            return None
        try:
            yuv = np.ascontiguousarray(raw).reshape(buf_h, native_w)
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
        except Exception:
            return None

    def closer():
        cap.release()

    return reader, closer


def stream_preview(stream_secs: float, panel_w: int = 480, panel_h: int = 360) -> None:
    """3-pane live preview. Per-camera threads keep one slow/dead camera from
    freezing the display; the main loop only composites latest snapshots."""
    disp, auth = _resolve_display()
    if not disp:
        print(
            "\n[stream] no usable X display found — install xdpyinfo "
            "(`apt install x11-utils`) or run with `ssh -X`. Skipping preview."
        )
        return
    os.environ["DISPLAY"] = disp
    if auth:
        os.environ["XAUTHORITY"] = auth
    else:
        os.environ.pop("XAUTHORITY", None)
    print(f"\n[stream] using DISPLAY={disp} XAUTHORITY={auth or '(unset)'}")

    import cv2
    import numpy as np

    # Build pumps. Order = display order: LEFT (1242), BASE (realsense), RIGHT (1173).
    pumps: list[tuple[str, _CameraThread]] = []
    pumps.append(
        (
            "LEFT lumos 1242",
            _CameraThread("left", lambda: _open_lumos(EXPECTED_LUMOS["left  (1242)"])),
        )
    )
    pumps.append(("BASE realsense", _CameraThread("base", _open_realsense)))
    pumps.append(
        (
            "RIGHT lumos 1173",
            _CameraThread("right", lambda: _open_lumos(EXPECTED_LUMOS["right (1173)"])),
        )
    )
    for _, t in pumps:
        t.start()

    window_name = "dual-franka cameras (q to quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, panel_w * 3, panel_h)

    blank = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    deadline = time.time() + stream_secs
    display_frames = 0
    print(f"\n[stream] streaming {stream_secs:.0f}s — q to exit early")

    try:
        while time.time() < deadline:
            panes = []
            for title, pump in pumps:
                frame, status, count = pump.get()
                if frame is None:
                    pane = blank.copy()
                    cv2.putText(
                        pane,
                        "NO SIGNAL",
                        (panel_w // 2 - 90, panel_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
                else:
                    pane = cv2.resize(frame, (panel_w, panel_h))
                cv2.putText(
                    pane,
                    title,
                    (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    pane,
                    f"{status} ({count})",
                    (8, panel_h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (200, 200, 200),
                    1,
                    cv2.LINE_AA,
                )
                panes.append(pane)

            grid = np.hstack(panes)
            remaining = deadline - time.time()
            cv2.putText(
                grid,
                f"{remaining:0.1f}s",
                (grid.shape[1] - 110, panel_h - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, grid)
            display_frames += 1

            # ~30 FPS display tick; pumps run independently in their threads.
            if cv2.waitKey(33) & 0xFF == ord("q"):
                print("[stream] user pressed q")
                break
    finally:
        for _, t in pumps:
            t.stop()
        cv2.destroyAllWindows()
        elapsed = stream_secs - (deadline - time.time())
        fps = display_frames / max(elapsed, 0.001)
        print(
            f"[stream] done — display {display_frames} frames in {elapsed:.1f}s "
            f"(~{fps:.1f} FPS)"
        )
        for title, pump in pumps:
            _, status, count = pump.get()
            print(f"  {title:<22s} {count} frames captured, status={status}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--stream-secs",
        type=float,
        default=60.0,
        help="Seconds to run the live 3-pane preview after all PASS (default 60).",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Skip the live preview window even if all cameras PASS.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Dual-Franka camera health check")
    print("=" * 60)

    results = [check_realsense()]
    for label, by_id in EXPECTED_LUMOS.items():
        results.append(check_lumos(label, by_id))

    print("\n=== per-camera details ===")
    emoji = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗"}
    for r in results:
        print(f"\n{emoji[r['status']]} {r['name']:<22s} {r['status']}")
        for d in r["details"]:
            print(f"    {d}")

    print("\n=== summary ===")
    for r in results:
        print(f"  {r['name']:<22s} {emoji[r['status']]} {r['status']}")

    show_video_holders()

    all_pass = all(r["status"] == "PASS" for r in results)
    if all_pass and not args.no_stream and args.stream_secs > 0:
        stream_preview(args.stream_secs)
    elif not all_pass:
        print("\n[stream] skipping preview — not all cameras passed")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
