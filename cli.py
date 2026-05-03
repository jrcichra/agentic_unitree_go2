"""
go2_cli.py — Natural language TUI for the Unitree Go2

A full-screen terminal UI where you type commands in plain English:
  - Top-right: live camera feed (sixel/kitty/unicode block art)
  - Main area: scrollable chat history
  - Bottom: input box (submit with Enter)

Install:
    uv sync

Run:
    uv run cli.py --ip 10.0.0.200
    uv run cli.py --ip 10.0.0.200 --model llava
    uv run cli.py --ip 10.0.0.200 --no-camera
"""

import asyncio
import argparse
import base64
import json
import sys
import time
import threading
import traceback
import warnings
import logging
import os
import io

warnings.filterwarnings("ignore")
os.environ["TERM_IMAGE_LOG_LEVEL"] = "error"

# POSIX-only: terminal probing for image protocol detection
try:
    import select as _select
    import termios as _termios
    import tty as _tty
    _HAS_TERMIOS = True
except ImportError:
    _HAS_TERMIOS = False


# ── Capture stderr in-memory so errors show inside the TUI, not on raw terminal ─
class _ErrorCapture(io.StringIO):
    """Intercepts stderr writes and routes them to the TUI error panel."""

    MAX = 20  # keep last N error lines

    def __init__(self):
        super().__init__()
        self._lines: list[str] = []
        self._app = None  # set after TUI starts

    def write(self, s: str) -> int:
        if s.strip():
            for line in s.rstrip().splitlines():
                self._lines.append(line)
            self._lines = self._lines[-self.MAX :]
            if self._app is not None:
                try:
                    self._app.call_from_thread(self._app.push_error, s.rstrip())
                except Exception:
                    pass
        return len(s)

    def flush(self):
        pass


_err_capture = _ErrorCapture()
sys.stderr = _err_capture

# Suppress noisy aiortc / aioice logging
logging.getLogger("aiortc").setLevel(logging.CRITICAL)
logging.getLogger("aiortc.codecs.h264").setLevel(logging.CRITICAL)
logging.getLogger("aioice").setLevel(logging.CRITICAL)
logging.getLogger("aioice.stun").setLevel(logging.CRITICAL)
logging.getLogger("asyncio").setLevel(logging.CRITICAL)
logging.basicConfig(
    stream=_err_capture,
    level=logging.WARNING,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

import requests
from PIL import Image

try:
    import cv2
    import numpy as np

    _CV2 = True
except ImportError:
    cv2 = None
    np = None
    _CV2 = False

# ── Optional: microphone capture for voice input ─────────────────────────────
try:
    import sounddevice as _sd
    _HAS_SD = True
except ImportError:
    _sd = None
    _HAS_SD = False

# ---------------------------------------------------------------------------
# TUI imports
# ---------------------------------------------------------------------------
try:
    from textual.app import App, ComposeResult
    from textual.widgets import RichLog, Input, Static, Label, Header, Footer
    from textual.containers import Horizontal, Vertical, ScrollableContainer
    from textual.reactive import reactive
    from textual import work
    from rich.text import Text
    from rich.panel import Panel
    from rich.console import Console
    from rich.markup import escape

    _TEXTUAL = True
except ImportError:
    _TEXTUAL = False
    print("ERROR: textual not installed. Run: pip install textual rich")
    sys.exit(1)

try:
    from unitree_webrtc_connect.webrtc_driver import (
        UnitreeWebRTCConnection,
        WebRTCConnectionMethod,
    )
    from unitree_webrtc_connect.constants import RTC_TOPIC, SPORT_CMD, MCF_CMD
except ImportError:
    print("ERROR: unitree_webrtc_connect not installed.")
    sys.exit(1)

# Monkey-patch library error handler
try:
    from unitree_webrtc_connect.msgs import error_handler as _eh

    def _safe(error):
        if not isinstance(error, (list, tuple)) or len(error) != 3:
            return
        try:
            ts, src, code = error
        except Exception:
            pass

    _eh.handle_error = _safe
except Exception:
    pass

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
MCF_TOPIC = "rt/api/sport/request"

_conn: "UnitreeWebRTCConnection | None" = None
_latest_frame_jpg: bytes | None = None
_latest_frame_ts: float = 0.0
_frame_lock = threading.Lock()
_sport_state: dict = {}
_low_state: dict = {}
_main_loop: "asyncio.AbstractEventLoop | None" = None

# ── Radio / audio playback ────────────────────────────────────────────────────
_radio_player = None   # aiortc MediaPlayer instance

# ── Voice / Whisper ───────────────────────────────────────────────────────────
_whisper_model = None
_whisper_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _code(resp: dict | None) -> int:
    if resp is None:
        return -1
    status = resp.get("data", {}).get("header", {}).get("status", {})
    if isinstance(status, dict):
        code = status.get("code", -1)
        if isinstance(code, int):
            return code
    return -1


async def _mcf(name: str, parameter: dict | None = None, timeout: float = 5.0) -> dict:
    payload: dict = {"api_id": MCF_CMD[name]}
    if parameter:
        payload["parameter"] = parameter
    if _conn is None or _conn.datachannel is None:
        return {"ok": False, "code": -1}
    try:
        resp = await asyncio.wait_for(
            _conn.datachannel.pub_sub.publish_request_new(MCF_TOPIC, payload),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        return {"ok": False, "code": -2}
    code = _code(resp)
    return {"ok": code == 0, "code": code}


async def _mcf_raw(
    api_id: int, parameter: dict | None = None, timeout: float = 5.0
) -> dict:
    payload: dict = {"api_id": api_id}
    if parameter:
        payload["parameter"] = parameter
    if _conn is None or _conn.datachannel is None:
        return {"ok": False, "code": -1}
    try:
        resp = await asyncio.wait_for(
            _conn.datachannel.pub_sub.publish_request_new(MCF_TOPIC, payload),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        return {"ok": False, "code": -2}
    code = _code(resp)
    return {"ok": code == 0, "code": code}


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------


async def _frame_loop(track) -> None:
    global _latest_frame_jpg, _latest_frame_ts
    consecutive_errors = 0
    while True:
        try:
            frame = await asyncio.wait_for(track.recv(), timeout=10.0)
            if frame is None:
                await asyncio.sleep(0.05)
                continue
            if not _CV2 or cv2 is None:
                continue
            img = frame.to_ndarray(format="bgr24")
            if img is None or img.size == 0:
                continue
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                jpg = buf.tobytes()  # copy before lock to minimise hold time
                with _frame_lock:
                    _latest_frame_jpg = jpg
                    _latest_frame_ts = time.time()
                consecutive_errors = 0
        except asyncio.TimeoutError:
            # track went silent — update timestamp so we don't report stale
            # but keep _latest_frame_jpg so we can still show the last frame
            consecutive_errors += 1
            if consecutive_errors > 5:
                # genuinely lost feed
                _latest_frame_ts = 0.0
            await asyncio.sleep(0.1)
        except Exception:
            consecutive_errors += 1
            await asyncio.sleep(0.1)


async def _start_camera() -> None:
    ch = getattr(_conn, "video", None)
    if ch is None:
        return

    async def on_track(track):
        asyncio.ensure_future(_frame_loop(track))

    ch.add_track_callback(on_track)
    await asyncio.sleep(0.5)
    ch.switchVideoChannel(True)


def _get_frame_b64(quality: int = 75) -> str | None:
    """Return base64 JPEG of latest frame, or None if unavailable/stale."""
    if not _CV2 or cv2 is None or np is None:
        return None
    with _frame_lock:
        if _latest_frame_jpg is None:
            return None
        # 5s stale threshold — if feed has truly stopped, _frame_loop zeroes the ts
        if _latest_frame_ts > 0 and (time.time() - _latest_frame_ts) > 5.0:
            return None
        frame_data = _latest_frame_jpg
    try:
        img = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return base64.b64encode(buf.tobytes()).decode("ascii") if ok else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Inline image rendering (iTerm2 / Kitty protocol)
# Alacritty supports the iTerm2 inline image protocol.
# We write the image escape directly to the real tty, bypassing Textual.
# ---------------------------------------------------------------------------


# Terminals that support the iTerm2 inline image protocol, identified by
# their XTVERSION response (sent in response to \x1b[>q).
_ITERM2_XTVERSION_NAMES = [
    b"alacritty", b"wezterm", b"iterm", b"windows terminal",
    b"vte", b"foot", b"mintty", b"rio",
]


def _detect_image_protocol_envvars() -> str:
    """
    Env-var fallback for image protocol detection when the terminal probe
    cannot run (non-TTY stdin, non-POSIX, etc.).
    """
    if os.environ.get("KITTY_WINDOW_ID"):
        return "kitty"
    term_prog = os.environ.get("TERM_PROGRAM", "")
    if "iTerm" in term_prog or "WezTerm" in term_prog:
        return "iterm2"
    if os.environ.get("WT_SESSION") or os.environ.get("WT_PROFILE_ID"):
        return "iterm2"
    if os.environ.get("COLORTERM") in ("truecolor", "24bit"):
        return "iterm2"
    return "halfblock"


def _probe_image_protocol() -> str:
    """
    Detect inline image protocol support by actually probing the terminal
    with escape sequences and reading back responses.  More reliable than
    env-var sniffing — works correctly in WSL2 where env vars lie.

    Probe order:
      1. Kitty graphics protocol query  (responds with APC \x1b_G…)
      2. XTVERSION query                (responds with terminal name string)
      3. iTerm2 ReportCellSize          (iTerm2 / some compatible respond)
      4. env-var fallback

    Must be called BEFORE Textual takes over the terminal.
    """
    if not _HAS_TERMIOS:
        return _detect_image_protocol_envvars()
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return _detect_image_protocol_envvars()

    try:
        fd = sys.stdin.fileno()
        old_attrs = _termios.tcgetattr(fd)
    except Exception:
        return _detect_image_protocol_envvars()

    def _drain(timeout: float = 0.2) -> bytes:
        chunks: list[bytes] = []
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            r, _, _ = _select.select([fd], [], [], min(remaining, 0.05))
            if not r:
                break
            chunk = os.read(fd, 512)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)

    try:
        _tty.setraw(fd, _termios.TCSANOW)

        # 1. Kitty graphics protocol — a=q: query only, no display
        os.write(sys.stdout.fileno(), b"\x1b_Ga=q,i=31,s=1,v=1,m=0;\x1b\\")
        if b"\x1b_G" in _drain(0.25):
            return "kitty"

        # 2. XTVERSION — Alacritty, WezTerm, Windows Terminal all respond with name
        os.write(sys.stdout.fileno(), b"\x1b[>q")
        resp_lower = _drain(0.35).lower()
        if any(name in resp_lower for name in _ITERM2_XTVERSION_NAMES):
            return "iterm2"

        # 3. iTerm2 ReportCellSize — iTerm2 itself responds; most others don't
        os.write(sys.stdout.fileno(), b"\x1b]1337;ReportCellSize\x07")
        resp = _drain(0.3)
        if b"1337" in resp or b"ReportCellSize" in resp:
            return "iterm2"

        return _detect_image_protocol_envvars()

    except Exception:
        return _detect_image_protocol_envvars()
    finally:
        try:
            _drain(0.05)  # flush stray response bytes before restoring mode
        except Exception:
            pass
        try:
            _termios.tcsetattr(fd, _termios.TCSADRAIN, old_attrs)
        except Exception:
            pass


_IMAGE_PROTOCOL: str | None = None  # set in main() before TUI starts
_IMAGE_PROTOCOL_FAILED: bool = False  # set True if inline emit fails repeatedly


def _emit_inline_image(jpg_bytes: bytes, width_px: int = 400) -> None:
    """
    Write an inline image to the real TTY using iTerm2 or Kitty protocol.
    This bypasses Textual entirely — we find the actual /dev/tty and write to it.
    """
    if not jpg_bytes:
        return

    try:
        if _IMAGE_PROTOCOL == "kitty":
            # Kitty graphics protocol: base64 PNG in APC escape
            img = cv2.imdecode(np.frombuffer(jpg_bytes, np.uint8), cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Resize to reasonable pixel width
            h, w = img_rgb.shape[:2]
            new_w = width_px
            new_h = int(h * new_w / w)
            img_rgb = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            pil = Image.fromarray(img_rgb)
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            data = base64.b64encode(buf.getvalue()).decode()
            # Chunk into 4096-byte pieces
            seq = ""
            for i in range(0, len(data), 4096):
                chunk = data[i : i + 4096]
                more = "1" if i + 4096 < len(data) else "0"
                seq += "\x1b_G" + f"a=T,f=100,m={more};" + chunk + "\x1b\\"
            return seq
        else:  # iterm2 (works in Alacritty, WezTerm, iTerm2)
            data = base64.b64encode(jpg_bytes).decode()
            size = len(jpg_bytes)
            return (
                "\x1b]1337;File=inline=1;size="
                + str(size)
                + ";width=auto;height=auto;preserveAspectRatio=1:"
                + data
                + "\x07"
            )
    except Exception:
        return None


def _frame_to_rich_text(max_w: int = 76, max_h: int = 40) -> "Text":
    """
    Fallback half-block renderer when inline image protocols unavailable.
    Also used as a fast placeholder while the image is loading.
    """
    from rich.text import Text
    from rich.style import Style
    from rich.color import Color

    with _frame_lock:
        has_frame = _CV2 and _latest_frame_jpg is not None and _latest_frame_ts != 0
        frame_data = _latest_frame_jpg if has_frame else None

    if not has_frame or frame_data is None:
        t = Text()
        for i in range(max_h // 4):
            t.append(" " * max_w + "\n")
        t.append("NO CAMERA FEED".center(max_w) + "\n", style="bold dim")
        return t

    try:
        img = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return Text("no frame")
        target_w = max_w
        target_h = (max_h * 2) & ~1  # keep even so row+1 is always valid
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(
            img_rgb, (target_w, target_h), interpolation=cv2.INTER_AREA
        )
        t = Text(overflow="fold", no_wrap=True)
        for row in range(0, target_h, 2):
            for col in range(target_w):
                r1, g1, b1 = (
                    int(resized[row, col, 0]),
                    int(resized[row, col, 1]),
                    int(resized[row, col, 2]),
                )
                r2, g2, b2 = (
                    int(resized[row + 1, col, 0]),
                    int(resized[row + 1, col, 1]),
                    int(resized[row + 1, col, 2]),
                )
                t.append(
                    "▀",
                    style=Style(
                        color=Color.from_rgb(r1, g1, b1),
                        bgcolor=Color.from_rgb(r2, g2, b2),
                    ),
                )
            t.append("\n")
        return t
    except Exception as e:
        return Text(f"camera error: {e}")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


def _setup_state() -> None:
    def on_sport(m):
        global _sport_state
        _sport_state = m.get("data", {})

    def on_low(m):
        global _low_state
        _low_state = m.get("data", {})

    if _conn is not None and _conn.datachannel is not None:
        _conn.datachannel.pub_sub.subscribe(RTC_TOPIC["LF_SPORT_MOD_STATE"], on_sport)
        _conn.datachannel.pub_sub.subscribe(RTC_TOPIC["LOW_STATE"], on_low)


def _state_summary() -> dict:
    s, l = _sport_state, _low_state
    return {
        "position": s.get("position", [0, 0, 0]),
        "velocity": s.get("velocity", [0, 0, 0]),
        "rpy_rad": s.get("imu_state", {}).get("rpy", [0, 0, 0]),
        "body_height": s.get("body_height", 0),
        "gait": s.get("gait_type", 0),
        "battery_pct": l.get("bms_state", {}).get("soc", "?"),
        "battery_v": l.get("bms_state", {}).get("voltage", "?"),
        # range_obstacle: [front, left, right, back] obstacle distances from LiDAR (metres)
        # 0 or negative means no obstacle / sensor not reporting
        "range_obstacle": s.get("range_obstacle", [0, 0, 0, 0]),
    }


def _forward_obstacle_m() -> float:
    """Return forward obstacle distance in metres. 0 means no reading."""
    r = _sport_state.get("range_obstacle", [0, 0, 0, 0])
    if isinstance(r, list) and len(r) > 0:
        return float(r[0])
    return 0.0


# ---------------------------------------------------------------------------
# Radio / audio helpers
# ---------------------------------------------------------------------------


async def _start_radio(url: str) -> str:
    global _radio_player
    try:
        from aiortc.contrib.media import MediaPlayer
    except ImportError:
        return "error: aiortc MediaPlayer not available"
    if _conn is None:
        return "error: not connected"
    await _stop_radio()
    try:
        _radio_player = MediaPlayer(url)
        # Reuse the existing audio sender so no SDP renegotiation is needed
        senders = _conn.pc.getSenders()
        audio_sender = next((s for s in senders if s.track and s.track.kind == "audio"), None)
        if audio_sender:
            await audio_sender.replaceTrack(_radio_player.audio)
        else:
            _conn.pc.addTrack(_radio_player.audio)
        return f"radio playing: {url}"
    except Exception as e:
        _radio_player = None
        return f"radio error: {e}"


async def _stop_radio() -> str:
    global _radio_player
    if _radio_player is None:
        return "no audio playing"
    try:
        senders = _conn.pc.getSenders() if _conn else []
        audio_sender = next((s for s in senders if s.track and s.track.kind == "audio"), None)
        if audio_sender:
            await audio_sender.replaceTrack(None)
        _radio_player.audio.stop()
    except Exception:
        pass
    _radio_player = None
    return "audio stopped"


# ---------------------------------------------------------------------------
# Whisper / voice transcription
# ---------------------------------------------------------------------------


def _get_whisper_model(size: str = "base"):
    global _whisper_model
    with _whisper_lock:
        if _whisper_model is None:
            from faster_whisper import WhisperModel
            _whisper_model = WhisperModel(size, device="cpu", compute_type="int8")
        return _whisper_model


def _transcribe_audio(audio_np, sample_rate: int = 16000) -> str:
    """Transcribe a float32 numpy array with faster-whisper. Returns text."""
    model = _get_whisper_model()
    segments, _ = model.transcribe(audio_np, beam_size=5, language="en")
    return " ".join(s.text for s in segments).strip()


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "move",
            "description": "Move the robot by displacement. x=forward/back metres, y=left/right metres. Do NOT use z for rotation — use the turn() tool instead which handles rotation correctly.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "number", "description": "Forward (+) / back (-)"},
                    "y": {"type": "number", "description": "Left (+) / right (-)"},
                    "z": {
                        "type": "number",
                        "description": "Yaw radians, left(+) right(-)",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "turn",
            "description": "Turn the robot in place. degrees: positive=left/CCW, negative=right/CW. Handles any angle including 90, 180, 360. Always use this for rotation, never move(z=...).",
            "parameters": {
                "type": "object",
                "properties": {"degrees": {"type": "number"}},
                "required": ["degrees"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stance",
            "description": "Change the robot's stance.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pose": {
                        "type": "string",
                        "enum": [
                            "stand_up",
                            "stand_down",
                            "balance_stand",
                            "recovery_stand",
                            "sit",
                            "stop",
                            "back_stand",
                        ],
                    }
                },
                "required": ["pose"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "trick",
            "description": "Perform a trick or social gesture.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "enum": [
                            "hello",
                            "stretch",
                            "wiggle_hips",
                            "scrape",
                            "wallow",
                            "show_heart",
                            "dance1",
                            "dance2",
                            "front_flip",
                            "back_flip",
                            "left_flip",
                            "right_flip",
                            "handstand",
                            "front_jump",
                            "front_pounce",
                        ],
                    }
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "led",
            "description": "Set the robot's body LED color.",
            "parameters": {
                "type": "object",
                "properties": {
                    "color": {
                        "type": "string",
                        "enum": [
                            "white",
                            "red",
                            "yellow",
                            "blue",
                            "green",
                            "cyan",
                            "purple",
                        ],
                    },
                    "duration": {"type": "integer", "description": "seconds"},
                },
                "required": ["color"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "look",
            "description": "Tilt the robot body. roll/pitch/yaw in radians.",
            "parameters": {
                "type": "object",
                "properties": {
                    "roll": {"type": "number"},
                    "pitch": {"type": "number"},
                    "yaw": {"type": "number"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_speed",
            "description": "Set walking speed level.",
            "parameters": {
                "type": "object",
                "properties": {
                    "level": {
                        "type": "integer",
                        "enum": [0, 1, 2],
                        "description": "0=slow 1=normal 2=fast",
                    },
                },
                "required": ["level"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_volume",
            "description": "Set the robot's speaker volume (0=silent, 10=max).",
            "parameters": {
                "type": "object",
                "properties": {
                    "level": {"type": "integer", "description": "Volume 0–10"},
                },
                "required": ["level"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "play_radio",
            "description": "Stream internet radio or any HTTP audio URL through the robot's speaker. Default station: https://nashe1.hostingradio.ru:80/ultra-128.mp3",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "HTTP audio stream URL (MP3/AAC). Defaults to https://nashe1.hostingradio.ru:80/ultra-128.mp3"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stop_audio",
            "description": "Stop any audio currently playing through the robot's speaker.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

# ---------------------------------------------------------------------------
# Execute a single tool call
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Velocity-loop helper — Move API sends velocity (m/s, rad/s), not displacement.
# Must be called repeatedly at ~20ms to keep robot moving.
# ---------------------------------------------------------------------------
MOVE_TICK_HZ = 25  # send rate in Hz
MOVE_TICK_S = 1.0 / MOVE_TICK_HZ

# Minimum safe forward distance (metres). Robot stops if LiDAR sees closer obstacle.
# Go2 LiDAR has ~0.1m minimum range so anything under 0.35m is "about to hit"
OBSTACLE_STOP_M = 0.35


async def _velocity_loop(vx: float, vy: float, vyaw: float, duration_s: float) -> bool:
    """
    Send Move velocity commands in a loop for duration_s seconds.
    Stops early if LiDAR detects a forward obstacle within OBSTACLE_STOP_M.
    Only checks obstacle distance for forward motion (vx > 0).
    """
    loop = asyncio.get_event_loop()
    deadline = loop.time() + duration_s
    ok = True
    while loop.time() < deadline:
        # Obstacle guard: only for forward motion
        if vx > 0:
            dist = _forward_obstacle_m()
            if 0 < dist < OBSTACLE_STOP_M:
                await _mcf("Move", {"x": 0, "y": 0, "z": 0})
                return "obstacle"
        r = await _mcf("Move", {"x": vx, "y": vy, "z": vyaw})
        if not r.get("ok", False):
            ok = False
        # Sleep only as long as needed — don't overshoot the deadline
        remaining = deadline - loop.time()
        if remaining > 0:
            await asyncio.sleep(min(MOVE_TICK_S, remaining))
    # Send stop
    await _mcf("Move", {"x": 0, "y": 0, "z": 0})
    return ok


async def run_tool(name: str, args: dict) -> str:
    if name == "move":
        # x/y are distances in metres; z is yaw in radians
        # Convert to velocity commands with appropriate durations
        # Walk speed ~0.5 m/s, yaw rate ~0.8 rad/s
        WALK_SPEED = 0.5  # m/s
        STRAFE_SPEED = 0.3  # m/s
        YAW_RATE = 0.8  # rad/s

        x = float(args.get("x", 0))
        y = float(args.get("y", 0))
        z = float(args.get("z", 0))  # discouraged but handle it

        errors = []
        # Handle x (forward/back)
        if abs(x) > 0.01:
            dur = abs(x) / WALK_SPEED
            result = await _velocity_loop(WALK_SPEED * (1 if x > 0 else -1), 0, 0, dur)
            if result == "obstacle":
                return f"move(x={x:.2f}m) → stopped: obstacle detected within {OBSTACLE_STOP_M}m"
            if not result:
                errors.append("x")
            await asyncio.sleep(0.2)
        # Handle y (strafe left/right)
        if abs(y) > 0.01:
            dur = abs(y) / STRAFE_SPEED
            ok = await _velocity_loop(0, STRAFE_SPEED * (1 if y > 0 else -1), 0, dur)
            if not ok:
                errors.append("y")
            await asyncio.sleep(0.2)
        # Handle z (yaw) if someone passes it directly
        if abs(z) > 0.01:
            dur = abs(z) / YAW_RATE
            ok = await _velocity_loop(0, 0, YAW_RATE * (1 if z > 0 else -1), dur)
            if not ok:
                errors.append("z")

        status = "ok" if not errors else f"errors on {errors}"
        return f"move(x={x:.2f}m, y={y:.2f}m) → {status}"

    elif name == "turn":
        deg = float(args.get("degrees", 0))
        rad = abs(deg) * 3.14159 / 180.0
        YAW_RATE_CMD = 0.8   # rad/s sent to robot
        YAW_RATE_ACTUAL = 0.6  # empirical actual rotation rate at that command
        duration = rad / YAW_RATE_ACTUAL
        sign = 1.0 if deg > 0 else -1.0
        ok = await _velocity_loop(0, 0, YAW_RATE_CMD * sign, duration)
        status = "ok" if ok else "error"
        return f"turn({deg:.1f}°, {duration:.1f}s) → {status}"

    elif name == "stance":
        pose_map = {
            "stand_up": ("StandUp", None),
            "stand_down": ("StandDown", None),
            "balance_stand": ("BalanceStand", None),
            "recovery_stand": ("RecoveryStand", None),
            "sit": ("Sit", None),
            "stop": ("StopMove", None),
            "back_stand": ("BackStand", None),
        }
        pose = args.get("pose", "balance_stand")
        cmd, param = pose_map.get(pose, ("BalanceStand", None))
        r = await _mcf(cmd, param)
        return f"{pose} → {'ok' if r['ok'] else 'error'}"

    elif name == "trick":
        trick_map = {
            "hello": lambda: _mcf("Hello", timeout=10.0),
            "stretch": lambda: _mcf("Stretch", timeout=10.0),
            "wiggle_hips": lambda: _mcf_raw(1033, timeout=10.0),
            "scrape": lambda: _mcf("Scrape", timeout=10.0),
            "wallow": lambda: _mcf_raw(1021, timeout=10.0),
            "show_heart": lambda: _mcf("Heart", timeout=10.0),
            "dance1": lambda: _mcf("Dance1", timeout=15.0),
            "dance2": lambda: _mcf("Dance2", timeout=15.0),
            "front_flip": lambda: _mcf("FrontFlip", timeout=15.0),
            "back_flip": lambda: _mcf("BackFlip", timeout=15.0),
            "left_flip": lambda: _mcf("LeftFlip", timeout=15.0),
            "right_flip": lambda: _mcf("RightFlip", timeout=15.0),
            "handstand": lambda: _mcf("Handstand", timeout=10.0),
            "front_jump": lambda: _mcf("FrontJump", timeout=10.0),
            "front_pounce": lambda: _mcf("FrontPounce", timeout=10.0),
        }
        t = args.get("name", "hello")
        fn = trick_map.get(t)
        if fn is None:
            return f"unknown trick: {t}"
        r = await fn()
        return f"{t} → {'ok' if r['ok'] else 'error'}"

    elif name == "led":
        color = args.get("color", "white")
        duration = int(args.get("duration", 3))
        if _conn is None or _conn.datachannel is None:
            return "error: no connection"
        resp = await _conn.datachannel.pub_sub.publish_request_new(
            RTC_TOPIC["VUI"],
            {"api_id": 1007, "parameter": {"color": color, "time": duration}},
        )
        ok = _code(resp) == 0
        return f"led({color}, {duration}s) → {'ok' if ok else 'error'}"

    elif name == "look":
        roll = float(args.get("roll", 0))
        pitch = float(args.get("pitch", 0))
        yaw = float(args.get("yaw", 0))
        r = await _mcf("Euler", {"x": roll, "y": pitch, "z": yaw})
        return f"look(roll={roll:.2f}, pitch={pitch:.2f}, yaw={yaw:.2f}) → {'ok' if r['ok'] else 'error'}"

    elif name == "set_speed":
        level = int(args.get("level", 1))
        r = await _mcf("SpeedLevel", {"data": level})
        return f"set_speed({level}) → {'ok' if r['ok'] else 'error'}"

    elif name == "set_volume":
        level = max(0, min(10, int(args.get("level", 5))))
        if _conn is None or _conn.datachannel is None:
            return "error: no connection"
        resp = await _conn.datachannel.pub_sub.publish_request_new(
            RTC_TOPIC["VUI"], {"api_id": 1003, "parameter": {"volume": level}}
        )
        ok = _code(resp) == 0
        return f"volume({level}) → {'ok' if ok else 'error'}"

    elif name == "play_radio":
        url = args.get("url", "")
        if not url:
            return "error: url required"
        return await _start_radio(url)

    elif name == "stop_audio":
        return await _stop_radio()

    else:
        return f"unknown tool: {name}"


# ---------------------------------------------------------------------------
# Ollama — fixed streaming + conversation history management
# ---------------------------------------------------------------------------

MAX_HISTORY_TURNS = 20  # keep last N user/assistant pairs to prevent context overflow


def _ollama_chat(
    messages: list[dict],
    model: str,
    ollama_url: str,
    token_fn=None,  # optional callback(str) called with each streamed token
) -> dict:
    """
    Call Ollama /api/chat with streaming.
    token_fn(chunk) is called for each text token as it arrives.
    Returns {"role":"assistant","content":...,"tool_calls":[...]}.
    """
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {"temperature": 0.2},
        "think": True,  # enable Qwen3 / DeepSeek-R1 thinking mode
        "tools": TOOLS,
    }

    max_attempts = 3
    resp = None
    last_exc: Exception = RuntimeError("unknown")
    for attempt in range(max_attempts):
        try:
            resp = requests.post(
                f"{ollama_url}/api/chat",
                json=payload,
                stream=True,
                timeout=120,
            )
            resp.raise_for_status()
            break  # success
        except requests.exceptions.ConnectionError as e:
            last_exc = ConnectionError(f"Cannot reach Ollama at {ollama_url}: {e}")
        except requests.exceptions.Timeout:
            last_exc = TimeoutError("Ollama request timed out after 120s")
        except Exception as e:
            last_exc = e
        if attempt < max_attempts - 1:
            backoff = 0.5 * (2 ** attempt)  # 0.5s, 1.0s
            time.sleep(backoff)
    if resp is None:
        raise last_exc

    content_parts = []
    thinking_parts = []
    tool_calls = []

    for line in resp.iter_lines():
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue

        msg = data.get("message", {})

        # Ollama 0.7+ streams thinking tokens in a separate "thinking" field
        think_chunk = msg.get("thinking", "")
        if think_chunk:
            thinking_parts.append(think_chunk)
            if token_fn:
                token_fn(("think", think_chunk))

        chunk = msg.get("content", "")
        if chunk:
            content_parts.append(chunk)
            if token_fn:
                token_fn(("content", chunk))

        # Ollama delivers tool_calls as a complete list in one chunk (not streamed)
        if msg.get("tool_calls"):
            tool_calls = msg["tool_calls"]

    return {
        "role": "assistant",
        "content": "".join(content_parts),
        "thinking": "".join(thinking_parts),
        "tool_calls": tool_calls,
    }


def _trim_history(history: list[dict]) -> list[dict]:
    """
    Keep system prompt + last MAX_HISTORY_TURNS user/assistant/tool turns.
    Strips embedded images from older messages to save context space.
    """
    system = [m for m in history if m.get("role") == "system"]
    turns = [m for m in history if m.get("role") != "system"]

    # Each agentic step now produces: assistant + tool + user(camera obs)
    # = 3 messages per step, plus original user message = 4 per turn
    # Keep last N turns worth of messages
    max_msgs = MAX_HISTORY_TURNS * 4
    if len(turns) > max_msgs:
        turns = turns[-max_msgs:]

    # Strip images from all user messages except the 2 most recent
    # (keep latest camera obs + the one before so model has context)
    user_indices = [
        i for i, m in enumerate(turns) if m.get("role") == "user" and "images" in m
    ]
    keep_image_indices = set(user_indices[-2:])  # keep last 2 camera frames

    trimmed = []
    for i, m in enumerate(turns):
        if m.get("role") == "user" and "images" in m and i not in keep_image_indices:
            m = {k: v for k, v in m.items() if k != "images"}
        trimmed.append(m)

    return system + trimmed


# ---------------------------------------------------------------------------
# Agentic turn processor
# ---------------------------------------------------------------------------


async def process_turn(
    user_input: str,
    history: list[dict],
    model: str,
    ollama_url: str,
    use_camera: bool,
    cam_quality: int,
    log_fn=None,  # callback(str) for progress updates
) -> str:
    """
    Run one full conversation turn with tool calling loop.
    Updates history in-place. Returns the final assistant text.
    log_fn is called with status strings during processing.
    """

    def log(msg):
        if log_fn:
            log_fn(msg)

    frame_b64 = _get_frame_b64(cam_quality) if use_camera else None
    state = _state_summary()

    user_msg: dict = {
        "role": "user",
        "content": (
            f"{user_input}\n\n[Robot state: {json.dumps(state)}]"
            + ("\n[Camera frame attached]" if frame_b64 else "\n[No camera]")
        ),
    }
    if frame_b64:
        user_msg["images"] = [frame_b64]

    history.append(user_msg)

    max_iterations = 10
    for iteration in range(max_iterations):
        trimmed = _trim_history(history)

        log(f"Thinking... (step {iteration + 1})")

        try:
            loop = asyncio.get_event_loop()
            assistant_msg = await loop.run_in_executor(
                None, lambda: _ollama_chat(trimmed, model, ollama_url)
            )
        except (ConnectionError, TimeoutError) as e:
            history.pop()  # remove user msg so we don't corrupt history
            raise

        # IMPORTANT: always add role to the assistant message before appending
        assistant_msg["role"] = "assistant"
        history.append(assistant_msg)

        # Show thinking tokens if present
        thinking = assistant_msg.get("thinking", "").strip()
        if thinking and log_fn:
            # Indent and truncate thinking for display — it can be very long
            lines = thinking.splitlines()
            preview = "\n    ".join(lines[:8])  # show up to 8 lines
            if len(lines) > 8:
                preview += f"\n    … ({len(lines) - 8} more lines)"
            log_fn(f"[dim italic]💭 {preview}[/dim italic]")

        tool_calls = assistant_msg.get("tool_calls", [])

        if not tool_calls:
            return assistant_msg.get("content", "").strip()

        # Execute tools — show camera tool calls prominently
        tool_results = []
        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception as _json_err:
                    print(f"WARNING: could not parse args for tool '{name}': {_json_err!r} — raw: {args!r}", file=sys.stderr)
                    args = {}

            # Show all tool calls including camera-related ones
            args_str = json.dumps(args, separators=(",", ":")) if args else ""
            log(f"→ {name}({args_str})")
            try:
                result = await run_tool(name, args)
            except Exception as e:
                import traceback

                traceback.print_exc()  # goes to log file, not terminal
                result = f"error: {type(e).__name__}: {e}"
            log(f"  ✓ {result}")
            tool_results.append({"tool": name, "result": result})

        # Ollama ignores images on role:tool messages — vision only works on role:user.
        # So: append a plain tool result, then inject a user observation message
        # with the fresh camera frame so the model actually sees it.
        fresh_frame = _get_frame_b64(cam_quality) if use_camera else None
        fresh_state = _state_summary()

        history.append(
            {
                "role": "tool",
                "content": json.dumps(tool_results),
            }
        )

        # Camera observation injected as a user message so Ollama processes the image
        obs_msg: dict = {
            "role": "user",
            "content": (
                f"[After action — Robot state: {json.dumps(fresh_state)}]"
                + (
                    "\n[Current camera view attached — what do you see? Use this to decide your next action.]"
                    if fresh_frame
                    else "\n[No camera available]"
                )
            ),
        }
        if fresh_frame:
            obs_msg["images"] = [fresh_frame]
        history.append(obs_msg)

    return "Reached maximum tool iterations."


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are controlling a Unitree Go2 robot dog via tool calls.
The user gives natural language instructions. Execute them decisively and efficiently.

HOW THE LOOP WORKS — this is critical to understand:
- You issue ONE tool call at a time. After each tool call completes, you receive a fresh camera frame and updated robot state before deciding what to do next.
- This means you can and should adapt after each action — if you turned and now see your target, stop turning and move toward it. If you moved and hit an obstacle, reassess.
- Do NOT batch multiple tool calls in a single response hoping they'll all succeed — you have eyes between every step, use them.
- Each response should contain exactly ONE tool call, then stop and wait for the camera feedback.

MOVEMENT RULES:
- move(x, y) — x=metres forward/back, y=left/right strafe. Code runs a timed velocity loop at real walking speed.
- turn(degrees) — positive=left/CCW, negative=right/CW. Never use move(z=...) for rotation.
- Good increments: turn(45) or turn(90) to scan, move(x=0.5) to move(x=1.5) for walking.
- Do NOT use tiny increments like move(x=0.1) or turn(5) — use natural human-scale steps.

SEARCH BEHAVIOUR:
- To find something: turn in increments (45–90°), check the camera after each turn, stop when you see the target.
- To approach something: alternate move(x=0.5–1.0) steps with camera checks. Stop when close enough.
- Describe what you see in the camera after each step so the user knows what's happening.

SAFETY:
- If the robot looks fallen (rpy pitch/roll > 0.5 rad), use stance(recovery_stand) first.
- If a move() returns "obstacle detected", reassess with camera before trying again.

AUDIO:
- play_radio(url) — stream internet radio or any HTTP audio URL through the robot's speaker.
- stop_audio() — stop playback.

RESPONSE FORMAT:
- After a tool call completes: briefly describe what you see and what you're doing next (1–2 sentences).
- When the task is fully done: say so clearly and stop issuing tool calls."""

# ---------------------------------------------------------------------------
# Textual TUI
# ---------------------------------------------------------------------------

CSS = """
Screen {
    layout: vertical;
}

#top-bar {
    height: 3;
    background: $panel;
    border-bottom: solid $accent;
}

#status-label {
    width: 1fr;
    content-align: left middle;
    padding: 0 1;
}

#battery-label {
    width: 20;
    content-align: right middle;
    padding: 0 1;
}

#main-area {
    layout: horizontal;
    height: 1fr;
}

#chat-panel {
    width: 1fr;
    height: 100%;
    border-right: solid $accent;
}

#chat-log {
    width: 100%;
    height: 1fr;
    scrollbar-size: 1 1;
}

#camera-panel {
    width: 80;
    height: 100%;
    background: $surface;
}

#camera-title {
    height: 1;
    background: $accent;
    content-align: center middle;
    color: $background;
    text-style: bold;
}

#camera-view {
    width: 100%;
    height: 1fr;
    overflow: hidden;
    padding: 0;
}

#input-area {
    height: 5;
    border-top: solid $accent;
    padding: 0 1;
    layout: vertical;
}

#hint-label {
    height: 1;
    color: $text-muted;
    content-align: left middle;
}

/* ── Error panel ── */
#error-panel {
    height: auto;
    max-height: 6;
    background: $error 15%;
    border-top: solid $error;
    padding: 0 1;
    display: none;
}

#error-panel.visible {
    display: block;
}

#error-log {
    width: 100%;
    height: auto;
    scrollbar-size: 1 1;
}

/* ── Overlay screens (help / prompt viewer) ── */
.overlay {
    width: 80%;
    height: 80%;
    background: $surface;
    border: double $accent;
    padding: 1 2;
    layer: overlay;
    offset: 10% 10%;
}

.overlay-title {
    text-style: bold;
    color: $accent;
    height: 1;
    margin-bottom: 1;
}

.overlay ScrollableContainer {
    height: 1fr;
    scrollbar-size: 1 1;
}

.overlay .close-hint {
    height: 1;
    color: $text-muted;
    content-align: right middle;
    margin-top: 1;
}

#user-input {
    height: 3;
}
"""


# ---------------------------------------------------------------------------
# Custom camera widget — uses iTerm2/Kitty inline images or half-block fallback
# ---------------------------------------------------------------------------


class InlineCameraWidget(Static):
    """
    Displays the robot's camera feed.
    - On terminals supporting iTerm2 protocol (Alacritty, WezTerm, iTerm2, etc.):
      emits a real inline image escape sequence for crisp full-color display.
    - Falls back to unicode half-block art otherwise.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_emit_ts: float = 0.0
        self._use_inline = False
        self._consecutive_failures: int = 0

    def on_mount(self) -> None:
        self._use_inline = _IMAGE_PROTOCOL in ("iterm2", "kitty")

    def render(self):
        """Called by Textual to get the renderable for this widget."""
        if self._use_inline and not _IMAGE_PROTOCOL_FAILED:
            # For inline protocol, return a placeholder — the actual image
            # is emitted via _emit_to_tty() called from refresh hook.
            return Text(f" 📷 {_IMAGE_PROTOCOL} ", style="dim")
        else:
            return _frame_to_rich_text()

    def on_idle(self) -> None:
        """After each render cycle, emit inline image if protocol supports it."""
        if self._use_inline and not _IMAGE_PROTOCOL_FAILED:
            with _frame_lock:
                ts = _latest_frame_ts
                frame_data = _latest_frame_jpg if ts > 0 else None
            if frame_data and ts != self._last_emit_ts:
                self._last_emit_ts = ts
                self._emit_to_tty(frame_data)

    def _emit_to_tty(self, jpg_bytes: bytes) -> None:
        """Write iTerm2/Kitty image escape to the real TTY."""
        global _IMAGE_PROTOCOL_FAILED
        try:
            seq = _emit_inline_image(jpg_bytes, width_px=500)
            if not seq:
                return
            region = self.content_region
            tty_escape = (
                "\x1b["
                + str(region.y + 1)
                + ";"
                + str(region.x + 1)
                + "H"  # cursor position
                + seq
            )
            encoded = tty_escape.encode()
            # os.write(1) writes directly to fd 1, bypassing Python buffering
            # and Textual's stream wrapping — the only reliable method in WSL
            # where /dev/tty is locked by Textual and silently fails.
            try:
                os.write(1, encoded)
            except OSError:
                # Fall back to /dev/tty on non-WSL systems
                with open("/dev/tty", "wb") as tty:
                    tty.write(encoded)
            self._consecutive_failures = 0
        except Exception:
            self._consecutive_failures += 1
            if self._consecutive_failures >= 3:
                _IMAGE_PROTOCOL_FAILED = True
                traceback.print_exc()


HELP_TEXT = """
╔══════════════════════════════════════════════════════════════════╗
║                    GO2 NATURAL LANGUAGE CLI                       ║
╚══════════════════════════════════════════════════════════════════╝

MOVEMENT
  "walk forward 1 metre"         move(x=1.0)
  "back up half a metre"         move(x=-0.5)
  "strafe left"                  move(y=0.5)
  "turn left 90 degrees"         turn(degrees=90)
  "spin around"                  turn(degrees=180)
  "turn to face the door"        auto-navigates with camera

SEARCH & NAVIGATION
  "find the red chair"           spins, checks camera each step
  "go to the kitchen"            moves + turns toward target
  "what do you see?"             describe current camera view
  "look around and describe the room"

STANCES & POSES
  "stand up"  /  "sit down"  /  "lie down"
  "recovery stand"               if robot has fallen
  "balance stand"                stable upright position

TRICKS  (say any of these naturally)
  hello / wave            stretch            wiggle hips
  scrape / paw            wallow             show heart
  dance 1 / dance 2       front flip         backflip
  left flip / right flip  handstand          front jump
  front pounce

LED
  "set led to blue"              led(r=0, g=0, b=255)
  "turn off the lights"          led(r=0, g=0, b=0)
  "flash red for 2 seconds"      led(r=255, g=0, b=0, duration=2)

SPEED
  "move faster"                  set_speed(level=2)   (0=slow 1=normal 2=fast)
  "slow down"                    set_speed(level=0)

LOOK / CAMERA
  "what's in front of you?"      attaches camera frame to next query
  "describe what you see"

ROBOT STATE  (F1 or type 'state')
  Shows position, velocity, battery %, rpy, gait type, LiDAR distances

RADIO / AUDIO  (just tell the model naturally)
  "play the radio"                plays default station
  "play <url>"                    streams any HTTP audio URL
  "stop the music"                stops playback

VOICE INPUT
  Ctrl+M                          toggle mic recording on/off
                                  (transcribed with Whisper, submitted as command)

KEYBOARD SHORTCUTS
  Enter          Send command
  Ctrl+M         Toggle microphone recording
  Ctrl+L         Clear conversation history
  Ctrl+E         Toggle error panel
  F1             Show robot state in chat
  F2             This help screen
  F3             Show current system prompt
  Ctrl+C         Quit (stops robot first)

BUILT-IN TEXT COMMANDS
  state          Print robot state
  clear          Clear conversation
  model <name>   Switch Ollama model (e.g. model llava)
"""

from textual.screen import ModalScreen


class HelpScreen(ModalScreen):
    """F2 — full help overlay."""

    BINDINGS = [("escape", "dismiss", "Close"), ("f2", "dismiss", "Close")]

    def compose(self) -> ComposeResult:
        with Vertical(classes="overlay"):
            yield Static(
                "[bold cyan]GO2 CLI — Help[/bold cyan]  (Esc to close)",
                classes="overlay-title",
            )
            with ScrollableContainer():
                yield Static(HELP_TEXT, markup=False)

    def on_key(self, event) -> None:
        self.dismiss()


class PromptScreen(ModalScreen):
    """F3 — system prompt viewer."""

    BINDINGS = [("escape", "dismiss", "Close"), ("f3", "dismiss", "Close")]

    def compose(self) -> ComposeResult:
        with Vertical(classes="overlay"):
            yield Static(
                "[bold cyan]Current System Prompt[/bold cyan]  (Esc to close)",
                classes="overlay-title",
            )
            with ScrollableContainer():
                yield Static(SYSTEM_PROMPT, markup=False)

    def on_key(self, event) -> None:
        self.dismiss()


class Go2App(App):
    CSS = CSS
    TITLE = "Go2 CLI"
    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+l", "clear_chat", "Clear chat"),
        ("ctrl+e", "toggle_errors", "Errors"),
        ("ctrl+m", "toggle_mic", "Mic"),
        ("f1", "show_state", "State"),
        ("f2", "show_help", "Help"),
        ("f3", "show_prompt", "Prompt"),
    ]

    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.args = args
        self.model = args.model
        self.ollama_url = args.ollama
        self.use_camera = not args.no_camera
        self.history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        self._processing = False
        self._camera_task = None
        self._think_timer = None
        self._think_idx = 0
        self._queue: list[str] = []
        self._recording = False
        self._recording_chunks: list = []
        self._recording_stream = None

    def compose(self) -> ComposeResult:
        # Top status bar
        with Horizontal(id="top-bar"):
            yield Label("● Go2 Connected", id="status-label")
            yield Label("🔋 ?%", id="battery-label")

        # Main area
        with Horizontal(id="main-area"):
            # Chat panel (left)
            with Vertical(id="chat-panel"):
                yield RichLog(
                    id="chat-log",
                    highlight=True,
                    markup=True,
                    wrap=True,
                    auto_scroll=True,
                )

            # Camera panel (right)
            with Vertical(id="camera-panel"):
                yield Label(" 📷 Camera ", id="camera-title")
                yield InlineCameraWidget(id="camera-view")

        # Error panel (hidden until errors arrive)
        with Vertical(id="error-panel"):
            yield RichLog(
                id="error-log",
                highlight=False,
                markup=True,
                wrap=True,
                auto_scroll=True,
                max_lines=20,
            )

        # Input area
        with Vertical(id="input-area"):
            yield Label(
                "Enter: send  │  Ctrl+M: mic  │  Ctrl+L: clear  │  Ctrl+E: errors  │  F1: state  │  F2: help  │  F3: prompt  │  Ctrl+C: quit",
                id="hint-label",
            )
            yield Input(placeholder="Type a command...", id="user-input")

    def on_mount(self) -> None:
        _err_capture._app = self  # wire up error capture to this app instance
        self.log_chat(
            "[bold cyan]Go2 CLI[/bold cyan] ready. "
            f"Model: [yellow]{self.model}[/yellow]  "
            f"Ollama: [dim]{self.ollama_url}[/dim]"
        )
        self.log_chat(
            "[dim]Type commands below. The robot awaits your instructions.[/dim]"
        )
        self.query_one("#user-input", Input).focus()
        # Start camera update loop
        self.set_interval(1.0, self._refresh_camera)
        # Start battery/state update loop
        self.set_interval(5.0, self._refresh_status)

    def _refresh_camera(self) -> None:
        """Render camera using inline image protocol if available, else halfblock."""
        cam = self.query_one("#camera-view", InlineCameraWidget)
        cam.refresh()

    def _refresh_status(self) -> None:
        state = _state_summary()
        batt = state.get("battery_pct", "?")
        label = self.query_one("#battery-label", Label)
        label.update(f"🔋 {batt}%")

    def log_chat(self, message: str, markup: bool = True) -> None:
        log = self.query_one("#chat-log", RichLog)
        if markup:
            log.write(Text.from_markup(message))
        else:
            log.write(message)
        self._stream_line_written = False  # new non-stream line resets stream state

    def push_error(self, msg: str) -> None:
        """Receive an error string and show it in the error panel."""
        panel = self.query_one("#error-panel")
        panel.add_class("visible")
        log = self.query_one("#error-log", RichLog)
        # Show only the most relevant line (last non-empty line of traceback)
        lines = [l for l in msg.splitlines() if l.strip()]
        display = lines[-1] if lines else msg
        log.write(
            Text.from_markup(f"[bold red]ERR[/bold red] [dim]{escape(display)}[/dim]")
        )

    def action_toggle_errors(self) -> None:
        """Ctrl+E — toggle error panel visibility."""
        panel = self.query_one("#error-panel")
        if "visible" in panel.classes:
            panel.remove_class("visible")
        else:
            panel.add_class("visible")

    def action_show_help(self) -> None:
        """F2 — show help overlay."""
        self.push_screen(HelpScreen())

    def action_show_prompt(self) -> None:
        """F3 — show system prompt overlay."""
        self.push_screen(PromptScreen())

    def action_clear_chat(self) -> None:
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]
        log = self.query_one("#chat-log", RichLog)
        log.clear()
        self.log_chat("[dim]Conversation cleared.[/dim]")

    def action_show_state(self) -> None:
        state = _state_summary()
        self.log_chat(
            f"[bold]Robot State:[/bold] {escape(json.dumps(state, indent=2))}"
        )

    def on_input_submitted(self, event: Input.Submitted) -> None:
        user_input = event.value.strip()
        if not user_input:
            return

        inp = self.query_one("#user-input", Input)
        inp.value = ""
        inp.focus()

        if self._processing:
            self._queue.append(user_input)
            self.log_chat(f"[dim]⏳ Queued: {escape(user_input)}[/dim]")
            return

        # Handle built-in commands
        if user_input.lower() == "state":
            self.action_show_state()
            return
        if user_input.lower() == "clear":
            self.action_clear_chat()
            return
        if user_input.lower().startswith("model "):
            self.model = user_input.split(None, 1)[1].strip()
            self.log_chat(f"[dim]Model switched to: {self.model}[/dim]")
            return
        self.log_chat(f"[bold green]You:[/bold green] {escape(user_input)}")
        self._run_turn(user_input)

    @work(exclusive=False, thread=True)
    def _run_turn(self, user_input: str) -> None:
        """Runs in a real background thread so call_from_thread works correctly.
        Spins up its own event loop to drive the async robot/Ollama code."""
        import asyncio as _asyncio

        self.call_from_thread(self._set_processing, True)

        def log_progress(msg):
            if not isinstance(msg, str):
                return
            # Thinking lines already have full Rich markup — pass through as-is
            if msg.startswith("[dim italic]💭"):
                _main_loop.call_soon_threadsafe(self.log_chat, f"  {msg}")
            else:
                _main_loop.call_soon_threadsafe(
                    self.log_chat, f"[dim]  {escape(msg)}[/dim]"
                )

        # Robot coroutines must run on the main loop (where WebRTC lives).
        # Ollama HTTP is synchronous (requests), so process_turn is safe to
        # drive from a thread as long as robot tool calls are dispatched via
        # run_coroutine_threadsafe back to the main loop.
        import concurrent.futures as _cf

        future = _asyncio.run_coroutine_threadsafe(
            process_turn(
                user_input=user_input,
                history=self.history,
                model=self.model,
                ollama_url=self.ollama_url,
                use_camera=self.use_camera,
                cam_quality=self.args.quality,
                log_fn=log_progress,
            ),
            _main_loop,
        )
        try:
            response = future.result(timeout=300)
            if response:
                self.call_from_thread(
                    self.log_chat,
                    f"[bold magenta]Go2:[/bold magenta] {escape(response)}",
                )
            else:
                self.call_from_thread(
                    self.log_chat, "[dim]Go2: (no text response)[/dim]"
                )
        except ConnectionError as e:
            self.call_from_thread(
                self.log_chat, f"[bold red]ERROR:[/bold red] {escape(str(e))}"
            )
            self.call_from_thread(
                self.log_chat, "[dim]Is Ollama running? Try: ollama serve[/dim]"
            )
            if self.history and self.history[-1].get("role") == "user":
                self.history.pop()
        except TimeoutError as e:
            self.call_from_thread(
                self.log_chat, f"[bold red]TIMEOUT:[/bold red] {escape(str(e))}"
            )
        except Exception as e:
            self.call_from_thread(
                self.log_chat, f"[bold red]ERROR:[/bold red] {escape(str(e))}"
            )
        finally:
            self.call_from_thread(self._set_processing, False)

    # ── Mic / voice input ────────────────────────────────────────────────────

    def action_toggle_mic(self) -> None:
        if not _HAS_SD:
            self.log_chat("[red]Voice input requires sounddevice: uv add sounddevice[/red]")
            return
        if self._recording:
            self._stop_and_transcribe()
        else:
            self._start_recording()

    def _start_recording(self) -> None:
        self._recording_chunks = []
        self._recording_stream = _sd.InputStream(
            samplerate=16000, channels=1, dtype="float32",
            callback=lambda indata, frames, time, status: self._recording_chunks.append(indata.copy()),
        )
        self._recording_stream.start()
        self._recording = True
        self.query_one("#status-label", Label).update("🎤 Recording... (Ctrl+M to stop)")

    @work(exclusive=False, thread=True)
    def _stop_and_transcribe(self) -> None:
        self._recording = False
        if self._recording_stream:
            self._recording_stream.stop()
            self._recording_stream.close()
            self._recording_stream = None

        if not self._recording_chunks:
            self.call_from_thread(self.log_chat, "[dim]No audio captured.[/dim]")
            self.call_from_thread(
                self.query_one("#status-label", Label).update, "● Go2 Connected"
            )
            return

        self.call_from_thread(
            self.query_one("#status-label", Label).update, "⏳ Transcribing..."
        )
        try:
            audio = np.concatenate(self._recording_chunks).flatten()
            try:
                text = _transcribe_audio(audio)
            except ImportError:
                self.call_from_thread(
                    self.log_chat,
                    "[red]Voice input requires faster-whisper: uv add faster-whisper[/red]",
                )
                return
            if text:
                self.call_from_thread(
                    self.log_chat, f"[bold green]You (voice):[/bold green] {escape(text)}"
                )
                self.call_from_thread(self._dispatch_voice, text)
            else:
                self.call_from_thread(self.log_chat, "[dim]No speech detected.[/dim]")
        finally:
            if not self._processing:
                self.call_from_thread(
                    self.query_one("#status-label", Label).update, "● Go2 Connected"
                )

    def _dispatch_voice(self, text: str) -> None:
        if self._processing:
            self._queue.append(text)
            self.log_chat(f"[dim]⏳ Queued: {escape(text)}[/dim]")
        else:
            self._run_turn(text)

    _THINK_FRAMES = ["⏳ Thinking", "⏳ Thinking.", "⏳ Thinking..", "⏳ Thinking..."]

    def _set_processing(self, value: bool) -> None:
        self._processing = value
        if value:
            self._think_idx = 0
            self._think_timer = self.set_interval(0.4, self._tick_thinking)
        else:
            if self._think_timer is not None:
                self._think_timer.stop()
                self._think_timer = None
            self.query_one("#status-label", Label).update("● Go2 Connected")
            self.query_one("#user-input", Input).focus()
            if self._queue:
                next_input = self._queue.pop(0)
                self.log_chat(f"[bold green]You:[/bold green] {escape(next_input)}")
                self._run_turn(next_input)

    def _tick_thinking(self) -> None:
        self._think_idx = (self._think_idx + 1) % len(self._THINK_FRAMES)
        self.query_one("#status-label", Label).update(self._THINK_FRAMES[self._think_idx])

    async def action_quit(self) -> None:
        self.log_chat("[dim]Stopping robot...[/dim]")
        try:
            await _mcf("StopMove")
            await _mcf("BalanceStand")
        except Exception:
            pass
        self.exit()


# ---------------------------------------------------------------------------
# Connect
# ---------------------------------------------------------------------------


async def connect(args: argparse.Namespace) -> None:
    global _conn
    print("Connecting to Go2...", end=" ", flush=True)

    if args.remote:
        _conn = UnitreeWebRTCConnection(
            WebRTCConnectionMethod.Remote,
            serialNumber=args.serial,
            username=args.username,
            password=args.password,
        )
    elif args.serial and not args.ip:
        _conn = UnitreeWebRTCConnection(
            WebRTCConnectionMethod.LocalSTA, serialNumber=args.serial
        )
    elif args.ip:
        _conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=args.ip)
    else:
        _conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalAP)

    await _conn.connect()
    print("connected!")
    _setup_state()

    if not args.no_camera:
        await _start_camera()
        print("Waiting for camera", end="", flush=True)
        for _ in range(40):
            if _latest_frame_jpg is not None and _latest_frame_ts > 0:
                print(" ready!")
                break
            await asyncio.sleep(0.2)
            print(".", end="", flush=True)
        else:
            print("\nWARNING: No camera frame yet — continuing without vision.")
    else:
        print("Camera disabled.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    p = argparse.ArgumentParser(
        description="Go2 natural language TUI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run cli.py --ip 10.0.0.200
  uv run cli.py --ip 10.0.0.200 --model llava
  uv run cli.py --ip 10.0.0.200 --no-camera
        """,
    )
    p.add_argument("--ip", required=True)
    p.add_argument("--serial", default=None)
    p.add_argument("--remote", action="store_true")
    p.add_argument("--username", default=None)
    p.add_argument("--password", default=None)
    p.add_argument("--model", default="qwen3.6:27b")
    p.add_argument("--ollama", default="http://localhost:11434")
    p.add_argument("--no-camera", action="store_true")
    p.add_argument("--quality", default=75, type=int, help="JPEG quality 1-100")
    args = p.parse_args()

    global _main_loop, _IMAGE_PROTOCOL
    _main_loop = asyncio.get_event_loop()

    # Probe for inline image protocol BEFORE Textual takes over the terminal.
    # This runs interactive escape-sequence tests that require a raw TTY.
    print("Probing terminal image protocol...", end=" ", flush=True)
    _IMAGE_PROTOCOL = _probe_image_protocol()
    print(_IMAGE_PROTOCOL)

    await connect(args)

    app = Go2App(args)
    await app.run_async()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown.")
