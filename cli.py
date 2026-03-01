"""
go2_cli.py — Natural language TUI for the Unitree Go2

A full-screen terminal UI where you type commands in plain English:
  - Top-right: live camera feed (sixel/kitty/unicode block art)
  - Main area: scrollable chat history
  - Bottom: input box (submit with Enter)

Install:
    pip uninstall unitree_webrtc_connect -y
    pip install "git+https://github.com/12-hak/unitree_webrtc_connect.git@patch-1"
    pip install opencv-python-headless numpy requests pillow textual

Run:
    python go2_cli.py
    python go2_cli.py --ip 10.0.0.207 --model llava
    python go2_cli.py --no-camera
"""

import asyncio
import argparse
import base64
import json
import sys
import time
import threading
import warnings
import logging
import os
import io

warnings.filterwarnings("ignore")
os.environ["TERM_IMAGE_LOG_LEVEL"] = "error"
# Suppress noisy aiortc H264 decode warnings
logging.getLogger("aiortc").setLevel(logging.ERROR)
logging.getLogger("aiortc.codecs.h264").setLevel(logging.ERROR)
logging.getLogger("aioice").setLevel(logging.ERROR)

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

# ---------------------------------------------------------------------------
# TUI imports
# ---------------------------------------------------------------------------
try:
    from textual.app import App, ComposeResult
    from textual.widgets import (
        RichLog, Input, Static, Label, Header, Footer
    )
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
_sport_state: dict = {}
_low_state: dict = {}
_main_loop: "asyncio.AbstractEventLoop | None" = None

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


async def _mcf_raw(api_id: int, parameter: dict | None = None, timeout: float = 5.0) -> dict:
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
                _latest_frame_jpg = buf.tobytes()
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
    if not _CV2 or _latest_frame_jpg is None:
        return None
    # Use a generous stale threshold — 30s. The camera loop now zeroes
    # _latest_frame_ts if the feed is truly gone.
    if _latest_frame_ts > 0 and (time.time() - _latest_frame_ts) > 30.0:
        return None
    if cv2 is None or np is None:
        return None
    try:
        img = cv2.imdecode(np.frombuffer(_latest_frame_jpg, np.uint8), cv2.IMREAD_COLOR)
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

def _detect_image_protocol() -> str:
    """Detect which inline image protocol the terminal supports."""
    kit_term = os.environ.get("KITTY_WINDOW_ID", "")
    term_prog = os.environ.get("TERM_PROGRAM", "")
    wt_session = os.environ.get("WT_SESSION", "")          # Windows Terminal
    wt_profile = os.environ.get("WT_PROFILE_ID", "")
    vte = os.environ.get("VTE_VERSION", "")

    if kit_term:
        return "kitty"
    if "iTerm" in term_prog:
        return "iterm2"
    if "WezTerm" in term_prog or os.environ.get("TERM_PROGRAM") == "WezTerm":
        return "iterm2"
    if wt_session or wt_profile:
        # Windows Terminal 1.22+ supports iTerm2 inline images
        return "iterm2"
    # Alacritty sets COLORTERM=truecolor and supports iTerm2 protocol
    if os.environ.get("COLORTERM") in ("truecolor", "24bit"):
        # But NOT in WSL with a non-supporting terminal - be conservative
        # Check we're not in a basic xterm/screen/tmux
        term = os.environ.get("TERM", "")
        if term not in ("xterm", "xterm-256color", "screen", "screen-256color", "tmux", "tmux-256color"):
            return "iterm2"
        # xterm-256color with truecolor = probably Alacritty or similar
        return "iterm2"
    return "halfblock"  # fallback

_IMAGE_PROTOCOL = None  # lazy init

def _emit_inline_image(jpg_bytes: bytes, width_px: int = 400) -> None:
    """
    Write an inline image to the real TTY using iTerm2 or Kitty protocol.
    This bypasses Textual entirely — we find the actual /dev/tty and write to it.
    """
    global _IMAGE_PROTOCOL
    if _IMAGE_PROTOCOL is None:
        _IMAGE_PROTOCOL = _detect_image_protocol()

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
                chunk = data[i:i+4096]
                more = "1" if i + 4096 < len(data) else "0"
                seq += "\x1b_G" + f"a=T,f=100,m={more};" + chunk + "\x1b\\"
            return seq
        else:  # iterm2 (works in Alacritty, WezTerm, iTerm2)
            data = base64.b64encode(jpg_bytes).decode()
            size = len(jpg_bytes)
            return "\x1b]1337;File=inline=1;size=" + str(size) + ";width=auto;height=auto;preserveAspectRatio=1:" + data + "\x07"
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

    if not _CV2 or _latest_frame_jpg is None or _latest_frame_ts == 0:
        t = Text()
        for i in range(max_h // 4):
            t.append(" " * max_w + "\n")
        t.append("NO CAMERA FEED".center(max_w) + "\n", style="bold dim")
        return t

    try:
        img = cv2.imdecode(np.frombuffer(_latest_frame_jpg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return Text("no frame")
        target_w = max_w
        target_h = max_h * 2
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(img_rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
        t = Text(overflow="fold", no_wrap=True)
        for row in range(0, target_h - 1, 2):
            for col in range(target_w):
                r1, g1, b1 = int(resized[row, col, 0]), int(resized[row, col, 1]), int(resized[row, col, 2])
                r2, g2, b2 = int(resized[row + 1, col, 0]), int(resized[row + 1, col, 1]), int(resized[row + 1, col, 2])
                t.append("▀", style=Style(
                    color=Color.from_rgb(r1, g1, b1),
                    bgcolor=Color.from_rgb(r2, g2, b2),
                ))
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
    }

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "move",
            "description": "Move the robot by displacement. x=forward/back metres, y=left/right metres, z=yaw radians.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "number", "description": "Forward (+) / back (-)"},
                    "y": {"type": "number", "description": "Left (+) / right (-)"},
                    "z": {"type": "number", "description": "Yaw radians, left(+) right(-)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "turn",
            "description": "Turn the robot. degrees: positive=left, negative=right.",
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
                        "enum": ["stand_up", "stand_down", "balance_stand",
                                 "recovery_stand", "sit", "stop", "back_stand"],
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
                        "enum": ["hello", "stretch", "wiggle_hips", "scrape", "wallow",
                                 "show_heart", "dance1", "dance2", "front_flip",
                                 "back_flip", "left_flip", "right_flip", "handstand",
                                 "front_jump", "front_pounce"],
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
                        "enum": ["white", "red", "yellow", "blue", "green", "cyan", "purple"],
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
                    "level": {"type": "integer", "enum": [0, 1, 2],
                              "description": "0=slow 1=normal 2=fast"},
                },
                "required": ["level"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Execute a single tool call
# ---------------------------------------------------------------------------

async def run_tool(name: str, args: dict) -> str:
    if name == "move":
        x = max(-0.8, min(0.8, float(args.get("x", 0))))
        y = max(-0.4, min(0.4, float(args.get("y", 0))))
        z = max(-1.2, min(1.2, float(args.get("z", 0))))
        r = await _mcf("Move", {"x": x, "y": y, "z": z})
        return f"move({x:.2f}, {y:.2f}, {z:.2f}) → {'ok' if r['ok'] else 'error ' + str(r['code'])}"

    elif name == "turn":
        deg = float(args.get("degrees", 0))
        rad = deg * 3.14159 / 180.0
        r = await _mcf("Move", {"x": 0, "y": 0, "z": rad})
        return f"turn({deg:.1f}°) → {'ok' if r['ok'] else 'error'}"

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
        "tools": TOOLS,
    }

    try:
        resp = requests.post(
            f"{ollama_url}/api/chat",
            json=payload,
            stream=True,
            timeout=120,
        )
        resp.raise_for_status()
    except requests.exceptions.ConnectionError as e:
        raise ConnectionError(f"Cannot reach Ollama at {ollama_url}: {e}")
    except requests.exceptions.Timeout:
        raise TimeoutError("Ollama request timed out after 120s")

    content_parts = []
    tool_calls = []

    for line in resp.iter_lines():
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue

        msg = data.get("message", {})
        chunk = msg.get("content", "")
        if chunk:
            content_parts.append(chunk)
            if token_fn:
                token_fn(chunk)

        # Ollama delivers tool_calls as a complete list in one chunk (not streamed)
        if msg.get("tool_calls"):
            tool_calls = msg["tool_calls"]

    return {
        "role": "assistant",
        "content": "".join(content_parts),
        "tool_calls": tool_calls,
    }


def _trim_history(history: list[dict]) -> list[dict]:
    """
    Keep system prompt + last MAX_HISTORY_TURNS user/assistant/tool turns.
    Strips embedded images from older messages to save context space.
    """
    system = [m for m in history if m.get("role") == "system"]
    turns = [m for m in history if m.get("role") != "system"]

    # Strip images from all but the most recent user message
    trimmed = []
    for i, m in enumerate(turns):
        if m.get("role") == "user" and i < len(turns) - 1:
            # Remove images key from older user messages
            m = {k: v for k, v in m.items() if k != "images"}
        trimmed.append(m)

    # Keep only recent turns
    if len(trimmed) > MAX_HISTORY_TURNS * 3:  # 3 messages per turn (user/assistant/tool)
        trimmed = trimmed[-(MAX_HISTORY_TURNS * 3):]

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
            assistant_msg = _ollama_chat(trimmed, model, ollama_url)
        except (ConnectionError, TimeoutError) as e:
            history.pop()  # remove user msg so we don't corrupt history
            raise

        # IMPORTANT: always add role to the assistant message before appending
        assistant_msg["role"] = "assistant"
        history.append(assistant_msg)

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
                except Exception:
                    args = {}

            # Show all tool calls including camera-related ones
            args_str = json.dumps(args, separators=(',', ':')) if args else ""
            log(f"→ {name}({args_str})")
            try:
                result = await run_tool(name, args)
            except Exception as e:
                result = f"error: {e}"
            log(f"  ✓ {result}")
            tool_results.append({"tool": name, "result": result})

        # Feed results back — use "tool" role which Ollama supports
        history.append({
            "role": "tool",
            "content": json.dumps(tool_results),
        })

    return "Reached maximum tool iterations."

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are controlling a Unitree Go2 robot dog via tool calls.
The user gives natural language instructions. Execute them decisively and efficiently.

You receive:
- A camera image showing what the robot sees (when available)
- The robot's current state (position, battery, etc.)
- The user's instruction

MOVEMENT RULES — follow these exactly:
- move() takes displacement in metres: x=forward/back, y=left/right, z=yaw radians
- Hard limits (enforced by code): |x|≤0.8m, |y|≤0.4m per call — the robot moves at a SLOW walking pace
- Each move() call covers 0.3–0.8m at normal walking speed — this feels natural, do not try to rush
- "a few steps" = move(x=0.5), "walk forward" = chain 2–3 x move(x=0.8) calls, "back up" = move(x=-0.4)
- Chain tool calls in ONE response when a task needs multiple actions (turn THEN move, etc.)
- If the user wants to go far, chain multiple move(x=0.8) calls in sequence — do NOT use tiny 0.1–0.2m moves

DECISION RULES:
- Be decisive. Pick the best action and do it. Don't hedge with tiny test moves.
- Only use the camera to assess the scene BEFORE acting, not to verify after every tiny step
- If the robot looks fallen (rpy pitch/roll > 0.5 rad), use stance(recovery_stand) first
- After completing actions, give a single brief response describing what you did"""

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
        self._checked_protocol = False

    def on_mount(self) -> None:
        global _IMAGE_PROTOCOL
        if _IMAGE_PROTOCOL is None:
            _IMAGE_PROTOCOL = _detect_image_protocol()
        self._use_inline = (_IMAGE_PROTOCOL in ("iterm2", "kitty"))

    def render(self):
        """Called by Textual to get the renderable for this widget."""
        if self._use_inline:
            # For inline protocol, return a placeholder — the actual image
            # is emitted via _emit_to_tty() called from refresh hook.
            return Text(" 📷 live ", style="dim")
        else:
            return _frame_to_rich_text()

    def on_idle(self) -> None:
        """After each render cycle, emit inline image if protocol supports it."""
        if self._use_inline and _latest_frame_jpg and _latest_frame_ts > 0:
            self._emit_to_tty()

    def _emit_to_tty(self) -> None:
        """Write iTerm2/Kitty image escape to the real TTY."""
        if not _latest_frame_jpg:
            return
        try:
            seq = _emit_inline_image(_latest_frame_jpg, width_px=500)
            if seq:
                # Get widget position to move cursor there first
                region = self.content_region
                # Move cursor to widget top-left and emit
                tty_escape = (
                    "\x1b[" + str(region.y + 1) + ";" + str(region.x + 1) + "H"  # cursor position
                    + seq
                )
                # Write to the actual tty fd
                # On WSL, /dev/tty works. On Windows native, use conout$.
                tty_path = "/dev/tty"
                if not os.path.exists(tty_path):
                    tty_path = "CONOUT$"
                try:
                    with open(tty_path, "wb") as tty:
                        tty.write(tty_escape.encode())
                except Exception:
                    # Last resort: write to stdout directly
                    sys.stdout.buffer.write(tty_escape.encode())
                    sys.stdout.buffer.flush()
        except Exception:
            pass  # silently degrade to halfblock


class Go2App(App):
    CSS = CSS
    TITLE = "Go2 CLI"
    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+l", "clear_chat", "Clear chat"),
        ("f1", "show_state", "Robot state"),
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

    def compose(self) -> ComposeResult:
        # Top status bar
        with Horizontal(id="top-bar"):
            yield Label("● Go2 Connected", id="status-label")
            yield Label("🔋 ?%", id="battery-label")

        # Main area
        with Horizontal(id="main-area"):
            # Chat panel (left)
            with Vertical(id="chat-panel"):
                yield RichLog(id="chat-log", highlight=True, markup=True, wrap=True, auto_scroll=True)

            # Camera panel (right)
            with Vertical(id="camera-panel"):
                yield Label(" 📷 Camera ", id="camera-title")
                yield InlineCameraWidget(id="camera-view")

        # Input area
        with Vertical(id="input-area"):
            yield Label(
                "Enter: send  |  Ctrl+L: clear  |  F1: state  |  Ctrl+C: quit",
                id="hint-label",
            )
            yield Input(placeholder="Type a command...", id="user-input")

    def on_mount(self) -> None:
        self.log_chat("[bold cyan]Go2 CLI[/bold cyan] ready. "
                      f"Model: [yellow]{self.model}[/yellow]  "
                      f"Ollama: [dim]{self.ollama_url}[/dim]")
        self.log_chat("[dim]Type commands below. The robot awaits your instructions.[/dim]")
        self.query_one("#user-input", Input).focus()
        # Start camera update loop
        self.set_interval(1.0, self._refresh_camera)
        # Start battery/state update loop
        self.set_interval(5.0, self._refresh_status)

    def _refresh_camera(self) -> None:
        """Render camera using inline image protocol if available, else halfblock."""
        global _IMAGE_PROTOCOL
        if _IMAGE_PROTOCOL is None:
            _IMAGE_PROTOCOL = _detect_image_protocol()

        if _IMAGE_PROTOCOL != "halfblock" and _latest_frame_jpg is not None and _latest_frame_ts > 0:
            # Use the InlineCameraWidget approach — emit to the widget's region
            cam = self.query_one("#camera-view", InlineCameraWidget)
            cam.refresh()
        else:
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



    def action_clear_chat(self) -> None:
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]
        log = self.query_one("#chat-log", RichLog)
        log.clear()
        self.log_chat("[dim]Conversation cleared.[/dim]")

    def action_show_state(self) -> None:
        state = _state_summary()
        self.log_chat(f"[bold]Robot State:[/bold] {escape(json.dumps(state, indent=2))}")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        user_input = event.value.strip()
        if not user_input:
            return

        inp = self.query_one("#user-input", Input)
        inp.value = ""

        if self._processing:
            self.log_chat("[yellow]⏳ Still processing previous command, please wait...[/yellow]")
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
            # Only plain status strings now (thinking, tool calls, etc.)
            if isinstance(msg, str):
                _main_loop.call_soon_threadsafe(
                    self.log_chat, f"[dim]  {escape(msg)}[/dim]"
                )
            # tuple messages (stream/stream_end) are intentionally ignored here
            # — the full response is written at once after process_turn returns

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
                self.call_from_thread(self.log_chat, "[dim]Go2: (no text response)[/dim]")
        except ConnectionError as e:
            self.call_from_thread(self.log_chat, f"[bold red]ERROR:[/bold red] {escape(str(e))}")
            self.call_from_thread(self.log_chat, "[dim]Is Ollama running? Try: ollama serve[/dim]")
            if self.history and self.history[-1].get("role") == "user":
                self.history.pop()
        except TimeoutError as e:
            self.call_from_thread(self.log_chat, f"[bold red]TIMEOUT:[/bold red] {escape(str(e))}")
        except Exception as e:
            self.call_from_thread(self.log_chat, f"[bold red]ERROR:[/bold red] {escape(str(e))}")
        finally:
            self.call_from_thread(self._set_processing, False)

    def _set_processing(self, value: bool) -> None:
        self._processing = value
        status_lbl = self.query_one("#status-label", Label)
        status_lbl.update("⏳ Processing..." if value else "● Go2 Connected")

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
  python go2_cli.py
  python go2_cli.py --ip 10.0.0.207 --model llava
  python go2_cli.py --no-camera --model mistral
        """,
    )
    p.add_argument("--ip", default="10.0.0.207")
    p.add_argument("--serial", default=None)
    p.add_argument("--remote", action="store_true")
    p.add_argument("--username", default=None)
    p.add_argument("--password", default=None)
    p.add_argument("--model", default="qwen3.5:35b")
    p.add_argument("--ollama", default="http://localhost:11434")
    p.add_argument("--no-camera", action="store_true")
    p.add_argument("--quality", default=75, type=int, help="JPEG quality 1-100")
    args = p.parse_args()

    global _main_loop
    _main_loop = asyncio.get_event_loop()

    await connect(args)

    app = Go2App(args)
    await app.run_async()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown.")
