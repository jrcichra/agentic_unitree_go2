"""
go2_cli.py — Natural language CLI for the Unitree Go2

A REPL where you type commands in plain English. The program:
  1. Grabs the current camera frame + robot state
  2. Sends them to Ollama with your message
  3. Ollama picks tools to call (move, flip, dance, etc.)
  4. We execute them directly via WebRTC
  5. Results feed back to Ollama for a final response
  6. Repeat

Install:
    pip uninstall unitree_webrtc_connect -y
    pip install "git+https://github.com/12-hak/unitree_webrtc_connect.git@patch-1"
    pip install opencv-python-headless numpy requests

Run:
    python go2_cli.py
    python go2_cli.py --ip 10.0.0.207 --model llava
    python go2_cli.py --no-camera        # skip camera if you just want commands
"""

import asyncio
import argparse
import base64
import json
import sys
import time

import warnings
import os
import sys

os.environ["TERM_IMAGE_LOG_LEVEL"] = "error"

warnings.filterwarnings("ignore", category=UserWarning, module="term_image")

import requests
from PIL import Image

try:
    import cv2
    import numpy as np

    _CV2 = True
except ImportError:
    cv2 = None  # type: ignore
    np = None  # type: ignore
    _CV2 = False
    print("WARNING: opencv-python-headless not installed — camera disabled.")
    print("         pip install opencv-python-headless numpy\n")

try:
    from term_image.image import AutoImage

    _TERM_IMAGE = True
except ImportError:
    AutoImage = None  # type: ignore
    _TERM_IMAGE = False
    print("WARNING: term-image not installed — image display disabled.")
    print("         pip install term-image\n")

from unitree_webrtc_connect.webrtc_driver import (
    UnitreeWebRTCConnection,
    WebRTCConnectionMethod,
)
from unitree_webrtc_connect.constants import RTC_TOPIC, SPORT_CMD, MCF_CMD

# ---------------------------------------------------------------------------
# Monkey-patch library error handler bug
# ---------------------------------------------------------------------------
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

_conn: UnitreeWebRTCConnection | None = None
_latest_frame_jpg: bytes | None = None
_latest_frame_ts: float = 0.0
_sport_state: dict = {}
_low_state: dict = {}


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
    count = 0
    decode_errors = 0
    while True:
        try:
            frame = await track.recv()
            if frame is None:
                await asyncio.sleep(0.01)
                continue
            count += 1
            if not _CV2 or cv2 is None:
                continue
            img = frame.to_ndarray(format="bgr24")
            if img is None or img.size == 0:
                continue
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                _latest_frame_jpg = buf.tobytes()
                _latest_frame_ts = time.time()
            else:
                decode_errors += 1
        except Exception as e:
            decode_errors += 1
            if decode_errors % 100 == 1:
                print(f"[camera] decode errors: {decode_errors}/{count}", flush=True)
            await asyncio.sleep(0.1)


async def _start_camera() -> None:
    ch = getattr(_conn, "video", None)
    if ch is None:
        print("[go2] WARNING: no video channel on connection object")
        return

    async def on_track(track):
        asyncio.ensure_future(_frame_loop(track))

    ch.add_track_callback(on_track)
    await asyncio.sleep(0.5)
    ch.switchVideoChannel(True)


def _get_frame_b64(quality: int = 75) -> str | None:
    if not _CV2 or _latest_frame_jpg is None:
        return None
    if time.time() - _latest_frame_ts > 5.0:
        return None
    if cv2 is None or np is None:
        return None
    img = cv2.imdecode(np.frombuffer(_latest_frame_jpg, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else None


def _display_frame() -> bool:
    """Display the current camera frame in the terminal. Returns True if displayed."""
    if not _TERM_IMAGE or not _CV2 or _latest_frame_jpg is None:
        return False
    if time.time() - _latest_frame_ts > 5.0:
        return False
    if not sys.stdout.isatty():
        return False
    try:
        img = cv2.imdecode(np.frombuffer(_latest_frame_jpg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return False
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        image = AutoImage(pil_img, width=80)
        print(image)
        return True
    except Exception:
        return False


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
# Tool definitions  (sent to Ollama so it knows what's available)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "move",
            "description": "Move the robot by displacement. x=forward/back metres, y=left/right metres, z=yaw radians. Resolves when motion completes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {
                        "type": "number",
                        "description": "Forward metres (+) / back (-)",
                    },
                    "y": {
                        "type": "number",
                        "description": "Left metres (+) / right (-)",
                    },
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
            "description": "Turn the robot. degrees: positive=left, negative=right.",
            "parameters": {
                "type": "object",
                "properties": {
                    "degrees": {"type": "number"},
                },
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
                    },
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
                    },
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
            "description": "Tilt the robot body to look in a direction. roll/pitch/yaw in radians.",
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
            "name": "capture_image",
            "description": "Capture and return the current camera view. Use this when the user wants to see what the robot sees.",
            "parameters": {
                "type": "object",
                "properties": {
                    "quality": {
                        "type": "integer",
                        "description": "JPEG quality 1-100",
                        "default": 75,
                    },
                },
            },
        },
    },
]

TOOL_NAMES = {t["function"]["name"] for t in TOOLS}


# ---------------------------------------------------------------------------
# Execute a single tool call
# ---------------------------------------------------------------------------


async def run_tool(name: str, args: dict) -> str:
    """Execute one tool call and return a human-readable result string."""

    if name == "move":
        x = max(-0.8, min(0.8, float(args.get("x", 0))))
        y = max(-0.4, min(0.4, float(args.get("y", 0))))
        z = max(-1.2, min(1.2, float(args.get("z", 0))))
        r = await _mcf("Move", {"x": x, "y": y, "z": z})
        return f"move({x:.2f}, {y:.2f}, {z:.2f}) → {'ok' if r['ok'] else 'error code ' + str(r['code'])}"

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

    elif name == "capture_image":
        quality = int(args.get("quality", 75))
        if not _CV2 or _latest_frame_jpg is None:
            return "no camera feed available"
        if time.time() - _latest_frame_ts > 5.0:
            return "camera feed stale"
        try:
            img = cv2.imdecode(
                np.frombuffer(_latest_frame_jpg, np.uint8), cv2.IMREAD_COLOR
            )
            if img is None:
                return "failed to decode frame"
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            image = AutoImage(pil_img, width=80)
            print(image)
            return "image captured and displayed"
        except Exception as e:
            return f"error displaying image: {e}"

    else:
        return f"unknown tool: {name}"


# ---------------------------------------------------------------------------
# Ollama chat turn
# ---------------------------------------------------------------------------


def _ollama_stream(
    messages: list[dict], model: str, ollama_url: str, tools: list | None
) -> dict:
    """Single call to Ollama /api/chat with streaming. Returns content and tool_calls."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {"temperature": 0.2},
    }
    if tools:
        payload["tools"] = tools

    resp = requests.post(
        f"{ollama_url}/api/chat", json=payload, stream=True, timeout=120
    )
    resp.raise_for_status()

    content = ""
    tool_calls = []

    for line in resp.iter_lines():
        if not line:
            continue
        data = json.loads(line)
        msg = data.get("message", {})

        chunk = msg.get("content", "")
        if chunk:
            content += chunk

        if msg.get("tool_calls"):
            tool_calls = msg["tool_calls"]

    return {"content": content, "tool_calls": tool_calls}

    return {"content": content, "tool_calls": tool_calls}


# ---------------------------------------------------------------------------
# One full REPL turn: user input → tool calls → final response
# ---------------------------------------------------------------------------


async def process_turn(
    user_input: str,
    history: list[dict],
    model: str,
    ollama_url: str,
    use_camera: bool,
    cam_quality: int,
) -> str:
    """
    Run one full conversation turn.
    Returns the assistant's final text response.
    Updates history in-place.
    """

    # Build user message — include camera frame and state as context
    frame_b64 = _get_frame_b64(cam_quality) if use_camera else None
    state = _state_summary()

    # Compose the user message content
    # We attach the image and state as part of the user message so the
    # vision model actually receives the image through the images field,
    # not as a base64 string in text (which models can't decode visually)
    user_msg: dict = {
        "role": "user",
        "content": (
            f"{user_input}\n\n"
            f"[Robot state: {json.dumps(state)}]"
            + (
                "\n[Camera frame attached]"
                if frame_b64
                else "\n[No camera frame available]"
            )
        ),
    }
    if frame_b64:
        user_msg["images"] = [frame_b64]

    history.append(user_msg)

    # --- Agentic loop: call Ollama, execute tools, feed results back ---
    tool_iterations = 0
    max_iterations = 10  # safety cap

    while tool_iterations < max_iterations:
        tool_iterations += 1

        # Determine whether to send tools on this call
        # After first tool use we keep sending tools so model can chain calls
        msg = _ollama_stream(history, model, ollama_url, tools=TOOLS)
        history.append(msg)

        tool_calls = msg.get("tool_calls", [])

        # No tool calls — model is done, return its text
        if not tool_calls:
            return msg.get("content", "").strip()

        # Execute all tool calls in this response
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

            print(f"  → {name}({json.dumps(args)})", flush=True)
            result = await run_tool(name, args)
            print(f"     {result}", flush=True)
            tool_results.append(result)

        # Feed results back as a tool message
        history.append(
            {
                "role": "tool",
                "content": "\n".join(tool_results),
            }
        )

    return "Reached maximum tool call iterations."


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are controlling a Unitree Go2 robot dog via direct tool calls.
The user gives you natural language instructions. You use the available tools to carry them out.

You receive:
- A camera image showing what the robot currently sees (when available)
- The robot's current state (position, battery, etc.)
- The user's instruction

Guidelines:
- Use the camera image to inform spatial decisions (obstacles, people, distances)
- Chain multiple tool calls when needed (e.g. stand_up then move then trick)
- For tricks like flips: the robot is on firmware 1.1.7+ MCF mode — call directly, no mode switching needed
- Keep move() values within safe limits: |x|≤0.8m, |y|≤0.4m, |z|≤1.2rad
- If the robot looks fallen (rpy pitch/roll > 0.5 rad), use stance(recovery_stand) first
- After executing actions, give a brief natural response describing what you did
- If you can see the camera image, describe what you see when relevant"""


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
        # Give camera a moment to get first frame
        print("Waiting for camera...", end=" ", flush=True)
        for _ in range(30):
            if _latest_frame_jpg is not None:
                print("ready!")
                break
            await asyncio.sleep(0.2)
            print(".", end="", flush=True)
        else:
            print("\nWARNING: no camera frame yet — continuing without vision.")
    else:
        print("Camera disabled (--no-camera).")


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------


async def repl(args: argparse.Namespace) -> None:
    history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    model = args.model
    ollama_url = args.ollama
    use_camera = not args.no_camera

    print(f"\nGo2 CLI ready. Model: {model}  Ollama: {ollama_url}")
    print("Type your instructions. 'quit' or Ctrl-C to exit.\n")

    while True:
        try:
            user_input = input("you: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            break

        # Built-in commands that bypass the LLM
        if user_input.lower() == "state":
            print(json.dumps(_state_summary(), indent=2))
            continue

        if user_input.lower() in ("camera", "vision", "see"):
            if _display_frame():
                print()
            else:
                print("[no camera feed available]")
            continue

        if user_input.lower() == "clear":
            history = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("[conversation cleared]")
            continue

        if user_input.lower().startswith("model "):
            model = user_input.split(None, 1)[1].strip()
            print(f"[model switched to: {model}]")
            continue

        print("go2: ", end="", flush=True)
        try:
            response = await process_turn(
                user_input=user_input,
                history=history,
                model=model,
                ollama_url=ollama_url,
                use_camera=use_camera,
                cam_quality=args.quality,
            )
            # response already streamed via _ollama_stream, only print if empty
            if response:
                print(response)
        except requests.exceptions.ConnectionError:
            print(f"ERROR: Can't reach Ollama at {ollama_url}")
            print("       Is Ollama running? Try: ollama serve")
        except Exception as e:
            print(f"ERROR: {e}")

        print()

    # Clean shutdown
    print("Stopping robot...", end=" ", flush=True)
    try:
        await _mcf("StopMove")
        await _mcf("BalanceStand")
        print("done.")
    except Exception:
        print("(failed)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    p = argparse.ArgumentParser(
        description="Go2 natural language CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python go2_cli.py
  python go2_cli.py --model qwen3.5:35b
  python go2_cli.py --ip 10.0.0.207 --model minicpm-v
  python go2_cli.py --no-camera --model mistral
        """,
    )
    p.add_argument("--ip", default="10.0.0.207")
    p.add_argument("--serial", default=None)
    p.add_argument("--remote", action="store_true")
    p.add_argument("--username", default=None)
    p.add_argument("--password", default=None)
    p.add_argument(
        "--model", default="qwen3.5:35b", help="Ollama model (default: qwen3.5:35b)"
    )
    p.add_argument("--ollama", default="http://localhost:11434", help="Ollama URL")
    p.add_argument("--no-camera", action="store_true", help="Disable camera / vision")
    p.add_argument("--quality", default=75, type=int, help="JPEG quality 1-100")
    args = p.parse_args()

    await connect(args)
    await repl(args)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown.")
