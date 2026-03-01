"""
Unitree Go2 WebRTC MCP Server — Streamable HTTP + stdio transport

Install deps:
    pip uninstall unitree_webrtc_connect -y
    pip install "git+https://github.com/12-hak/unitree_webrtc_connect.git@patch-1"
    pip install "mcp[cli]" uvicorn starlette opencv-python-headless numpy

Run (HTTP — OpenWebUI):
    python go2_mcp_server.py
    python go2_mcp_server.py --ip 10.0.0.207 --port 8000

Run (stdio — Claude Desktop):
    python go2_mcp_server.py --stdio

OpenWebUI:
    Admin Panel -> Settings -> Tools -> Add Connection
    Type : MCP (Streamable HTTP)
    URL  : http://<this-machine-ip>:8000/mcp
"""

import asyncio
import argparse
import json
import logging
import sys
from typing import Any

import base64
import io
import time

import uvicorn
from starlette.applications import Starlette
from starlette.routing import Mount

try:
    import cv2
    import numpy as np
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False
    print("[go2-mcp] WARNING: opencv-python not installed — capture_image will be unavailable.", file=sys.stderr)
    print("[go2-mcp]   Install with: pip install opencv-python-headless numpy", file=sys.stderr)

from mcp.server import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.stdio import stdio_server
from mcp import types

from unitree_webrtc_connect.webrtc_driver import UnitreeWebRTCConnection, WebRTCConnectionMethod
from unitree_webrtc_connect.constants import RTC_TOPIC, SPORT_CMD, MCF_CMD

logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# Monkey-patch: fix broken error_handler in unitree_webrtc_connect
# The library's handle_error() tries to unpack an error response as a tuple
# (timestamp, error_source, error_code) but sometimes the robot sends a plain
# int. This causes a TypeError that kills the data channel message loop.
# ---------------------------------------------------------------------------
try:
    from unitree_webrtc_connect.msgs import error_handler as _eh

    def _safe_handle_error(error):
        try:
            if isinstance(error, (list, tuple)) and len(error) == 3:
                timestamp, error_source, error_code_int = error
                _eh._original_handle_error_logic(timestamp, error_source, error_code_int)
            else:
                logging.warning(f"[go2-mcp] Unexpected error format from robot (ignored): {error!r}")
        except Exception as e:
            logging.warning(f"[go2-mcp] Error handler exception (ignored): {e} — raw error: {error!r}")

    # Preserve any original logic if needed, then replace
    _eh.handle_error = _safe_handle_error
    print("[go2-mcp] Applied error_handler patch.", file=sys.stderr)
except Exception as _patch_err:
    print(f"[go2-mcp] Could not patch error_handler (non-fatal): {_patch_err}", file=sys.stderr)

# ---------------------------------------------------------------------------
# MCF topic — all MCF_CMD calls go here (firmware 1.1.7+)
# ---------------------------------------------------------------------------
MCF_TOPIC = "rt/api/sport/request"   # same wire topic as SPORT_MOD; MCF just uses different api_ids

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
_conn: UnitreeWebRTCConnection | None = None
_latest_sport_state: dict = {}
_latest_low_state:   dict = {}
_latest_multi_state: dict = {}

# Camera — latest decoded frame as JPEG bytes, updated continuously in background
_latest_frame_jpg: bytes | None = None
_latest_frame_ts: float = 0.0      # unix time of last frame
_video_track = None                 # aiortc VideoStreamTrack from the robot


async def get_conn() -> UnitreeWebRTCConnection:
    if _conn is None:
        raise RuntimeError("Robot not connected.")
    return _conn


def _code(resp: dict) -> int:
    return resp.get("data", {}).get("header", {}).get("status", {}).get("code", -1)


async def _mcf(conn, cmd_name: str, parameter: dict | None = None) -> dict:
    """Fire an MCF_CMD and return a normalised result dict."""
    payload: dict = {"api_id": MCF_CMD[cmd_name]}
    if parameter:
        payload["parameter"] = parameter
    resp = await conn.datachannel.pub_sub.publish_request_new(MCF_TOPIC, payload)
    code = _code(resp)
    return {"status": "ok" if code == 0 else "error", "command": cmd_name, "response_code": code}


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------
server = Server("unitree-go2")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [

        # ── Movement ──────────────────────────────────────────────────────────
        types.Tool(
            name="move",
            description=(
                "Move the Go2 by a specific displacement. Resolves when the motion completes. "
                "x = forward/backward metres (+forward), "
                "y = left/right metres (+left), "
                "z = yaw radians (+left turn). "
                "Examples: 1 m forward → x=1.0 | 90° left → z=1.5708 | 0.5 m right → y=-0.5"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "x": {"type": "number", "description": "Forward/back metres"},
                    "y": {"type": "number", "description": "Left/right metres"},
                    "z": {"type": "number", "description": "Yaw radians"},
                },
                "required": [],
            },
        ),

        types.Tool(
            name="stop",
            description="Stop all motion immediately.",
            inputSchema={"type": "object", "properties": {}},
        ),

        # ── Stance ────────────────────────────────────────────────────────────
        types.Tool(
            name="stand_up",
            description="Stand up from lying or sitting.",
            inputSchema={"type": "object", "properties": {}},
        ),

        types.Tool(
            name="stand_down",
            description="Lie down / sit down.",
            inputSchema={"type": "object", "properties": {}},
        ),

        types.Tool(
            name="balance_stand",
            description="Enter balanced stand mode — stands still, ready.",
            inputSchema={"type": "object", "properties": {}},
        ),

        types.Tool(
            name="recovery_stand",
            description="Right the robot if it has fallen over.",
            inputSchema={"type": "object", "properties": {}},
        ),

        types.Tool(
            name="sit",
            description="Make the robot sit.",
            inputSchema={"type": "object", "properties": {}},
        ),

        types.Tool(
            name="rise_sit",
            description="Rise from sitting position.",
            inputSchema={"type": "object", "properties": {}},
        ),

        types.Tool(
            name="damp",
            description="Damp all joints — robot goes limp/safe. Use before picking up or powering down.",
            inputSchema={"type": "object", "properties": {}},
        ),

        types.Tool(
            name="back_stand",
            description="Rear-leg stand — robot stands up on its hind legs.",
            inputSchema={"type": "object", "properties": {}},
        ),

        # ── Gaits ─────────────────────────────────────────────────────────────
        types.Tool(
            name="set_gait",
            description=(
                "Switch the robot's walking gait. "
                "economic = energy-saving slow walk. "
                "static = very stable slow walk. "
                "trot_run = fast trot/run gait. "
                "free_walk = natural free walk. "
                "free_bound = bounding gait. "
                "free_jump = jumping locomotion. "
                "free_avoid = avoidance-aware free walk. "
                "classic = classic stable walk. "
                "cross_step = lateral cross-step gait. "
                "continuous = continuous gait mode."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "gait": {
                        "type": "string",
                        "enum": [
                            "economic", "static", "trot_run",
                            "free_walk", "free_bound", "free_jump", "free_avoid",
                            "classic", "cross_step", "continuous",
                        ],
                    }
                },
                "required": ["gait"],
            },
        ),

        types.Tool(
            name="lead_follow",
            description="Activate Lead-Follow mode — robot follows the nearest person.",
            inputSchema={"type": "object", "properties": {}},
        ),

        # ── Body / pose ───────────────────────────────────────────────────────
        types.Tool(
            name="set_body_height",
            description="Adjust body height offset in metres. Negative = lower, positive = raise.",
            inputSchema={
                "type": "object",
                "properties": {
                    "height": {"type": "number", "description": "Height offset in metres"}
                },
                "required": ["height"],
            },
        ),

        types.Tool(
            name="set_foot_raise_height",
            description="Set how high the robot lifts its feet while walking, in metres (e.g. 0.08).",
            inputSchema={
                "type": "object",
                "properties": {
                    "height": {"type": "number", "description": "Foot raise height in metres"}
                },
                "required": ["height"],
            },
        ),

        types.Tool(
            name="set_speed_level",
            description="Set walking speed level: 0 = slow, 1 = normal, 2 = fast.",
            inputSchema={
                "type": "object",
                "properties": {
                    "level": {"type": "integer", "enum": [0, 1, 2]}
                },
                "required": ["level"],
            },
        ),

        types.Tool(
            name="set_euler",
            description=(
                "Tilt the robot's body while standing still. "
                "roll = lean left/right (radians), "
                "pitch = lean forward/back (radians), "
                "yaw = rotate body (radians). "
                "Useful for looking around or dramatic poses."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "roll":  {"type": "number", "description": "Roll radians"},
                    "pitch": {"type": "number", "description": "Pitch radians"},
                    "yaw":   {"type": "number", "description": "Yaw radians"},
                },
                "required": [],
            },
        ),

        # ── Friendly / social actions ─────────────────────────────────────────
        types.Tool(
            name="hello",
            description="Wave hello — robot raises a leg and waves at you.",
            inputSchema={"type": "object", "properties": {}},
        ),

        types.Tool(
            name="show_heart",
            description="Robot makes a heart shape / finger-heart gesture. Uses MCF Heart (api_id 1036). Same underlying command as finger_heart.",
            inputSchema={"type": "object", "properties": {}},
        ),

        types.Tool(
            name="stretch",
            description="Robot does a full stretch.",
            inputSchema={"type": "object", "properties": {}},
        ),

        # content (1020) removed — RoboVerse docs confirm "API not implemented on the server"

        types.Tool(
            name="scrape",
            description="Robot does a scraping/pawing motion with its front leg.",
            inputSchema={"type": "object", "properties": {}},
        ),

        types.Tool(
            name="wallow",
            description="Robot rolls around on its back (wallowing/rolling).",
            inputSchema={"type": "object", "properties": {}},
        ),

        # ── Dance ─────────────────────────────────────────────────────────────
        types.Tool(
            name="dance",
            description="Make the robot dance. routine = 1 or 2 for two different dance sequences.",
            inputSchema={
                "type": "object",
                "properties": {
                    "routine": {"type": "integer", "enum": [1, 2], "description": "Dance routine 1 or 2"}
                },
                "required": [],
            },
        ),

        # ── Acrobatics ────────────────────────────────────────────────────────
        types.Tool(
            name="flip",
            description=(
                "Perform a flip. direction = front, back, left, or right. "
                "IMPORTANT: On firmware 1.1.7+ (MCF mode) do NOT call set_motion_mode first — "
                "just call this tool directly. The robot is already in MCF mode which handles all acrobatics. "
                "Switching to ai mode will return error 7004 and is not required. "
                "Uses MCF api_ids: BackFlip=2043, LeftFlip=2041, RightFlip=2040, FrontFlip=1030. "
                "WARNING: Requires clear area ~2m in the flip direction, flat non-slip surface."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["front", "back", "left", "right"],
                        "description": "Flip direction",
                    }
                },
                "required": ["direction"],
            },
        ),

        types.Tool(
            name="handstand",
            description=(
                "Robot performs a handstand (balances on front legs). "
                "IMPORTANT: Call directly — do NOT switch to ai mode first on firmware 1.1.7+ (MCF). "
                "WARNING: Requires clear space behind the robot."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),

        types.Tool(
            name="front_jump",
            description="Robot performs a forward jump. Call directly — no mode switch needed on firmware 1.1.7+ MCF. Requires flat surface with landing space.",
            inputSchema={"type": "object", "properties": {}},
        ),

        types.Tool(
            name="front_pounce",
            description="Robot performs a pouncing leap forward. Call directly — no mode switch needed on firmware 1.1.7+ MCF.",
            inputSchema={"type": "object", "properties": {}},
        ),

        types.Tool(
            name="wiggle_hips",
            description="Robot wiggles its hips. Good for celebrations.",
            inputSchema={"type": "object", "properties": {}},
        ),

        types.Tool(
            name="finger_heart",
            description="Robot makes a finger-heart gesture with its leg. Uses api_id 1036 (same as show_heart / MCF Heart).",
            inputSchema={"type": "object", "properties": {}},
        ),

        types.Tool(
            name="moon_walk",
            description=(
                "Robot does a moonwalk sideways shuffle. "
                "Uses SPORT_CMD api_id 1046 — present in library but NOT confirmed in MCF_CMD or RoboVerse docs. "
                "May return an error on some firmware versions."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),

        types.Tool(
            name="one_sided_step",
            description="Robot does a one-sided stepping motion (api_id 1303).",
            inputSchema={"type": "object", "properties": {}},
        ),

        types.Tool(
            name="bound",
            description="Robot performs a bounding gallop motion (api_id 1304).",
            inputSchema={"type": "object", "properties": {}},
        ),

        # ── Motion mode ───────────────────────────────────────────────────────
        types.Tool(
            name="set_motion_mode",
            description=(
                "Switch motion mode. "
                "IMPORTANT: On firmware 1.1.7+ (MCF) this call returns error 7004 — that is NORMAL and expected. "
                "It means the robot is already in MCF unified mode. "
                "Do NOT attempt to switch to ai mode before flips, handstands or acrobatics — "
                "just call the flip/handstand tools directly. "
                "Only use this for older firmware or obstacle_avoidance mode toggle. "
                "'normal' = standard walking (old firmware). "
                "'ai' = advanced acrobatics (old firmware only, 7004 on MCF). "
                "'obstacle_avoidance' = obstacle avoidance walking."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["normal", "ai", "obstacle_avoidance"],
                    }
                },
                "required": ["mode"],
            },
        ),

        types.Tool(
            name="get_motion_mode",
            description="Get the current motion mode name.",
            inputSchema={"type": "object", "properties": {}},
        ),

        # ── Auto recovery / avoid mode ────────────────────────────────────────
        types.Tool(
            name="set_auto_recovery",
            description="Enable or disable automatic self-righting when the robot falls.",
            inputSchema={
                "type": "object",
                "properties": {
                    "enabled": {"type": "boolean"}
                },
                "required": ["enabled"],
            },
        ),

        types.Tool(
            name="get_auto_recovery",
            description="Get whether automatic self-righting recovery is enabled.",
            inputSchema={"type": "object", "properties": {}},
        ),

        types.Tool(
            name="set_avoid_mode",
            description="Switch the obstacle avoidance mode on or off via MCF SwitchAvoidMode.",
            inputSchema={
                "type": "object",
                "properties": {
                    "enabled": {"type": "boolean"}
                },
                "required": ["enabled"],
            },
        ),

        types.Tool(
            name="set_obstacle_avoidance",
            description="Enable or disable obstacle avoidance via the OBSTACLES_AVOID API.",
            inputSchema={
                "type": "object",
                "properties": {
                    "enabled": {"type": "boolean"}
                },
                "required": ["enabled"],
            },
        ),

        # ── Telemetry ─────────────────────────────────────────────────────────
        types.Tool(
            name="get_sport_state",
            description=(
                "Get current sport state: position, velocity, orientation (RPY), "
                "body height, gait type, foot forces."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),

        types.Tool(
            name="get_low_state",
            description=(
                "Get low-level state: motor joint angles + temperatures, "
                "IMU RPY, battery SOC and voltage, foot forces."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),

        types.Tool(
            name="get_multiple_state",
            description=(
                "Get miscellaneous settings state: body height, LED brightness, "
                "foot raise height, obstacle avoidance switch, speed level, UWB, volume."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),

        types.Tool(
            name="get_robot_state",
            description="Query the robot's full internal state snapshot via MCF GetState.",
            inputSchema={"type": "object", "properties": {}},
        ),

        # ── VUI — lights & sound ──────────────────────────────────────────────
        types.Tool(
            name="set_led_color",
            description=(
                "Set body LED color. Colors: white, red, yellow, blue, green, cyan, purple. "
                "duration = seconds. flash_cycle = optional flash period ms (min 499)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "color":       {"type": "string",  "enum": ["white","red","yellow","blue","green","cyan","purple"]},
                    "duration":    {"type": "number",  "description": "Seconds"},
                    "flash_cycle": {"type": "integer", "description": "Flash period ms (optional, min 499)"},
                },
                "required": ["color", "duration"],
            },
        ),

        types.Tool(
            name="set_brightness",
            description="Set front LED flashlight brightness (0=off, 10=max).",
            inputSchema={
                "type": "object",
                "properties": {
                    "brightness": {"type": "integer", "minimum": 0, "maximum": 10}
                },
                "required": ["brightness"],
            },
        ),

        types.Tool(
            name="set_volume",
            description="Set speaker volume (0=silent, 10=max).",
            inputSchema={
                "type": "object",
                "properties": {
                    "volume": {"type": "integer", "minimum": 0, "maximum": 10}
                },
                "required": ["volume"],
            },
        ),

        types.Tool(
            name="get_volume",
            description="Get current speaker volume (0-10).",
            inputSchema={"type": "object", "properties": {}},
        ),

        # ── LiDAR ─────────────────────────────────────────────────────────────
        types.Tool(
            name="lidar_snapshot",
            description=(
                "One-shot LiDAR scan. Returns point count and 3-D bounding box in metres. "
                "Turns LiDAR on, grabs one frame, turns it off."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),

        # ── Camera ────────────────────────────────────────────────────────────
        types.Tool(
            name="capture_image",
            description=(
                "Capture the current view from the Go2's front camera and return it as a base64-encoded JPEG. "
                "Use this to see what the robot sees — obstacles, people, rooms, objects, navigation. "
                "Returns the image plus metadata: timestamp, resolution, and frame age. "
                "The camera streams continuously in the background; this just grabs the latest frame. "
                "Requires: pip install opencv-python-headless numpy"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "quality": {
                        "type": "integer",
                        "description": "JPEG quality 1-100 (default 75). Lower = smaller payload, higher = more detail.",
                        "minimum": 1,
                        "maximum": 100,
                    }
                },
                "required": [],
            },
        ),
    ]


# ---------------------------------------------------------------------------
# Gait name -> MCF_CMD key
# ---------------------------------------------------------------------------
_GAIT_MAP = {
    "economic":   "EconomicGait",
    "static":     "StaticWalk",
    "trot_run":   "TrotRun",
    "free_walk":  "FreeWalk",
    "free_bound": "FreeBound",
    "free_jump":  "FreeJump",
    "free_avoid": "FreeAvoid",
    "classic":    "ClassicWalk",
    "cross_step": "CrossStep",
    "continuous": "ContinuousGait",
}

_FLIP_MAP = {
    "front": "FrontFlip",
    "back":  "BackFlip",
    "left":  "LeftFlip",
    "right": "RightFlip",
}


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------

@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    conn = await get_conn()
    try:
        result = await _dispatch(conn, name, arguments)
    except Exception as e:
        result = {"error": str(e)}
    return [types.TextContent(type="text", text=json.dumps(result, indent=2))]


async def _dispatch(conn: UnitreeWebRTCConnection, name: str, args: dict) -> dict:

    # ── Movement ──────────────────────────────────────────────────────────────
    if name == "move":
        x = float(args.get("x", 0))
        y = float(args.get("y", 0))
        z = float(args.get("z", 0))
        return await _mcf(conn, "Move", {"x": x, "y": y, "z": z})

    elif name == "stop":
        return await _mcf(conn, "StopMove")

    # ── Stance ────────────────────────────────────────────────────────────────
    elif name == "stand_up":
        return await _mcf(conn, "StandUp")

    elif name == "stand_down":
        return await _mcf(conn, "StandDown")

    elif name == "balance_stand":
        return await _mcf(conn, "BalanceStand")

    elif name == "recovery_stand":
        return await _mcf(conn, "RecoveryStand")

    elif name == "sit":
        return await _mcf(conn, "Sit")

    elif name == "rise_sit":
        return await _mcf(conn, "RiseSit")

    elif name == "damp":
        return await _mcf(conn, "Damp")

    elif name == "back_stand":
        return await _mcf(conn, "BackStand")

    # ── Gaits ─────────────────────────────────────────────────────────────────
    elif name == "set_gait":
        gait = args["gait"]
        cmd  = _GAIT_MAP.get(gait)
        if not cmd:
            return {"error": f"Unknown gait '{gait}'"}
        return await _mcf(conn, cmd)

    elif name == "lead_follow":
        return await _mcf(conn, "LeadFollow")

    # ── Body / pose ───────────────────────────────────────────────────────────
    elif name == "set_body_height":
        return await _mcf(conn, "BodyHeight", {"data": float(args["height"])})

    elif name == "set_foot_raise_height":
        return await _mcf(conn, "FootRaiseHeight", {"data": float(args["height"])})

    elif name == "set_speed_level":
        return await _mcf(conn, "SpeedLevel", {"data": int(args["level"])})

    elif name == "set_euler":
        roll  = float(args.get("roll",  0))
        pitch = float(args.get("pitch", 0))
        yaw   = float(args.get("yaw",   0))
        return await _mcf(conn, "Euler", {"x": roll, "y": pitch, "z": yaw})

    # ── Friendly / social ─────────────────────────────────────────────────────
    elif name == "hello":
        return await _mcf(conn, "Hello")

    elif name == "show_heart":
        return await _mcf(conn, "Heart")

    elif name == "stretch":
        return await _mcf(conn, "Stretch")

    elif name == "scrape":
        return await _mcf(conn, "Scrape")

    elif name == "wallow":
        # Wallow=1021: same ID in both SPORT_CMD and MCF, send on MCF_TOPIC
        resp = await conn.datachannel.pub_sub.publish_request_new(
            MCF_TOPIC, {"api_id": 1021}
        )
        code = _code(resp)
        return {"status": "ok" if code == 0 else "error", "command": "Wallow", "response_code": code}

    # ── Dance ─────────────────────────────────────────────────────────────────
    elif name == "dance":
        routine = int(args.get("routine", 1))
        cmd = "Dance1" if routine == 1 else "Dance2"
        return await _mcf(conn, cmd)

    # ── Acrobatics ────────────────────────────────────────────────────────────
    elif name == "flip":
        direction = args["direction"]
        cmd = _FLIP_MAP.get(direction)
        if not cmd:
            return {"error": f"Unknown flip direction '{direction}'"}
        return await _mcf(conn, cmd)

    elif name == "handstand":
        return await _mcf(conn, "Handstand")

    elif name == "front_jump":
        return await _mcf(conn, "FrontJump")

    elif name == "front_pounce":
        return await _mcf(conn, "FrontPounce")

    elif name == "wiggle_hips":
        # WiggleHips=1033: same ID everywhere, send on MCF_TOPIC
        resp = await conn.datachannel.pub_sub.publish_request_new(
            MCF_TOPIC, {"api_id": 1033}
        )
        code = _code(resp)
        return {"status": "ok" if code == 0 else "error", "command": "WiggleHips", "response_code": code}

    elif name == "finger_heart":
        # FingerHeart=1036 same as MCF Heart=1036 — they are the same command
        return await _mcf(conn, "Heart")

    elif name == "moon_walk":
        # MoonWalk=1046 in SPORT_CMD but not confirmed in MCF — try MCF_TOPIC anyway
        resp = await conn.datachannel.pub_sub.publish_request_new(
            MCF_TOPIC, {"api_id": 1046}
        )
        code = _code(resp)
        return {"status": "ok" if code == 0 else "error", "command": "MoonWalk", "response_code": code}

    elif name == "one_sided_step":
        resp = await conn.datachannel.pub_sub.publish_request_new(
            MCF_TOPIC, {"api_id": 1303}
        )
        code = _code(resp)
        return {"status": "ok" if code == 0 else "error", "command": "OnesidedStep", "response_code": code}

    elif name == "bound":
        resp = await conn.datachannel.pub_sub.publish_request_new(
            MCF_TOPIC, {"api_id": 1304}
        )
        code = _code(resp)
        return {"status": "ok" if code == 0 else "error", "command": "Bound", "response_code": code}

    # ── Motion mode ───────────────────────────────────────────────────────────
    elif name == "set_motion_mode":
        mode = args["mode"]
        resp = await conn.datachannel.pub_sub.publish_request_new(
            RTC_TOPIC["MOTION_SWITCHER"],
            {"api_id": 1002, "parameter": {"name": mode}},
        )
        code = _code(resp)
        if code == 7004:
            return {"status": "ok", "mode": "mcf", "response_code": 7004,
                    "note": "Firmware 1.1.7+ MCF mode active — mode switch not needed, all commands work directly"}
        return {"status": "ok" if code == 0 else "error", "mode": mode, "response_code": code}

    elif name == "get_motion_mode":
        resp = await conn.datachannel.pub_sub.publish_request_new(
            RTC_TOPIC["MOTION_SWITCHER"], {"api_id": 1001}
        )
        code = _code(resp)
        if code == 0:
            data = json.loads(resp["data"]["data"])
            return {"status": "ok", "mode": data.get("name")}
        if code == 7004:
            return {"status": "ok", "mode": "mcf", "note": "Firmware 1.1.7+ MCF unified mode active — all acrobatics work directly"}
        return {"status": "error", "response_code": code}

    # ── Auto recovery / avoid ─────────────────────────────────────────────────
    elif name == "set_auto_recovery":
        enabled = bool(args["enabled"])
        return await _mcf(conn, "SetAutoRecovery", {"data": 1 if enabled else 0})

    elif name == "get_auto_recovery":
        resp = await conn.datachannel.pub_sub.publish_request_new(
            MCF_TOPIC, {"api_id": MCF_CMD["GetAutoRecovery"]}
        )
        code = _code(resp)
        if code == 0:
            data = resp.get("data", {}).get("data", {})
            return {"status": "ok", "auto_recovery": data}
        return {"status": "error", "response_code": code}

    elif name == "set_avoid_mode":
        enabled = bool(args["enabled"])
        return await _mcf(conn, "SwitchAvoidMode", {"data": 1 if enabled else 0})

    elif name == "set_obstacle_avoidance":
        enabled = bool(args["enabled"])
        resp = await conn.datachannel.pub_sub.publish_request_new(
            RTC_TOPIC["OBSTACLES_AVOID"],
            {"api_id": 1001, "parameter": {"switch": 1 if enabled else 0}},
        )
        code = _code(resp)
        return {"status": "ok" if code == 0 else "error", "enabled": enabled, "response_code": code}

    # ── Telemetry ─────────────────────────────────────────────────────────────
    elif name == "get_sport_state":
        if not _latest_sport_state:
            return {"status": "no_data", "message": "No data yet — try again in a moment."}
        return {"status": "ok", "state": _latest_sport_state}

    elif name == "get_low_state":
        if not _latest_low_state:
            return {"status": "no_data", "message": "No data yet — try again in a moment."}
        return {"status": "ok", "state": _latest_low_state}

    elif name == "get_multiple_state":
        if not _latest_multi_state:
            return {"status": "no_data", "message": "No data yet — try again in a moment."}
        return {"status": "ok", "state": _latest_multi_state}

    elif name == "get_robot_state":
        resp = await conn.datachannel.pub_sub.publish_request_new(
            MCF_TOPIC, {"api_id": MCF_CMD["GetState"]}
        )
        code = _code(resp)
        if code == 0:
            return {"status": "ok", "state": resp.get("data", {}).get("data")}
        return {"status": "error", "response_code": code}

    # ── VUI ───────────────────────────────────────────────────────────────────
    elif name == "set_led_color":
        color    = args["color"]
        duration = int(float(args["duration"]))
        param: dict = {"color": color, "time": duration}
        if "flash_cycle" in args:
            param["flash_cycle"] = int(args["flash_cycle"])
        resp = await conn.datachannel.pub_sub.publish_request_new(
            RTC_TOPIC["VUI"], {"api_id": 1007, "parameter": param}
        )
        code = _code(resp)
        return {"status": "ok" if code == 0 else "error", "color": color, "response_code": code}

    elif name == "set_brightness":
        brightness = int(args["brightness"])
        resp = await conn.datachannel.pub_sub.publish_request_new(
            RTC_TOPIC["VUI"], {"api_id": 1005, "parameter": {"brightness": brightness}}
        )
        code = _code(resp)
        return {"status": "ok" if code == 0 else "error", "brightness": brightness, "response_code": code}

    elif name == "set_volume":
        volume = int(args["volume"])
        resp = await conn.datachannel.pub_sub.publish_request_new(
            RTC_TOPIC["VUI"], {"api_id": 1003, "parameter": {"volume": volume}}
        )
        code = _code(resp)
        return {"status": "ok" if code == 0 else "error", "volume": volume, "response_code": code}

    elif name == "get_volume":
        resp = await conn.datachannel.pub_sub.publish_request_new(
            RTC_TOPIC["VUI"], {"api_id": 1004}
        )
        code = _code(resp)
        if code == 0:
            data = json.loads(resp["data"]["data"])
            return {"status": "ok", "volume": data.get("volume")}
        return {"status": "error", "response_code": code}

    # ── LiDAR ─────────────────────────────────────────────────────────────────
    elif name == "lidar_snapshot":
        snap: dict = {}
        event = asyncio.Event()

        def lidar_cb(message):
            try:
                import numpy as np
                data = message.get("data", {}).get("data", {})

                # positions may be a numpy array, nested list, or flat list
                positions = data.get("positions", None)
                if positions is None:
                    snap["point_count"] = 0
                else:
                    # Always normalise to a plain flat Python list of numbers
                    if isinstance(positions, np.ndarray):
                        positions = positions.flatten().tolist()
                    elif hasattr(positions, 'tolist'):
                        positions = positions.tolist()
                    else:
                        positions = list(positions)

                    if len(positions) >= 3:
                        pts = [[positions[i], positions[i+1], positions[i+2]]
                               for i in range(0, len(positions) - 2, 3)]
                        if pts:
                            xs = [p[0] for p in pts]
                            ys = [p[1] for p in pts]
                            zs = [p[2] for p in pts]
                            snap["point_count"]  = len(pts)
                            snap["bounding_box"] = {
                                "x_m": [round(min(xs), 3), round(max(xs), 3)],
                                "y_m": [round(min(ys), 3), round(max(ys), 3)],
                                "z_m": [round(min(zs), 3), round(max(zs), 3)],
                            }

                snap["origin"]     = message.get("data", {}).get("origin")
                snap["resolution"] = message.get("data", {}).get("resolution")
            except Exception as e:
                snap["lidar_parse_error"] = str(e)
            finally:
                event.set()

        await conn.datachannel.disableTrafficSaving(True)
        conn.datachannel.pub_sub.publish_without_callback("rt/utlidar/switch", "on")
        conn.datachannel.pub_sub.subscribe(RTC_TOPIC["ULIDAR_ARRAY"], lidar_cb)

        try:
            await asyncio.wait_for(event.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            conn.datachannel.pub_sub.unsubscribe(RTC_TOPIC["ULIDAR_ARRAY"])
            conn.datachannel.pub_sub.publish_without_callback("rt/utlidar/switch", "off")
            return {"status": "error", "message": "LiDAR timeout — no data in 5 s"}

        conn.datachannel.pub_sub.unsubscribe(RTC_TOPIC["ULIDAR_ARRAY"])
        conn.datachannel.pub_sub.publish_without_callback("rt/utlidar/switch", "off")
        snap["status"] = "ok"
        return snap

    elif name == "capture_image":
        if not _CV2_AVAILABLE:
            return {"status": "error", "message": "opencv-python not installed. Run: pip install opencv-python-headless numpy"}
        if _latest_frame_jpg is None:
            return {"status": "no_data", "message": "No camera frame received yet — video track may still be initialising. Try again in a moment."}
        quality  = int(args.get("quality", 75))
        age_s    = round(time.time() - _latest_frame_ts, 2)
        if age_s > 5.0:
            return {"status": "stale", "message": f"Latest frame is {age_s}s old — video track may have dropped.", "age_seconds": age_s}
        # Re-encode at requested quality
        img_np = cv2.imdecode(np.frombuffer(_latest_frame_jpg, np.uint8), cv2.IMREAD_COLOR)
        ok, buf = cv2.imencode(".jpg", img_np, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return {"status": "error", "message": "Failed to encode frame"}
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        h, w = img_np.shape[:2]
        return {
            "status": "ok",
            "image_base64": b64,
            "media_type": "image/jpeg",
            "width": w,
            "height": h,
            "age_seconds": age_s,
            "quality": quality,
        }

    else:
        return {"error": f"Unknown tool '{name}'"}


# ---------------------------------------------------------------------------
# Background state subscriptions
# ---------------------------------------------------------------------------

def _setup_state_subscriptions(conn: UnitreeWebRTCConnection) -> None:
    def on_sport(msg):
        global _latest_sport_state
        _latest_sport_state = msg.get("data", {})

    def on_low(msg):
        global _latest_low_state
        _latest_low_state = msg.get("data", {})

    def on_multi(msg):
        global _latest_multi_state
        raw = msg.get("data", "{}")
        try:
            _latest_multi_state = json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            _latest_multi_state = {}

    conn.datachannel.pub_sub.subscribe(RTC_TOPIC["LF_SPORT_MOD_STATE"], on_sport)
    conn.datachannel.pub_sub.subscribe(RTC_TOPIC["LOW_STATE"],           on_low)
    conn.datachannel.pub_sub.subscribe(RTC_TOPIC["MULTIPLE_STATE"],      on_multi)


# ---------------------------------------------------------------------------
# Background video frame loop
# ---------------------------------------------------------------------------

async def _video_frame_loop(track) -> None:
    """Continuously pull frames from the robot camera and keep the latest as JPEG bytes."""
    global _latest_frame_jpg, _latest_frame_ts
    print("[go2-mcp] Video frame loop started.", file=sys.stderr)
    frame_count = 0
    while True:
        try:
            frame = await track.recv()
            if frame is None:
                await asyncio.sleep(0.01)
                continue
            frame_count += 1
            if not _CV2_AVAILABLE:
                break
            # av.VideoFrame -> numpy BGR array -> JPEG bytes
            img = frame.to_ndarray(format="bgr24")
            if img is None or img.size == 0:
                await asyncio.sleep(0.01)
                continue
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                _latest_frame_jpg = buf.tobytes()
                _latest_frame_ts  = time.time()
                if frame_count % 30 == 0:
                    print(f"[go2-mcp] Camera: {frame_count} frames captured, shape={img.shape}", file=sys.stderr)
        except Exception as e:
            print(f"[go2-mcp] Video loop error: {e}", file=sys.stderr)
            await asyncio.sleep(0.1)


async def _setup_video_track(conn: UnitreeWebRTCConnection) -> None:
    """
    Start receiving camera frames using WebRTCVideoChannel's callback API.
    add_track_callback registers a coroutine that fires when the track is live,
    then switchVideoChannel(True) tells the robot to start streaming.
    """
    global _video_track

    video_channel = getattr(conn, "video", None)
    if video_channel is None:
        print("[go2-mcp] WARNING: conn.video is None — camera unavailable.", file=sys.stderr)
        return

    async def on_track(track):
        global _video_track
        _video_track = track
        print(f"[go2-mcp] Video track callback fired: {type(track).__name__}", file=sys.stderr)
        asyncio.ensure_future(_video_frame_loop(track))

    video_channel.add_track_callback(on_track)
    await asyncio.sleep(0.5)
    video_channel.switchVideoChannel(True)
    print("[go2-mcp] Video channel enabled — waiting for track callback.", file=sys.stderr)


# ---------------------------------------------------------------------------
# Robot connection
# ---------------------------------------------------------------------------

async def _connect_robot(args: argparse.Namespace) -> UnitreeWebRTCConnection:
    print("[go2-mcp] Connecting to Go2 via WebRTC ...", file=sys.stderr)
    if args.remote:
        conn = UnitreeWebRTCConnection(
            WebRTCConnectionMethod.Remote,
            serialNumber=args.serial,
            username=args.username,
            password=args.password,
        )
    elif args.serial and not args.ip:
        conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, serialNumber=args.serial)
    elif args.ip:
        conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=args.ip)
    else:
        conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalAP)

    await conn.connect()
    print("[go2-mcp] WebRTC connected!", file=sys.stderr)
    return conn


# ---------------------------------------------------------------------------
# HTTP mode
# ---------------------------------------------------------------------------

async def run_http(args: argparse.Namespace) -> None:
    global _conn
    _conn = await _connect_robot(args)
    _setup_state_subscriptions(_conn)
    await _setup_video_track(_conn)

    session_manager = StreamableHTTPSessionManager(
        app=server, event_store=None, json_response=False, stateless=True,
    )

    async def mcp_handler(scope, receive, send):
        await session_manager.handle_request(scope, receive, send)

    starlette_app = Starlette(routes=[Mount("/mcp", app=mcp_handler)])

    print(f"[go2-mcp] Listening on http://{args.host}:{args.port}/mcp", file=sys.stderr)
    print(f"[go2-mcp] OpenWebUI -> Admin Panel -> Tools -> Add Connection", file=sys.stderr)
    print(f"[go2-mcp]   Type: MCP (Streamable HTTP)  URL: http://<YOUR-IP>:{args.port}/mcp", file=sys.stderr)

    config  = uvicorn.Config(starlette_app, host=args.host, port=args.port, log_level="warning")
    userver = uvicorn.Server(config)
    async with session_manager.run():
        await userver.serve()


# ---------------------------------------------------------------------------
# stdio mode
# ---------------------------------------------------------------------------

async def run_stdio(args: argparse.Namespace) -> None:
    global _conn
    _conn = await _connect_robot(args)
    _setup_state_subscriptions(_conn)
    await _setup_video_track(_conn)
    print("[go2-mcp] Running in stdio mode.", file=sys.stderr)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unitree Go2 WebRTC MCP Server")
    p.add_argument("--ip",       default="10.0.0.207", help="Robot IP (default: 10.0.0.207)")
    p.add_argument("--serial",   help="Robot serial number")
    p.add_argument("--remote",   action="store_true")
    p.add_argument("--username", help="Unitree account email (remote mode)")
    p.add_argument("--password", help="Unitree account password (remote mode)")
    p.add_argument("--host",     default="0.0.0.0")
    p.add_argument("--port",     default=8000, type=int)
    p.add_argument("--stdio",    action="store_true", help="Use stdio instead of HTTP")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        asyncio.run(run_stdio(args) if args.stdio else run_http(args))
    except KeyboardInterrupt:
        print("\n[go2-mcp] Shutting down.", file=sys.stderr)
