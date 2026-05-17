"""Shared safety governor for robot tool calls."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from threading import Event
from typing import Callable


FALLEN_RPY_THRESHOLD_RAD = 0.5
DEFAULT_OBSTACLE_STOP_M = 0.35


def as_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def robot_is_fallen(rpy) -> bool:
    if not isinstance(rpy, list) or len(rpy) < 2:
        return False
    return (
        abs(as_float(rpy[0])) > FALLEN_RPY_THRESHOLD_RAD
        or abs(as_float(rpy[1])) > FALLEN_RPY_THRESHOLD_RAD
    )


@dataclass
class TuiSafetyConfig:
    max_step_m: float = 1.0
    max_turn_deg: float = 720.0
    allow_acrobatics: bool = False
    obstacle_stop_m: float = DEFAULT_OBSTACLE_STOP_M


@dataclass
class McpSafetyConfig:
    max_step_m: float = 1.0
    max_turn_rad: float = 4 * pi
    allow_acrobatics: bool = False
    obstacle_stop_m: float = DEFAULT_OBSTACLE_STOP_M


TUI_HIGH_RISK_TRICKS = {
    "front_flip",
    "back_flip",
    "left_flip",
    "right_flip",
    "handstand",
    "front_jump",
    "front_pounce",
}

MCP_HIGH_RISK_TOOLS = {
    "flip",
    "handstand",
    "front_jump",
    "front_pounce",
    "moon_walk",
    "one_sided_step",
    "bound",
}

MCP_MOTION_TOOLS = {
    "navigate",
    "move",
    "set_gait",
    "lead_follow",
    "set_body_height",
    "set_foot_raise_height",
    "set_speed_level",
    "set_euler",
    "hello",
    "show_heart",
    "stretch",
    "scrape",
    "wallow",
    "dance",
    "flip",
    "handstand",
    "front_jump",
    "front_pounce",
    "wiggle_hips",
    "finger_heart",
    "moon_walk",
    "one_sided_step",
    "bound",
}


class TuiSafetyGovernor:
    def __init__(
        self,
        config: TuiSafetyConfig | None = None,
        *,
        state_summary: Callable[[], dict],
        forward_obstacle_m: Callable[[], float],
        cancel_requested: Event | None = None,
    ) -> None:
        self.config = config or TuiSafetyConfig()
        self._state_summary = state_summary
        self._forward_obstacle_m = forward_obstacle_m
        self._cancel_requested = cancel_requested

    def robot_is_fallen(self) -> bool:
        return robot_is_fallen(self._state_summary().get("rpy_rad", [0, 0, 0]))

    def motion_blocked_by_state(self, name: str) -> str | None:
        if self.robot_is_fallen():
            if name == "stance":
                return None
            return "safety governor: robot appears fallen; use stance(recovery_stand) first"
        return None

    def govern(self, name: str, args: dict) -> tuple[bool, dict, str | None]:
        args = dict(args or {})
        if (
            self._cancel_requested is not None
            and self._cancel_requested.is_set()
            and name not in {"stance", "stop_audio"}
        ):
            return False, args, "cancelled: robot command ignored"

        if name == "move":
            blocked = self.motion_blocked_by_state(name)
            if blocked:
                return False, args, blocked

            max_step = float(self.config.max_step_m)
            max_yaw = float(self.config.max_turn_deg) * pi / 180.0
            values = {
                "x": as_float(args.get("x", 0)),
                "y": as_float(args.get("y", 0)),
                "z": as_float(args.get("z", 0)),
            }
            clamped = False
            for key, limit in (("x", max_step), ("y", max_step), ("z", max_yaw)):
                new_value = clamp(values[key], -limit, limit)
                clamped = clamped or new_value != values[key]
                args[key] = new_value
            if args["x"] > 0:
                dist = self._forward_obstacle_m()
                if 0 < dist < float(self.config.obstacle_stop_m):
                    return False, args, f"safety governor: forward obstacle at {dist:.2f}m"
            if clamped:
                return (
                    True,
                    args,
                    f"safety governor: command clamped to x={args['x']:.2f}, "
                    f"y={args['y']:.2f}, z={args['z']:.2f}",
                )
            return True, args, None

        if name == "turn":
            blocked = self.motion_blocked_by_state(name)
            if blocked:
                return False, args, blocked
            deg = as_float(args.get("degrees", 0))
            clamped = clamp(deg, -float(self.config.max_turn_deg), float(self.config.max_turn_deg))
            args["degrees"] = clamped
            if clamped != deg:
                return True, args, f"safety governor: turn clamped to {clamped:.1f} degrees"
            return True, args, None

        if name == "trick":
            blocked = self.motion_blocked_by_state(name)
            if blocked:
                return False, args, blocked
            trick = str(args.get("name", ""))
            if trick in TUI_HIGH_RISK_TRICKS and not self.config.allow_acrobatics:
                return (
                    False,
                    args,
                    f"safety governor: {trick} is disabled; restart with --allow-acrobatics to enable it",
                )
            return True, args, None

        if name == "look":
            for key in ("roll", "pitch", "yaw"):
                args[key] = clamp(as_float(args.get(key, 0)), -0.35, 0.35)
            return True, args, None

        if name == "led":
            args["duration"] = max(1, min(30, int(as_float(args.get("duration", 3), 3))))
            return True, args, None

        return True, args, None


class McpSafetyGovernor:
    def __init__(
        self,
        config: McpSafetyConfig | None = None,
        *,
        sport_state: Callable[[], dict],
    ) -> None:
        self.config = config or McpSafetyConfig()
        self._sport_state = sport_state

    def robot_is_fallen(self) -> bool:
        return robot_is_fallen(self._sport_state().get("imu_state", {}).get("rpy", [0, 0, 0]))

    def forward_obstacle_m(self) -> float:
        ranges = self._sport_state().get("range_obstacle", [0, 0, 0, 0])
        if isinstance(ranges, list) and ranges:
            return as_float(ranges[0])
        return 0.0

    def motion_blocked_by_state(self, name: str) -> str | None:
        recovery_tools = {
            "stop",
            "recovery_stand",
            "stand_down",
            "damp",
            "get_sport_state",
            "get_low_state",
            "get_robot_state",
        }
        if self.robot_is_fallen() and name not in recovery_tools:
            return "safety governor: robot appears fallen; use recovery_stand first"
        return None

    def govern(self, name: str, args: dict) -> tuple[bool, dict, str | None]:
        args = dict(args or {})
        if name in MCP_MOTION_TOOLS:
            blocked = self.motion_blocked_by_state(name)
            if blocked:
                return False, args, blocked

        if name == "move":
            max_step = float(self.config.max_step_m)
            max_turn = float(self.config.max_turn_rad)
            clamped = False
            for key, limit in (("x", max_step), ("y", max_step), ("z", max_turn)):
                old_value = as_float(args.get(key, 0))
                args[key] = clamp(old_value, -limit, limit)
                clamped = clamped or args[key] != old_value
            if args["x"] > 0:
                dist = self.forward_obstacle_m()
                if 0 < dist < float(self.config.obstacle_stop_m):
                    return False, args, f"safety governor: forward obstacle at {dist:.2f}m"
            if clamped:
                return (
                    True,
                    args,
                    f"safety governor: move clamped to x={args['x']:.2f}, "
                    f"y={args['y']:.2f}, z={args['z']:.2f}",
                )
            return True, args, None

        if name in MCP_HIGH_RISK_TOOLS and not self.config.allow_acrobatics:
            return (
                False,
                args,
                f"safety governor: {name} is disabled; restart with --allow-acrobatics to enable it",
            )

        if name == "set_euler":
            for key in ("roll", "pitch", "yaw"):
                args[key] = clamp(as_float(args.get(key, 0)), -0.35, 0.35)
        elif name == "set_body_height":
            args["height"] = clamp(as_float(args.get("height", 0)), -0.12, 0.12)
        elif name == "set_foot_raise_height":
            args["height"] = clamp(as_float(args.get("height", 0.08)), 0.0, 0.12)
        elif name == "set_speed_level":
            args["level"] = max(0, min(1, int(as_float(args.get("level", 1), 1))))
        elif name == "set_volume":
            args["volume"] = max(0, min(10, int(as_float(args.get("volume", 5), 5))))
        elif name == "set_brightness":
            args["brightness"] = max(0, min(10, int(as_float(args.get("brightness", 5), 5))))
        elif name == "set_led_color":
            args["duration"] = max(1, min(30, int(as_float(args.get("duration", 3), 3))))
        return True, args, None
