from math import pi
from threading import Event

from agentic_unitree_go2.robot.safety import (
    McpSafetyConfig,
    McpSafetyGovernor,
    TuiSafetyConfig,
    TuiSafetyGovernor,
    robot_is_fallen,
)


def test_robot_is_fallen_uses_roll_or_pitch_threshold():
    assert not robot_is_fallen([0.1, 0.2, 0.0])
    assert robot_is_fallen([0.6, 0.0, 0.0])
    assert robot_is_fallen([0.0, -0.6, 0.0])


def test_tui_governor_clamps_move_and_reports_note():
    governor = TuiSafetyGovernor(
        TuiSafetyConfig(max_step_m=1.0, max_turn_deg=90.0),
        state_summary=lambda: {"rpy_rad": [0.0, 0.0, 0.0]},
        forward_obstacle_m=lambda: 0.0,
    )

    allowed, args, note = governor.govern("move", {"x": 2.0, "y": -3.0, "z": pi})

    assert allowed
    assert args == {"x": 1.0, "y": -1.0, "z": pi / 2}
    assert "clamped" in note


def test_tui_governor_blocks_cancelled_motion():
    event = Event()
    event.set()
    governor = TuiSafetyGovernor(
        state_summary=lambda: {"rpy_rad": [0.0, 0.0, 0.0]},
        forward_obstacle_m=lambda: 0.0,
        cancel_requested=event,
    )

    allowed, _, note = governor.govern("move", {"x": 0.5})

    assert not allowed
    assert note == "cancelled: robot command ignored"


def test_mcp_governor_blocks_forward_obstacle():
    governor = McpSafetyGovernor(
        McpSafetyConfig(obstacle_stop_m=0.35),
        sport_state=lambda: {"imu_state": {"rpy": [0, 0, 0]}, "range_obstacle": [0.2]},
    )

    allowed, _, note = governor.govern("move", {"x": 0.5})

    assert not allowed
    assert note == "safety governor: forward obstacle at 0.20m"


def test_mcp_governor_blocks_high_risk_tools_by_default():
    governor = McpSafetyGovernor(
        sport_state=lambda: {"imu_state": {"rpy": [0, 0, 0]}, "range_obstacle": [0]}
    )

    allowed, _, note = governor.govern("flip", {"direction": "front"})

    assert not allowed
    assert "disabled" in note
