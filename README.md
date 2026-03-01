# 🤖 Go2 Agentic Robotics Control

Control your Unitree Go2 robot dog entirely through natural language — no code, no buttons, just type. This toolkit gives you two powerful ways to interact with your robot:

| Tool | Best For | Interface |
|------|----------|-----------|
| **`go2_mcp_server.py`** | AI assistants (OpenWebUI, Claude Desktop) | 50+ MCP tools |
| **`cli.py`** | Direct conversation with live camera | Terminal TUI |

---

## Quick Start

```bash
# Install dependencies
pip uninstall unitree_webrtc_connect -y
pip install "git+https://github.com/12-hak/unitree_webrtc_connect.git@patch-1"

# Option 1: MCP Server (for OpenWebUI / Claude Desktop)
pip install "mcp[cli]" uvicorn starlette opencv-python-headless numpy
python go2_mcp_server.py --ip 10.0.0.207 --port 8000

# Option 2: Natural Language TUI (for direct control)
pip install opencv-python-headless numpy requests pillow textual rich
python cli.py --ip 10.0.0.207 --model qwen3.5:35b
```

---

## Two Ways to Control

### 1. MCP Server — AI Agent Integration

A full-featured MCP server exposing 50+ tools for robot control via HTTP (Streamable HTTP) or stdio transport.

**Run (HTTP mode — for OpenWebUI):**
```bash
python go2_mcp_server.py --ip 10.0.0.207 --port 8000
```
Then configure OpenWebUI: **Admin Panel → Settings → Tools → Add Connection → MCP (Streamable HTTP) → URL: `http://<your-ip>:8000/mcp`**

**Run (stdio mode — for Claude Desktop):**
```bash
python go2_mcp_server.py --stdio
```

**Available tools (50+):**
- **Movement**: `move(x,y,z)`, `stop()` — displacement-based control with LiDAR obstacle avoidance
- **Stances**: `stand_up`, `stand_down`, `balance_stand`, `recovery_stand`, `sit`, `back_stand`
- **Gaits**: `set_gait(economic|static|trot_run|free_walk|free_bound|free_jump|...)`
- **Body control**: `set_body_height`, `set_foot_raise_height`, `set_speed_level(0-2)`, `set_euler`
- **Social**: `hello`, `show_heart`, `stretch`, `scrape`, `wallow`
- **Dances**: `dance(routine=1|2)`
- **Acrobatics**: `flip(front|back|left|right)`, `handstand`, `front_jump`, `front_pounce`, `wiggle_hips`
- **Motion mode**: `set_motion_mode(normal|ai|obstacle_avoidance)`, `get_motion_mode`
- **Telemetry**: `get_sport_state`, `get_low_state`, `get_multiple_state`, `get_robot_state`
- **VUI**: `set_led_color`, `set_brightness(0-10)`, `set_volume(0-10)`
- **Sensors**: `lidar_snapshot()` — point cloud + bounding box, `capture_image(quality=1-100)` — base64 JPEG

> **Firmware 1.1.7+ (MCF mode):** The server automatically handles MCF unified mode. Acrobatics work directly — error 7004 is expected and handled gracefully.

---

### 2. CLI — Natural Language TUI

A full-screen terminal UI for conversational robot control with embedded camera feed.

```bash
python cli.py --ip 10.0.0.207 --model qwen3.5:35b
python cli.py --no-camera  # for headless operation
```

**Features:**
- **Live camera view**: iTerm2/Kitty inline images (Alacritty, WezTerm, iTerm2) or half-block fallback
- **Agentic AI control**: Powered by Ollama with Qwen3/DeepSeek-R1 thinking mode — model autonomously plans and executes, adapting after each action based on fresh camera feedback
- **Natural language commands**: "walk forward 1 metre", "turn left 90 degrees", "find the red chair", "do a front flip"
- **Real-time feedback**: Live robot state (position, velocity, battery, LiDAR distances) with thinking visualization
- **Rich conversation history**: Scrollable view with markup highlighting

**Keyboard shortcuts:**
- `Enter` — Send command
- `Ctrl+L` — Clear conversation history
- `Ctrl+E` — Toggle error panel
- `F1` — Print robot state snapshot
- `F2` — Show help screen
- `F3` — View system prompt
- `Ctrl+C` — Quit (safely stops robot first)

**Built-in text commands:**
- `state` — Print robot state JSON
- `clear` — Clear conversation
- `model <name>` — Switch Ollama model on the fly

---

## Connection Modes

| Mode | How to Connect | Command Flag |
|------|----------------|--------------|
| **Local AP** (default) | Robot's built-in Wi-Fi | `python script.py` |
| **Local STA** | Connect to robot as station | `--ip 10.0.0.207` or `--serial <SN>` |
| **Remote** | Via Unitree cloud | `--remote --serial <SN> --username <user> --password <pass>` |
| **MCF** (firmware 1.1.7+) | Unified mode — all commands work directly | Automatic |

---

## Safety Features

- **Obstacle avoidance guard**: `move()` stops if LiDAR detects obstacles within 0.35m during forward motion
- **Recovery stance**: Use `stance(recovery_stand)` if robot has fallen (pitch/roll > 0.5 rad)
- **Motion mode warnings**: Scripts warn before flips/handstands require clear 2m space
- **Automatic stop on quit**: `Ctrl+C` safely stops and balances robot before exiting
- **Error recovery**: 7004 errors (MCF mode) handled gracefully — no manual intervention needed

---

## Architecture

- **Communication**: WebRTC via `unitree_webrtc_connect` library
- **Camera**: VideoTrack streamed via WebRTC, converted to JPEG for display
- **MCP Server**: Implements MCP protocol (list_tools, call_tool, capabilities) for LLM integration
- **TUI**: Textual framework with inline image rendering via terminal escape sequences
- **LLM**: Ollama with configurable model (Qwen3, DeepSeek-R1, LLaVA, etc.)
- **Shared**: Error handler patch fixes broken unpacking in unitree_webrtc_connect library
