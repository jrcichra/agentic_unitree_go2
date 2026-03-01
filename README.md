# 🤖 Go2 Agentic Robotics Control

Control your Unitree Go2 robot dog entirely through natural language — no code, no buttons, just type. This toolkit gives you two powerful ways to interact with your robot.

Built on [unitree_webrtc_connect](https://github.com/legion1581/unitree_webrtc_connect) for WebRTC communication. Thanks to [@legion1581](https://github.com/legion1581) for the great library!

> This project was fully written by Claude.ai, Grok, MiniMax M2.5, and Qwen3.5.
> 
> **Disclaimer:** This software controls a real robot. Use at your own risk — I'm not responsible if your Go2 puts a hole in your drywall, trips over your cat, or does anything else dumb.

| Tool | Best For | Interface |
|------|----------|-----------|
| **`go2_mcp_server.py`** | AI assistants (OpenWebUI, Claude Desktop) | 50+ MCP tools |
| **`cli.py`** | Assign tasks — "find the red chair", "go to the kitchen" | Terminal TUI |

---

## Quick Start

```bash
# Set up virtual environment and install deps
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Option 1: MCP Server (for OpenWebUI / Claude Desktop)
python go2_mcp_server.py --ip 10.0.0.200 --port 8000

# Option 2: Natural Language TUI (for direct control)
python cli.py --ip 10.0.0.200 --model qwen3.5:35b
```

---

## Two Ways to Control

### 1. MCP Server — AI Agent Integration

A full-featured MCP server exposing 50+ tools for robot control via HTTP (Streamable HTTP) or stdio transport.

**Run (HTTP mode — for OpenWebUI):**
```bash
python go2_mcp_server.py --ip 10.0.0.200 --port 8000
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
python cli.py --ip 10.0.0.200 --model qwen3.5:35b
python cli.py --no-camera  # for headless operation
```

**Features:**
- **Live camera feed**: iTerm2/Kitty inline images (Alacritty, WezTerm, iTerm2) or half-block ASCII art fallback
- **Agentic control loop**: Model receives fresh camera frame after every action — it sees what the robot sees and adapts its next move accordingly
- **Thinking mode**: Qwen3/DeepSeek-R1 reasoning visible in real-time
- **Natural language**: "walk forward 1 metre", "turn left 90°", "find the red chair", "do a front flip"
- **Full-state telemetry**: Position, velocity, battery %, orientation (RPY), gait type, LiDAR obstacle distances
- **Autonomous tool calling**: move, turn, stance poses, tricks (flips, dances, waves), LED control, speed adjustment
- **Multi-model support**: Switch between any Ollama model on the fly (`model qwen3.5:35b`)
- **Conversation history**: Scrollable chat with rich markup highlighting
- **Error panel**: Toggleable view for debugging connection/command issues

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
| **Local STA** | Connect to robot as station | `--ip 10.0.0.200` or `--serial <SN>` |
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

---

## Todo

- [ ] Audio playback (speakers)
- [ ] Audio recording (microphone)
- [ ] Text to speech
