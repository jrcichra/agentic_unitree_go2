# Go2 Agentic Robotics Control

*Agentic command-line interface and MCP server for Unitree Go2 robot control*

A complete agentic toolkit for controlling the Unitree Go2 robot dog via natural language and MCP (Model Context Protocol) integration.

## Scripts

### 1. `go2_mcp_server.py` — MCP Server for OpenWebUI and Claude Desktop

A full-featured MCP server that exposes 50+ tools for robot control via HTTP (Streamable HTTP) or stdio transport.

**Install dependencies:**
```bash
pip uninstall unitree_webrtc_connect -y
pip install "git+https://github.com/12-hak/unitree_webrtc_connect.git@patch-1"
pip install "mcp[cli]" uvicorn starlette opencv-python-headless numpy
```

**Run (HTTP mode — for OpenWebUI):**
```bash
python go2_mcp_server.py
python go2_mcp_server.py --ip 10.0.0.207 --port 8000
```
Then configure OpenWebUI:
- Admin Panel → Settings → Tools → Add Connection
- Type: MCP (Streamable HTTP)
- URL: `http://<your-ip>:8000/mcp`

**Run (stdio mode — for Claude Desktop):**
```bash
python go2_mcp_server.py --stdio
```

**Available tools (50+):**
- **Movement**: `move(x,y,z)`, `stop()` — displacement-based control with obstacle avoidance guard
- **Stance**: `stand_up`, `stand_down`, `balance_stand`, `recovery_stand`, `sit`, `rise_sit`, `damp`, `back_stand`
- **Gaits**: `set_gait(economic|static|trot_run|free_walk|free_bound|free_jump|free_avoid|classic|cross_step|continuous)`, `lead_follow`
- **Body control**: `set_body_height`, `set_foot_raise_height`, `set_speed_level(0-2)`, `set_euler(roll,pitch,yaw)`
- **Social actions**: `hello`, `show_heart`, `stretch`, `scrape`, `wallow`
- **Dances**: `dance(routine=1|2)`
- **Acrobatics**: `flip(front|back|left|right)`, `handstand`, `front_jump`, `front_pounce`, `wiggle_hips`
- **Special moves**: `finger_heart`, `moon_walk`, `one_sided_step`, `bound`
- **Configuration**: `set_motion_mode(normal|ai|obstacle_avoidance)`, `get_motion_mode`, `set_auto_recovery`, `set_avoid_mode`, `set_obstacle_avance`
- **Telemetry**: `get_sport_state`, `get_low_state`, `get_multiple_state`, `get_robot_state`
- **VUI (lights/sound)**: `set_led_color`, `set_brightness(0-10)`, `set_volume(0-10)`, `get_volume`
- **Sensors**: `lidar_snapshot()` — get LiDAR point cloud and bounding box, `capture_image(quality=1-100)` — get base64-encoded camera frame

**Note on firmware 1.1.7+ (MCF mode):** The server automatically handles MCF unified mode. Acrobatics tools work directly without switching motion mode — error 7004 is expected and handled gracefully.

### 2. `cli.py` — Natural Language TUI

A full-screen terminal UI for conversational robot control with embedded camera feed rendering.

**Install dependencies:**
```bash
pip uninstall unitree_webrtc_connect -y
pip install "git+https://github.com/12-hak/unitree_webrtc_connect.git@patch-1"
pip install opencv-python-headless numpy requests pillow textual rich
```

**Run the TUI:**
```bash
python cli.py
python cli.py --ip 10.0.0.207 --model qwen3.5:35b
python cli.py --no-camera --model llava
```

**Features:**
- **Inline camera view**: Uses iTerm2/Kitty native inline image protocol for crisp display (falls back to half-block art for other terminals)
- **Agentic AI control**: Powered by Ollama with Qwen3 / DeepSeek-R1 thinking mode. The model automatically plans and executes tools, adapting after each action based on fresh camera feedback
- **Natural language commands**: "walk forward 1 metre", "turn left 90 degrees", "find the red chair", "do a front flip"
- **Real-time feedback**: Live robot state (position, velocity, battery %, LiDAR obstacle distances) with thinking visualization
- **Rich log history**: Scrollable conversation view with markup highlighting
- **Built-in error handling**: Errors displayed in a toggleable panel

**Keyboard shortcuts:**
- `Enter` — Send command
- `Ctrl+L` — Clear conversation history
- `Ctrl+E` — Toggle error panel visibility
- `F1` — Print robot state snapshot
- `F2` — Show help screen
- `F3` — View system prompt
- `Ctrl+C` — Quit (stops robot first)

**Built-in text commands:**
- `state` — Print robot state JSON
- `clear` — Clear conversation
- `model <name>` — Switch Ollama model on the fly

## Connection Modes

The scripts connect to the Go2 via WebRTC in four ways:
1. **Local AP** (default) — connects to robot's built-in Wi-Fi access point
2. **Local STA** — connects to robot as a station (provide `--ip` or `--serial`)
3. **Remote** — connects via Unitree cloud (provide `--remote --serial --username --password`)
4. **MCF firmware 1.1.7+** — unified mode for all modern robots, handles acrobatics directly

## Safety Features

- **Obstacle avoidance guard**: The `move()` tool stops if LiDAR detects obstacles within 0.35m during forward motion
- **Recovery stance**: Use `stance(recovery_stand)` if robot has fallen (pitch/roll > 0.5 rad)
- **Motion mode warnings**: Scripts warn before flips/handstands require clear 2m space
- **Automatic stop on quit**: `Ctrl+C` safely stops and balances robot before exiting
- **Error recovery**: 7004 errors (MCF mode) are handled gracefully — no manual intervention needed

## Architecture

- Both scripts use `unitree_webrtc_connect` library for WebRTC communication
- Camera frames streamed via WebRTC VideoTrack, converted to JPEG for display
- MCP server implements MCP protocol (list_tools, call_tool, capabilities) for LLM integration
- TUI uses Textual framework with inline image rendering via escape sequences
- Both use Ollama for LLM — configurable model and URL
- Shared error_handler patch fixes broken unpacking in unitree_webrtc_connect library
