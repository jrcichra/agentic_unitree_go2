"""Mock Unitree Go2 connection used for offline development and tests."""

from __future__ import annotations

import asyncio
import json


def mock_status(code: int = 0, data=None) -> dict:
    return {
        "data": {
            "header": {"status": {"code": code}},
            "data": data if data is not None else "{}",
        }
    }


class MockPubSub:
    def __init__(self, rtc_topic: dict, *, state_tick_s: float = 0.04) -> None:
        self._rtc_topic = rtc_topic
        self._state_tick_s = state_tick_s
        self._callbacks: dict[str, list] = {}
        self.volume = 5
        self.position = [0.0, 0.0, 0.0]
        self.velocity = [0.0, 0.0, 0.0]

    async def publish_request_new(self, topic, payload):
        api_id = payload.get("api_id") if isinstance(payload, dict) else None
        parameter = payload.get("parameter", {}) if isinstance(payload, dict) else {}
        if api_id == 1003:
            self.volume = int(parameter.get("volume", self.volume))
        elif api_id == 1004:
            return mock_status(data=json.dumps({"volume": self.volume}))
        elif isinstance(parameter, dict) and {"x", "y", "z"} & set(parameter):
            self.velocity = [
                float(parameter.get("x", 0)),
                float(parameter.get("y", 0)),
                float(parameter.get("z", 0)),
            ]
            self.position[0] += self.velocity[0] * self._state_tick_s
            self.position[1] += self.velocity[1] * self._state_tick_s
            self.position[2] += self.velocity[2] * self._state_tick_s
            self._emit_state()
        await asyncio.sleep(0.01)
        return mock_status()

    def publish_without_callback(self, topic, payload):
        return None

    def subscribe(self, topic, callback):
        self._callbacks.setdefault(topic, []).append(callback)
        self._emit_state()

    def unsubscribe(self, topic):
        self._callbacks.pop(topic, None)

    def _emit_state(self):
        sport = {
            "position": [round(v, 3) for v in self.position],
            "velocity": [round(v, 3) for v in self.velocity],
            "imu_state": {"rpy": [0.0, 0.0, round(self.position[2], 3)]},
            "body_height": 0.31,
            "gait_type": 1,
            "range_obstacle": [0, 0, 0, 0],
        }
        low = {"bms_state": {"soc": 87, "voltage": 31.2}}
        multi = {"volume": self.volume, "mode": "mock"}

        for cb in self._callbacks.get(self._rtc_topic["LF_SPORT_MOD_STATE"], []):
            cb({"data": sport})
        for cb in self._callbacks.get(self._rtc_topic["LOW_STATE"], []):
            cb({"data": low})
        for cb in self._callbacks.get(self._rtc_topic.get("MULTIPLE_STATE"), []):
            cb({"data": json.dumps(multi)})


class MockDataChannel:
    def __init__(self, rtc_topic: dict, *, state_tick_s: float = 0.04) -> None:
        self.pub_sub = MockPubSub(rtc_topic, state_tick_s=state_tick_s)

    async def disableTrafficSaving(self, enabled):
        return None


class MockFrame:
    def __init__(self, tick: int, *, cv2_module=None, np_module=None) -> None:
        self.tick = tick
        self._cv2 = cv2_module
        self._np = np_module

    def to_ndarray(self, format="bgr24"):
        if self._cv2 is None or self._np is None:
            return None
        h, w = 360, 640
        img = self._np.zeros((h, w, 3), dtype=self._np.uint8)
        img[:, :] = (32, 40, 48)
        img[90:260, 60:220] = (40, 120, 220)
        img[120:310, 390:560] = (80, 180, 90)
        x = 260 + (self.tick * 7) % 80
        self._cv2.circle(img, (x, 180), 38, (230, 230, 70), -1)
        self._cv2.putText(
            img,
            "MOCK GO2 CAMERA",
            (170, 45),
            self._cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (235, 235, 235),
            2,
            self._cv2.LINE_AA,
        )
        return img


class MockVideoTrack:
    def __init__(self, *, cv2_module=None, np_module=None, frame_interval_s: float = 1.0) -> None:
        self.tick = 0
        self._cv2 = cv2_module
        self._np = np_module
        self._frame_interval_s = frame_interval_s

    async def recv(self):
        await asyncio.sleep(self._frame_interval_s)
        self.tick += 1
        return MockFrame(self.tick, cv2_module=self._cv2, np_module=self._np)


class MockVideoChannel:
    def __init__(self, *, cv2_module=None, np_module=None, frame_interval_s: float = 1.0) -> None:
        self._callbacks = []
        self._cv2 = cv2_module
        self._np = np_module
        self._frame_interval_s = frame_interval_s

    def add_track_callback(self, callback):
        self._callbacks.append(callback)

    def switchVideoChannel(self, enabled):
        if enabled:
            for cb in self._callbacks:
                asyncio.ensure_future(
                    cb(
                        MockVideoTrack(
                            cv2_module=self._cv2,
                            np_module=self._np,
                            frame_interval_s=self._frame_interval_s,
                        )
                    )
                )


class MockAudioChannel:
    def __init__(self):
        self._callbacks = []

    def add_track_callback(self, callback):
        self._callbacks.append(callback)

    def switchAudioChannel(self, enabled):
        return None


class MockConnection:
    mock = True

    def __init__(
        self,
        rtc_topic: dict,
        *,
        cv2_module=None,
        np_module=None,
        state_tick_s: float = 0.04,
        frame_interval_s: float = 1.0,
        include_audio: bool = False,
    ) -> None:
        self.datachannel = MockDataChannel(rtc_topic, state_tick_s=state_tick_s)
        self.video = MockVideoChannel(
            cv2_module=cv2_module,
            np_module=np_module,
            frame_interval_s=frame_interval_s,
        )
        if include_audio:
            self.audio = MockAudioChannel()
        self.pc = None

    async def connect(self):
        await asyncio.sleep(0.05)


def is_mock_conn(conn) -> bool:
    return bool(getattr(conn, "mock", False))
