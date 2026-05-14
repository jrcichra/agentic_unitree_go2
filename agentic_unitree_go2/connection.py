"""Connection helpers for Unitree WebRTC."""

from __future__ import annotations

import logging


def patch_unitree_error_handler(logger: logging.Logger | None = None, prefix: str = "go2") -> None:
    """Make the upstream error handler tolerant of malformed robot error payloads."""
    log = logger or logging.getLogger(__name__)
    try:
        from unitree_webrtc_connect.msgs import error_handler as error_handler

        def _safe_handle_error(error):
            try:
                if isinstance(error, (list, tuple)) and len(error) == 3:
                    timestamp, error_source, error_code_int = error
                    original = getattr(error_handler, "_original_handle_error_logic", None)
                    if original is not None:
                        original(timestamp, error_source, error_code_int)
                else:
                    log.warning("[%s] Unexpected error format from robot ignored: %r", prefix, error)
            except Exception as exc:
                log.warning("[%s] Error handler exception ignored: %s; raw error: %r", prefix, exc, error)

        error_handler.handle_error = _safe_handle_error
    except Exception as exc:
        log.warning("[%s] Could not patch error_handler; continuing: %s", prefix, exc)


def build_webrtc_connection(
    *,
    ip: str | None = None,
    serial: str | None = None,
    remote: bool = False,
    username: str | None = None,
    password: str | None = None,
):
    from unitree_webrtc_connect.webrtc_driver import UnitreeWebRTCConnection, WebRTCConnectionMethod

    if remote:
        return UnitreeWebRTCConnection(
            WebRTCConnectionMethod.Remote,
            serialNumber=serial,
            username=username,
            password=password,
        )
    if serial and not ip:
        return UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, serialNumber=serial)
    if ip:
        return UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=ip)
    return UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalAP)
