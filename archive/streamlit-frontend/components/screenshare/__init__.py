from pathlib import Path

import streamlit.components.v1 as components

_FRONTEND_DIR = Path(__file__).parent / "frontend"

_component = components.declare_component("screenshare", path=str(_FRONTEND_DIR))

INTERVAL_MS = 500
CAPTURE_WIDTH = 640
JPEG_QUALITY = 0.95

RISK_SUSPICIOUS = 0.4
RISK_HIGH = 0.7


def screenshare(
    api_url,
    interval_ms=INTERVAL_MS,
    capture_width=CAPTURE_WIDTH,
    jpeg_quality=JPEG_QUALITY,
    risk_suspicious=RISK_SUSPICIOUS,
    risk_high=RISK_HIGH,
    key="screenshare",
):
    return _component(
        api_url=api_url.rstrip("/"),
        interval_ms=interval_ms,
        capture_width=capture_width,
        jpeg_quality=jpeg_quality,
        risk_suspicious=risk_suspicious,
        risk_high=risk_high,
        key=key,
        default=None,
    )
