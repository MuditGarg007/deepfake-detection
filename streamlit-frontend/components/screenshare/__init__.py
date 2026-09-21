"""Screen-share capture as a Streamlit component.

Streamlit cannot reach `getDisplayMedia`, and its rerun model is the wrong
shape for a 2 fps stream, so capture and scoring live in the iframe: the JS
opens a live session on the backend, posts each sampled frame to it, and
paints the verdict the backend smooths. Python only starts the component and
receives the summary row the component sends back when the share stops.
"""

from pathlib import Path

import streamlit.components.v1 as components

_FRONTEND_DIR = Path(__file__).parent / "frontend"

_component = components.declare_component("screenshare", path=str(_FRONTEND_DIR))

# Defaults come from the performance budget in docs/live-screenshare-refactor.md.
# JPEG quality in particular is not a free knob: the classifier reads a known
# fake as REAL below ~0.85, so do not lower it without re-running that
# measurement.
INTERVAL_MS = 500
CAPTURE_WIDTH = 640
JPEG_QUALITY = 0.95

# Only the sparkline's threshold rules; the badge state itself is decided by
# the backend, which owns the one set of thresholds both paths share.
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
    """Render the capture panel.

    `api_url` must be reachable from the browser, not just from the Streamlit
    server. Returns None while nothing has finished, and for the most recent
    completed share the `live_sessions` row the backend wrote - id, started_at,
    ended_at, duration_seconds, frames_scored, mean_probability,
    peak_probability, status, timeline - plus `frames_sent`, which only the
    browser knows. A share that never saw a face records nothing and returns
    None.
    """
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
