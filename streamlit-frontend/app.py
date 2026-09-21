import os

import requests
import streamlit as st

from components.screenshare import screenshare

API_URL = os.environ.get(
    "API_URL", os.environ.get("NEXT_PUBLIC_API_URL", "http://localhost:8000")
).rstrip("/")

# The screen-share component posts from the browser, not from this process, so
# it needs an address the browser can reach. Locally that is the same one; on
# Cloud Run the backend has its own public URL.
BROWSER_API_URL = os.environ.get("PUBLIC_API_URL", API_URL).rstrip("/")

ALLOWED_EXTENSIONS = ["mp4", "avi", "mov"]

st.set_page_config(page_title="Deepfake Detection")
st.title("Deepfake Detection")
st.caption(f"API: {API_URL}")

upload_tab, live_tab, history_tab = st.tabs(
    ["Upload a video", "Live screen share", "History"]
)


def verdict_banner(status, probability):
    message = f"{status} : {probability:.1%} fake probability"
    if status == "REAL":
        st.success(message)
    elif status == "SUSPICIOUS":
        st.warning(message)
    else:
        st.error(message)


def timeline_chart(scores, x_label):
    st.line_chart(
        {
            x_label: [point["timestamp"] for point in scores],
            "fake probability": [point["fake_probability"] for point in scores],
        },
        x=x_label,
        y="fake probability",
    )


with upload_tab:
    upload = st.file_uploader("Video", type=ALLOWED_EXTENSIONS)

    if st.button("Run analysis", disabled=upload is None):
        with st.spinner("Analyzing..."):
            try:
                response = requests.post(
                    f"{API_URL}/analyze",
                    files={"file": (upload.name, upload.getvalue())},
                    timeout=600,
                )
            except requests.RequestException as exc:
                st.session_state.pop("result", None)
                st.session_state["error"] = f"Could not reach the API: {exc}"
            else:
                if response.status_code == 201:
                    st.session_state["result"] = response.json()
                    st.session_state.pop("error", None)
                else:
                    st.session_state.pop("result", None)
                    try:
                        detail = response.json().get("detail", response.text)
                    except ValueError:
                        detail = response.text
                    st.session_state["error"] = f"{response.status_code}: {detail}"

    if error := st.session_state.get("error"):
        st.error(error)

    if result := st.session_state.get("result"):
        st.subheader("Result")

        verdict_banner(result["status"], result["fake_probability"])

        start, end = result["suspicious_start"], result["suspicious_end"]
        st.write(f"File: {result['filename']}")
        st.write(f"Frames scored: {len(result['frame_scores'])}")
        st.write(
            f"Suspicious region: {start:.1f}s – {end:.1f}s"
            if start is not None and end is not None
            else "Suspicious region: none"
        )

        if result["frame_scores"]:
            timeline_chart(result["frame_scores"], "second")
            frame = requests.get(f"{API_URL}/analysis/{result['id']}/frame", timeout=60)
            if frame.ok:
                st.image(frame.content, caption="Most suspicious frame")

with live_tab:
    st.write(
        "Share a tab or window that is playing a face, and every frame is scored "
        "as it plays. Nothing is recorded — see the note below."
    )

    summary = screenshare(BROWSER_API_URL)

    if summary:
        # A finished share is a new history row, so the cached listing is stale.
        if st.session_state.get("last_live_id") != summary["id"]:
            st.session_state["last_live_id"] = summary["id"]
            st.session_state.pop("history", None)

        st.subheader("Last share")
        verdict_banner(summary["status"], summary["mean_probability"])
        columns = st.columns(3)
        columns[0].metric("Mean", f"{summary['mean_probability']:.1%}")
        columns[1].metric("Peak", f"{summary['peak_probability']:.1%}")
        columns[2].metric("Duration", f"{summary['duration_seconds']:.0f}s")
        st.write(
            f"{summary['frames_scored']} of {summary.get('frames_sent', 0)} frames "
            f"had a face · saved as live session {summary['id']}"
        )
        if summary["timeline"]:
            timeline_chart(summary["timeline"], "second")

with history_tab:
    if st.button("Refresh"):
        st.session_state.pop("history", None)

    if "history" not in st.session_state:
        try:
            response = requests.get(f"{API_URL}/history", timeout=60)
            response.raise_for_status()
        except requests.RequestException as exc:
            st.session_state["history"] = {"error": f"Could not reach the API: {exc}"}
        else:
            st.session_state["history"] = {"rows": response.json()}

    history = st.session_state["history"]
    if error := history.get("error"):
        st.error(error)
    elif not history["rows"]:
        st.info("Nothing analysed yet.")
    else:
        # Uploads and live sessions come back interleaved, newest first; ids
        # are only unique within a kind, so the label carries the kind too.
        st.dataframe(
            [
                {
                    "kind": row["kind"],
                    "id": row["id"],
                    "name": row["filename"],
                    "fake probability": row["fake_probability"],
                    "status": row["status"],
                    "when": row["created_at"],
                }
                for row in history["rows"]
            ],
            hide_index=True,
            width="stretch",
        )

        live_rows = [row for row in history["rows"] if row["kind"] == "live"]
        if live_rows:
            chosen = st.selectbox(
                "Show a live session",
                live_rows,
                format_func=lambda row: f"#{row['id']} · {row['status']} · {row['created_at']}",
            )
            detail = requests.get(
                f"{API_URL}/live/sessions/{chosen['id']}", timeout=60
            )
            if detail.ok:
                session = detail.json()
                verdict_banner(session["status"], session["mean_probability"])
                st.write(
                    f"{session['frames_scored']} frames scored over "
                    f"{session['duration_seconds']:.0f}s · peak "
                    f"{session['peak_probability']:.1%}"
                )
                if session["timeline"]:
                    timeline_chart(session["timeline"], "second")
