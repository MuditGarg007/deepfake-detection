import requests
import streamlit as st

from components.screenshare import screenshare

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Deepfake Detection")
st.title("Deepfake Detection")
st.caption(API_URL)

live_tab, history_tab = st.tabs(["Live", "History"])


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


with live_tab:
    summary = screenshare(API_URL)

    if summary:
        if st.session_state.get("last_live_id") != summary["id"]:
            st.session_state["last_live_id"] = summary["id"]
            st.session_state.pop("history", None)

        st.subheader("Last share")
        verdict_banner(summary["status"], summary["mean_probability"])
        columns = st.columns(3)
        columns[0].metric("Mean", f"{summary['mean_probability']:.1%}")
        columns[1].metric("Peak", f"{summary['peak_probability']:.1%}")
        columns[2].metric("Duration", f"{summary['duration_seconds']:.0f}s")
        st.caption(
            f"{summary['frames_scored']} of {summary.get('frames_sent', 0)} frames "
            f"had a face · session {summary['id']}"
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
        st.caption("Nothing yet.")
    else:
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
                "Session",
                live_rows,
                format_func=lambda row: f"#{row['id']} · {row['status']} · {row['created_at']}",
            )
            detail = requests.get(
                f"{API_URL}/live/sessions/{chosen['id']}", timeout=60
            )
            if detail.ok:
                session = detail.json()
                verdict_banner(session["status"], session["mean_probability"])
                st.caption(
                    f"{session['frames_scored']} frames over "
                    f"{session['duration_seconds']:.0f}s · peak "
                    f"{session['peak_probability']:.1%}"
                )
                if session["timeline"]:
                    timeline_chart(session["timeline"], "second")
