import os

import requests
import streamlit as st

API_URL = os.environ.get(
    "API_URL", os.environ.get("NEXT_PUBLIC_API_URL", "http://localhost:8000")
).rstrip("/")

# The backend rejects anything else with a 400.
ALLOWED_EXTENSIONS = ["mp4", "avi", "mov"]

st.set_page_config(page_title="Deepfake Detection")
st.title("Deepfake Detection")
st.caption(f"API: {API_URL}")

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

    status = result["status"]
    probability = result["fake_probability"]
    message = f"{status} : {probability:.1%} fake probability"
    if status == "REAL":
        st.success(message)
    elif status == "SUSPICIOUS":
        st.warning(message)
    else:
        st.error(message)

    start, end = result["suspicious_start"], result["suspicious_end"]
    st.write(f"File: {result['filename']}")
    st.write(f"Frames scored: {len(result['frame_scores'])}")
    st.write(
        f"Suspicious region: {start:.1f}s – {end:.1f}s"
        if start is not None and end is not None
        else "Suspicious region: none"
    )

    if result["frame_scores"]:
        st.line_chart(
            {
                "second": [f["timestamp"] for f in result["frame_scores"]],
                "fake probability": [
                    f["fake_probability"] for f in result["frame_scores"]
                ],
            },
            x="second",
            y="fake probability",
        )
        frame = requests.get(f"{API_URL}/analysis/{result['id']}/frame", timeout=60)
        if frame.ok:
            st.image(frame.content, caption="Most suspicious frame")
