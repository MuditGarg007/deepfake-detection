"use client";

import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

import {
  ApiError,
  errorMessage,
  sendLiveFrame,
  startLive,
  stopLive,
} from "@/lib/api";
import { STATUS_LABEL, percent, seconds } from "@/lib/format";
import type { LiveFrame, LiveSession, RiskStatus } from "@/lib/types";
import { LiveSummary } from "./live-summary";
import { Timeline } from "./timeline";
import { Button, Card, ErrorNote, Verdict, cx } from "./ui";

const INTERVAL_MS = 500;
const CAPTURE_WIDTH = 640;
const JPEG_QUALITY = 0.95;
const MAX_FAILURES = 5;

type Phase = "idle" | "starting" | "sharing" | "saving";

// Document Picture-in-Picture (Chrome/Edge) is not in the TS DOM lib yet.
type PipApi = {
  requestWindow(options: { width: number; height: number }): Promise<Window>;
};

// The overlay window has no Tailwind, so it uses the globals.css colours inline.
const OVERLAY_COLOR: Record<RiskStatus, string> = {
  REAL: "#2f6b3f",
  SUSPICIOUS: "#95650a",
  HIGH_RISK: "#a3302a",
};

export function ScreenShare({ onSaved }: { onSaved: () => void }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const sessionRef = useRef<string | null>(null);
  const timerRef = useRef<number | null>(null);
  const busyRef = useRef(false);
  const failuresRef = useRef(0);
  const startedAtRef = useRef(0);
  const pipRef = useRef<Window | null>(null);

  const [phase, setPhase] = useState<Phase>("idle");
  const [frame, setFrame] = useState<LiveFrame | null>(null);
  const [sent, setSent] = useState(0);
  const [elapsed, setElapsed] = useState(0);
  const [summary, setSummary] = useState<LiveSession | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pip, setPip] = useState<Window | null>(null);
  const [overlayBlocked, setOverlayBlocked] = useState(false);

  useEffect(() => {
    const video = videoRef.current;
    return () => {
      if (timerRef.current !== null) window.clearInterval(timerRef.current);
      streamRef.current?.getTracks().forEach((track) => track.stop());
      if (video) video.srcObject = null;
      pipRef.current?.close();
    };
  }, []);

  function release() {
    if (timerRef.current !== null) window.clearInterval(timerRef.current);
    timerRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    pipRef.current?.close();
    pipRef.current = null;
    setOverlayBlocked(false);
  }

  // Opens the floating verdict window. Chrome only allows it a few seconds
  // after a click, so if the share picker took longer we fall back to a button.
  async function openOverlay() {
    const api = (window as { documentPictureInPicture?: PipApi })
      .documentPictureInPicture;
    if (!api || pipRef.current) return;
    try {
      const win = await api.requestWindow({ width: 240, height: 80 });
      win.document.body.style.margin = "0";
      win.addEventListener("pagehide", () => {
        pipRef.current = null;
        setPip(null);
      });
      pipRef.current = win;
      setPip(win);
      setOverlayBlocked(false);
    } catch {
      setOverlayBlocked(true);
    }
  }

  function grab(): Promise<Blob | null> {
    const video = videoRef.current;
    if (!video || !video.videoWidth) return Promise.resolve(null);

    const canvas = (canvasRef.current ??= document.createElement("canvas"));
    canvas.width = Math.min(CAPTURE_WIDTH, video.videoWidth);
    canvas.height = Math.round(
      (video.videoHeight * canvas.width) / video.videoWidth,
    );
    canvas
      .getContext("2d")
      ?.drawImage(video, 0, 0, canvas.width, canvas.height);

    return new Promise((resolve) =>
      canvas.toBlob(resolve, "image/jpeg", JPEG_QUALITY),
    );
  }

  async function tick() {
    const sessionId = sessionRef.current;
    if (!sessionId || busyRef.current) return;

    busyRef.current = true;
    try {
      const image = await grab();
      if (!image) return;

      const verdict = await sendLiveFrame(sessionId, image);
      if (sessionRef.current !== sessionId || verdict.dropped) return;

      failuresRef.current = 0;
      setError(null);
      setFrame(verdict);
      setSent((count) => count + 1);
      setElapsed((Date.now() - startedAtRef.current) / 1000);
    } catch (cause) {
      if (sessionRef.current !== sessionId) return;
      failuresRef.current += 1;
      setError(`Scoring failed: ${errorMessage(cause)}`);
      if (failuresRef.current >= MAX_FAILURES) {
        void stop(`Stopped after ${MAX_FAILURES} failed frames in a row.`);
      }
    } finally {
      busyRef.current = false;
    }
  }

  async function start() {
    setError(null);
    if (!navigator.mediaDevices?.getDisplayMedia) {
      setError("This browser cannot capture the screen.");
      return;
    }

    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getDisplayMedia({
        video: true,
        audio: false,
      });
    } catch (cause) {
      if (!(
        cause instanceof DOMException && cause.name === "NotAllowedError"
      )) {
        setError(`Could not start the share: ${errorMessage(cause)}`);
      }
      return;
    }

    setPhase("starting");
    const overlay = openOverlay();
    try {
      sessionRef.current = await startLive();
    } catch (cause) {
      stream.getTracks().forEach((track) => track.stop());
      await overlay;
      release();
      setPhase("idle");
      setError(errorMessage(cause));
      return;
    }

    streamRef.current = stream;
    stream.getVideoTracks()[0]?.addEventListener("ended", () => void stop());

    const video = videoRef.current;
    if (video) {
      video.srcObject = stream;
      await video.play().catch(() => {});
    }

    startedAtRef.current = Date.now();
    failuresRef.current = 0;
    setSummary(null);
    setFrame(null);
    setSent(0);
    setElapsed(0);
    setPhase("sharing");
    timerRef.current = window.setInterval(tick, INTERVAL_MS);
  }

  async function stop(reason?: string) {
    const sessionId = sessionRef.current;
    if (!sessionId) return;

    sessionRef.current = null;
    release();
    setPhase("saving");
    setError(reason ?? null);

    try {
      setSummary(await stopLive(sessionId));
      onSaved();
    } catch (cause) {
      if (cause instanceof ApiError && cause.status === 422) {
        setError(
          "No face showed up during the share, so there was nothing to save.",
        );
      } else {
        setError(`Could not save the session: ${errorMessage(cause)}`);
      }
    } finally {
      setPhase("idle");
    }
  }

  const sharing = phase === "sharing";
  const status = frame?.status === "WAITING" ? null : (frame?.status ?? null);
  const probability = frame?.smoothed ?? null;

  return (
    <div className="space-y-6">
      <Card>
        <div className="p-5">
          <h2 className="text-xl font-semibold">Check a screen share</h2>
          <p className="mt-1 text-sm text-muted">
            Share a window or tab with a video call or clip in it. Frames are
            scored twice a second while you share.
          </p>

          <div className="mt-4 flex flex-wrap items-center gap-3">
            {sharing ? (
              <Button variant="primary" onClick={() => void stop()}>
                Stop sharing
              </Button>
            ) : (
              <Button
                variant="primary"
                disabled={phase !== "idle"}
                onClick={() => void start()}
              >
                Share a screen
              </Button>
            )}
            {sharing && overlayBlocked ? (
              <Button onClick={() => void openOverlay()}>
                Pop out overlay
              </Button>
            ) : null}
            {phase === "starting" || phase === "saving" ? (
              <span className="flex items-center gap-2 text-sm text-muted">
                <span className="df-spinner" />
                {phase === "starting"
                  ? "Starting a session"
                  : "Saving the session"}
              </span>
            ) : null}
          </div>

          {error ? (
            <div className="mt-3">
              <ErrorNote>{error}</ErrorNote>
            </div>
          ) : null}

          <div className={cx("mt-5 space-y-4", !sharing && "hidden")}>
            <Verdict
              status={status}
              probability={probability}
              placeholder={
                frame ? "No face in view yet" : "Waiting for the first frame"
              }
            />
            {frame && !frame.face_found && status ? (
              <p className="text-sm text-muted">No face in the latest frame.</p>
            ) : null}
            <Timeline points={frame?.recent ?? []} />
            <p className="text-sm text-muted tabular-nums">
              {sent} frames sent, {frame?.frames_scored ?? 0} with a face,{" "}
              {seconds(elapsed)}
            </p>
            <video
              ref={videoRef}
              muted
              playsInline
              className="block w-full max-w-sm border border-line"
            />
          </div>
        </div>
      </Card>

      {summary ? <LiveSummary session={summary} title="Last share" /> : null}

      {pip
        ? createPortal(
            <div
              style={{
                padding: "10px 14px",
                fontFamily: "system-ui, sans-serif",
                color: status ? OVERLAY_COLOR[status] : "#9c8776",
              }}
            >
              {status && probability !== null ? (
                <>
                  <div style={{ fontSize: 18, fontWeight: 600 }}>
                    {STATUS_LABEL[status]}
                  </div>
                  <div style={{ fontSize: 14 }}>
                    {percent(probability)} fake
                  </div>
                </>
              ) : (
                <div style={{ fontSize: 16 }}>Waiting for a face</div>
              )}
            </div>,
            pip.document.body,
          )
        : null}
    </div>
  );
}
