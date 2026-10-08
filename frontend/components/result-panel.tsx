"use client";

import { useState, type ReactNode } from "react";

import { errorMessage, frameUrl, rerunAnalysis, sendFeedback } from "@/lib/api";
import { percent, seconds } from "@/lib/format";
import type { Analysis, Feedback } from "@/lib/types";
import { Timeline } from "./timeline";
import { Button, Card, CardHeader, ErrorNote, Stats, Verdict } from "./ui";

const FEEDBACK_LABEL: Record<Feedback, string> = {
  REAL: "real",
  FAKE: "fake",
};

function contradicts(analysis: Analysis, label: Feedback): boolean {
  if (label === "FAKE") return analysis.status === "REAL";
  return analysis.status === "HIGH_RISK";
}

export function ResultPanel({
  analysis,
  onUpdate,
  action,
}: {
  analysis: Analysis;
  onUpdate: (updated: Analysis) => void;
  action?: ReactNode;
}) {
  const [frameFailedFor, setFrameFailedFor] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const [rechecking, setRechecking] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const showFrame = analysis.has_video && frameFailedFor !== analysis.id;
  const scores = analysis.frame_scores;
  const peak = scores.reduce(
    (best, score) => Math.max(best, score.fake_probability),
    0,
  );
  const suspicious =
    analysis.suspicious_start !== null && analysis.suspicious_end !== null
      ? ([analysis.suspicious_start, analysis.suspicious_end] as [
          number,
          number,
        ])
      : null;

  async function submit(label: Feedback | null) {
    setBusy(true);
    setError(null);
    try {
      let updated = await sendFeedback(analysis.id, label);
      if (label && updated.has_video && contradicts(updated, label)) {
        setRechecking(true);
        updated = await rerunAnalysis(updated.id);
      }
      onUpdate(updated);
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setRechecking(false);
      setBusy(false);
    }
  }

  return (
    <Card>
      <CardHeader
        title={<span className="block truncate">{analysis.filename}</span>}
        hint={`${scores.length} frames with a face`}
        action={action}
      />
      <div className="space-y-4 p-5">
        {showFrame ? (
          <img
            key={analysis.id}
            src={frameUrl(analysis.id)}
            alt="The most suspicious frame of this video"
            onError={() => setFrameFailedFor(analysis.id)}
            className="block max-h-80 w-full border border-line bg-surface object-contain"
          />
        ) : null}

        <Verdict
          status={analysis.status}
          probability={analysis.fake_probability}
        />

        <Stats
          items={[
            ["Average", percent(analysis.fake_probability)],
            ["Peak", percent(peak)],
            [
              "Suspicious part",
              suspicious
                ? `${seconds(suspicious[0])} to ${seconds(suspicious[1])}`
                : "None",
            ],
          ]}
        />

        <Timeline points={scores} highlight={suspicious} />

        {analysis.user_feedback ? (
          <p>
            You said this video is {FEEDBACK_LABEL[analysis.user_feedback]}.{" "}
            <button
              type="button"
              disabled={busy}
              onClick={() => submit(null)}
              className="text-brown underline disabled:no-underline"
            >
              Undo
            </button>
          </p>
        ) : (
          <div className="flex flex-wrap items-center gap-3">
            <span>Do you know if this video is real or fake?</span>
            <Button size="sm" disabled={busy} onClick={() => submit("REAL")}>
              Real
            </Button>
            <Button size="sm" disabled={busy} onClick={() => submit("FAKE")}>
              Fake
            </Button>
          </div>
        )}

        {rechecking ? (
          <p className="flex items-center gap-2 text-muted">
            <span className="df-spinner" />
            That does not match the result, checking the video again
          </p>
        ) : null}

        {error ? <ErrorNote>{error}</ErrorNote> : null}
      </div>
    </Card>
  );
}
