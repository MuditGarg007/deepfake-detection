"use client";

/** Result screen: the flagged frame, the verdict, the score and feedback. */

import { useState } from "react";

import { frameUrl, rerunAnalysis, sendFeedback } from "@/lib/api";
import type { Analysis, Feedback } from "@/lib/types";
import { STATUS_LABEL, percent } from "@/lib/format";
import { Button, Card, ErrorNote } from "./ui";

const FEEDBACK_LABEL: Record<Feedback, string> = {
  REAL: "real",
  FAKE: "fake",
};

export function ResultPanel({
  analysis,
  onUpdate,
  onReset,
}: {
  analysis: Analysis;
  onUpdate: (updated: Analysis) => void;
  onReset: () => void;
}) {
  // The frame is read back out of the source video, so it is missing whenever
  // that file is not on the server any more. Tracked by id so that switching to
  // another analysis does not inherit the previous one's failure.
  const [frameFailedFor, setFrameFailedFor] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const showFrame = analysis.has_video && frameFailedFor !== analysis.id;

  async function run(work: () => Promise<Analysis>) {
    setBusy(true);
    setError(null);
    try {
      onUpdate(await work());
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Something went wrong");
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card>
      <div className="p-5 text-base">
        {showFrame ? (
          // next/image cannot optimise a one-off JPEG served from the backend
          // origin, and would need a remotePatterns entry to load it at all.
          // eslint-disable-next-line @next/next/no-img-element
          <img
            // Reload when a different analysis is shown instead of keeping the
            // previous frame on screen while the new one loads.
            key={analysis.id}
            src={frameUrl(analysis.id)}
            alt="The most suspicious frame of this video"
            onError={() => setFrameFailedFor(analysis.id)}
            className="mb-4 block max-h-80 w-full rounded border border-line object-contain"
          />
        ) : null}

        <p>Fake probability: {percent(analysis.fake_probability)}</p>
        <p className="mt-1">Verdict: {STATUS_LABEL[analysis.status]}</p>

        {analysis.user_feedback ? (
          <p className="mt-4">
            You said this video is really{" "}
            {FEEDBACK_LABEL[analysis.user_feedback]}.{" "}
            <button
              type="button"
              disabled={busy}
              onClick={() => run(() => sendFeedback(analysis.id, null))}
              className="text-brown underline disabled:no-underline"
            >
              Undo
            </button>
          </p>
        ) : (
          <div className="mt-4 flex flex-wrap items-center gap-3">
            <span>Was this video real or fake?</span>
            <Button
              disabled={busy}
              onClick={() => run(() => sendFeedback(analysis.id, "REAL"))}
            >
              Real
            </Button>
            <Button
              disabled={busy}
              onClick={() => run(() => sendFeedback(analysis.id, "FAKE"))}
            >
              Fake
            </Button>
          </div>
        )}

        {error ? (
          <div className="mt-4">
            <ErrorNote>{error}</ErrorNote>
          </div>
        ) : null}

        <div className="mt-6 flex flex-wrap gap-3">
          <Button variant="secondary" onClick={onReset}>
            Analyze another video
          </Button>
          <Button
            variant="secondary"
            disabled={busy || !analysis.has_video}
            onClick={() => run(() => rerunAnalysis(analysis.id))}
          >
            {busy ? "Working..." : "Run again"}
          </Button>
        </div>
      </div>
    </Card>
  );
}
