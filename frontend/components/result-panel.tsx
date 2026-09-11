"use client";

/** Result screen: the flagged frame, the verdict and the aggregate score. */

import { useState } from "react";

import { frameUrl } from "@/lib/api";
import type { Analysis } from "@/lib/types";
import { STATUS_LABEL, percent } from "@/lib/format";
import { Button, Card } from "./ui";

export function ResultPanel({
  analysis,
  onReset,
}: {
  analysis: Analysis;
  onReset: () => void;
}) {
  // The frame is read back out of the source video, so it is missing whenever
  // that file is not on the server any more.
  const [frameFailed, setFrameFailed] = useState(false);

  return (
    <Card>
      <div className="p-5 text-base">
        {analysis.has_video && !frameFailed ? (
          // next/image cannot optimise a one-off JPEG served from the backend
          // origin, and would need a remotePatterns entry to load it at all.
          // eslint-disable-next-line @next/next/no-img-element
          <img
            // Reload when a different analysis is shown instead of keeping the
            // previous frame on screen while the new one loads.
            key={analysis.id}
            src={frameUrl(analysis.id)}
            alt="The most suspicious frame of this video"
            onError={() => setFrameFailed(true)}
            className="mb-4 block max-h-80 w-full rounded border border-line object-contain"
          />
        ) : null}

        <p>Fake probability: {percent(analysis.fake_probability)}</p>
        <p className="mt-1">Verdict: {STATUS_LABEL[analysis.status]}</p>

        <div className="mt-6">
          <Button variant="secondary" onClick={onReset}>
            Analyze another video
          </Button>
        </div>
      </div>
    </Card>
  );
}
