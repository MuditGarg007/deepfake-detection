"use client";

/**
 * The clip the verdict was drawn from, played back from
 * `GET /analysis/{id}/video`. Native controls carry the transport (keyboard,
 * volume, fullscreen and picture-in-picture come free) and the one thing added
 * on top is a jump straight to the suspicious window the backend flagged.
 */

import { useCallback, useRef, useState } from "react";

import { videoUrl } from "@/lib/api";
import type { Analysis } from "@/lib/types";
import { seconds } from "@/lib/format";
import { Button, Card, CardHeader, ErrorNote } from "./ui";

export function VideoPlayer({ analysis }: { analysis: Analysis }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [failed, setFailed] = useState(false);

  const start = analysis.suspicious_start;
  const end = analysis.suspicious_end;
  const hasWindow = start !== null && end !== null;

  // Seek to the window's first frame so the flagged stretch is already on screen
  // when playback starts.
  const playSuspicious = useCallback(() => {
    const video = videoRef.current;
    if (!video || start === null) return;
    video.currentTime = start;
    void video.play().catch(() => {
      // Autoplay can be refused; the seek still landed, so leave it paused.
    });
  }, [start]);

  if (!analysis.has_video) {
    return (
      <Card>
        <CardHeader title="Source video" />
        <p className="px-5 py-4 text-base text-muted">
          The source file for this analysis is no longer on the server, so there is
          nothing to play back. The scores below are unaffected.
        </p>
      </Card>
    );
  }

  return (
    <Card className="overflow-hidden">
      <CardHeader
        title="Source video"
        hint="The clip these scores were taken from"
        action={
          hasWindow ? (
            <Button size="sm" onClick={playSuspicious} className="shrink-0">
              Jump to {seconds(start)}
            </Button>
          ) : undefined
        }
      />

      <div className="bg-foreground">
        <video
          // Remount on a different analysis so the element reloads its source
          // instead of holding the previous clip's buffered state.
          key={analysis.id}
          ref={videoRef}
          src={videoUrl(analysis.id)}
          controls
          preload="metadata"
          playsInline
          onError={() => setFailed(true)}
          className="mx-auto block max-h-[420px] w-full"
        >
          Your browser cannot play this video.
        </video>
      </div>

      {failed ? (
        <div className="px-5 py-4">
          <ErrorNote>
            The browser could not decode this file. Detection still ran on it. Some
            containers the pipeline accepts (.avi in particular) have no in-browser
            playback support.
          </ErrorNote>
        </div>
      ) : null}

      {hasWindow ? (
        <p className="border-t border-line px-5 py-3 text-sm text-muted">
          Suspicious window: {seconds(start)} to {seconds(end)}
        </p>
      ) : null}
    </Card>
  );
}
