"use client";

/**
 * Result screen — the verdict, the aggregate score against the two risk
 * thresholds, the suspicious window, and the per-frame timeline.
 */

import type { Analysis } from "@/lib/types";
import {
  RISK_HIGH,
  RISK_SUSPICIOUS,
  STATUS_BLURB,
  absoluteTime,
  percent,
  seconds,
} from "@/lib/format";
import { FrameTimeline } from "./frame-timeline";
import { StatusBadge } from "./status-badge";
import { VideoPlayer } from "./video-player";
import { ArrowIcon, ClockIcon } from "./icons";
import { Button, Card, CardHeader, Label } from "./ui";

/** The aggregate score on the 0–1 axis, with both thresholds marked. */
function ScoreScale({ value }: { value: number }) {
  return (
    <div className="mt-4">
      <div className="relative h-2 rounded-full bg-sunken">
        <div
          className="absolute inset-y-0 left-0 rounded-full bg-foreground"
          style={{ width: `${Math.max(1.5, value * 100)}%` }}
        />
        {[RISK_SUSPICIOUS, RISK_HIGH].map((threshold) => (
          <span
            key={threshold}
            className="absolute -top-1 h-4 w-px bg-line-strong"
            style={{ left: `${threshold * 100}%` }}
          />
        ))}
      </div>
      <div className="relative mt-1.5 h-4 text-[10px] text-faint">
        <span className="absolute left-0">0 · real</span>
        <span className="absolute -translate-x-1/2" style={{ left: "40%" }}>
          0.40
        </span>
        <span className="absolute -translate-x-1/2" style={{ left: "70%" }}>
          0.70
        </span>
        <span className="absolute right-0">1 · fake</span>
      </div>
    </div>
  );
}

function Metric({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="px-5 py-4">
      <Label>{label}</Label>
      <p className="mt-1 font-mono text-[15px] tabular-nums">{value}</p>
      {hint ? <p className="mt-0.5 text-[11px] text-faint">{hint}</p> : null}
    </div>
  );
}

export function ResultPanel({
  analysis,
  onReset,
}: {
  analysis: Analysis;
  onReset: () => void;
}) {
  const scores = analysis.frame_scores;
  const peak = scores.reduce((best, s) => Math.max(best, s.fake_probability), 0);
  const flagged = scores.filter((s) => s.fake_probability >= RISK_HIGH).length;
  const hasWindow =
    analysis.suspicious_start !== null && analysis.suspicious_end !== null;

  return (
    <div className="df-enter space-y-4">
      <Card>
        <div className="flex flex-wrap items-start justify-between gap-4 px-5 pt-5">
          <div className="min-w-0">
            <StatusBadge status={analysis.status} />
            <h2 className="mt-3 truncate text-lg font-semibold tracking-tight">
              {analysis.filename}
            </h2>
            <p className="mt-1 font-mono text-[11px] text-faint">
              analysis #{analysis.id} · {absoluteTime(analysis.created_at)}
            </p>
          </div>
          <div className="text-right">
            <Label>Fake probability</Label>
            <p className="font-mono text-4xl font-medium tracking-tight tabular-nums">
              {percent(analysis.fake_probability)}
            </p>
          </div>
        </div>

        <div className="px-5 pb-5">
          <ScoreScale value={analysis.fake_probability} />
          <p className="mt-4 border-t border-line pt-4 text-[13px] leading-relaxed text-muted">
            {STATUS_BLURB[analysis.status]}
          </p>
        </div>
      </Card>

      <VideoPlayer analysis={analysis} />

      <Card>
        <CardHeader
          title="Suspicious region"
          hint="Longest run of frames at or above 0.70"
        />
        <div className="px-5 py-4">
          {hasWindow ? (
            <div className="flex flex-wrap items-center gap-3">
              <span className="inline-flex items-center gap-2 rounded-lg border border-foreground bg-foreground px-3 py-2 font-mono text-sm text-background tabular-nums">
                <ClockIcon className="size-3.5" />
                {seconds(analysis.suspicious_start!)}
                <ArrowIcon className="size-3.5" />
                {seconds(analysis.suspicious_end!)}
              </span>
              <span className="text-[13px] text-muted">
                {seconds(analysis.suspicious_end! - analysis.suspicious_start!)} of
                sustained manipulation signal.
              </span>
            </div>
          ) : (
            <p className="text-[13px] text-muted">
              No run of three or more consecutive frames crossed the threshold, so no
              window was flagged.
            </p>
          )}
        </div>
      </Card>

      <Card>
        <CardHeader
          title="Per-frame probability"
          hint="Every sampled frame that contained a usable face"
        />
        <FrameTimeline
          scores={scores}
          suspiciousStart={analysis.suspicious_start}
          suspiciousEnd={analysis.suspicious_end}
        />
        <div className="grid grid-cols-2 divide-x divide-line border-t border-line sm:grid-cols-4">
          <Metric label="Frames scored" value={String(scores.length)} />
          <Metric label="Mean" value={analysis.fake_probability.toFixed(4)} hint="the verdict score" />
          <Metric label="Peak frame" value={peak.toFixed(4)} />
          <Metric
            label="Above 0.70"
            value={`${flagged}`}
            hint={scores.length ? `${percent(flagged / scores.length, 0)} of frames` : undefined}
          />
        </div>
      </Card>

      <Button variant="secondary" className="w-full" onClick={onReset}>
        Analyze another video
      </Button>
    </div>
  );
}
