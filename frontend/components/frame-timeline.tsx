"use client";

/**
 * Per-frame fake probability over the clip — one series, so the title names it
 * and no legend is needed. Reference lines mark the two risk thresholds from
 * `backend/config.py`, and the shaded band is the suspicious region the backend
 * derived. Hover gives a crosshair + tooltip; a table view holds the same numbers.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import type { FrameScore } from "@/lib/types";
import { FRAME_THRESHOLD, RISK_SUSPICIOUS, percent, seconds } from "@/lib/format";
import { Button } from "./ui";

const HEIGHT = 236;
const PAD = { top: 16, right: 16, bottom: 30, left: 40 };
const MIN_WIDTH = 320;

function useElementWidth<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [width, setWidth] = useState(720);

  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => {
      setWidth(Math.max(MIN_WIDTH, entry.contentRect.width));
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  return [ref, width] as const;
}

/** ~5 evenly spaced time ticks, snapped to the frames that actually exist. */
function timeTicks(scores: FrameScore[], count = 5): FrameScore[] {
  if (scores.length <= count) return scores;
  const step = (scores.length - 1) / (count - 1);
  return Array.from({ length: count }, (_, i) => scores[Math.round(i * step)]);
}

export function FrameTimeline({
  scores,
  suspiciousStart,
  suspiciousEnd,
}: {
  scores: FrameScore[];
  suspiciousStart: number | null;
  suspiciousEnd: number | null;
}) {
  const [wrapRef, width] = useElementWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const [showTable, setShowTable] = useState(false);

  const plotWidth = Math.max(1, width - PAD.left - PAD.right);
  const plotHeight = HEIGHT - PAD.top - PAD.bottom;

  const domain = useMemo(() => {
    const first = scores[0]?.timestamp ?? 0;
    const last = scores.at(-1)?.timestamp ?? 1;
    return { first, span: Math.max(last - first, 0.001) };
  }, [scores]);

  const x = useCallback(
    (timestamp: number) =>
      PAD.left + ((timestamp - domain.first) / domain.span) * plotWidth,
    [domain, plotWidth],
  );
  const y = useCallback(
    (probability: number) => PAD.top + (1 - probability) * plotHeight,
    [plotHeight],
  );

  const { line, area } = useMemo(() => {
    if (scores.length === 0) return { line: "", area: "" };
    const points = scores.map((s) => `${x(s.timestamp).toFixed(2)},${y(s.fake_probability).toFixed(2)}`);
    const path = `M${points.join("L")}`;
    const baseline = y(0).toFixed(2);
    return {
      line: path,
      area: `${path}L${x(scores.at(-1)!.timestamp).toFixed(2)},${baseline}L${x(scores[0].timestamp).toFixed(2)},${baseline}Z`,
    };
  }, [scores, x, y]);

  const peak = useMemo(
    () =>
      scores.reduce(
        (best, s) => (s.fake_probability > best.fake_probability ? s : best),
        scores[0],
      ),
    [scores],
  );

  const onMove = useCallback(
    (event: React.PointerEvent<SVGSVGElement>) => {
      const bounds = event.currentTarget.getBoundingClientRect();
      const offset = event.clientX - bounds.left;
      let nearest = 0;
      let best = Infinity;
      scores.forEach((score, index) => {
        const distance = Math.abs(x(score.timestamp) - offset);
        if (distance < best) {
          best = distance;
          nearest = index;
        }
      });
      setHover(nearest);
    },
    [scores, x],
  );

  if (scores.length === 0) {
    return (
      <p className="px-5 py-8 text-center text-xs text-muted">
        No frame scores were recorded for this video.
      </p>
    );
  }

  const active = hover === null ? null : scores[hover];
  const hasBand =
    suspiciousStart !== null && suspiciousEnd !== null && suspiciousEnd > suspiciousStart;
  const tooltipLeft = active ? Math.min(Math.max(x(active.timestamp), 70), width - 70) : 0;

  return (
    <div className="px-3 pb-3">
      {/* The plot never shrinks past MIN_WIDTH; below that it scrolls in place. */}
      <div ref={wrapRef} className="relative overflow-x-auto">
        <svg
          width={width}
          height={HEIGHT}
          role="img"
          aria-label={`Fake probability per sampled frame, ${scores.length} frames, peaking at ${percent(peak.fake_probability)}`}
          className="block touch-none select-none"
          onPointerMove={onMove}
          onPointerLeave={() => setHover(null)}
        >
          {/* Gridlines — recessive, behind everything. */}
          {[0, 0.25, 0.5, 0.75, 1].map((tick) => (
            <g key={tick}>
              <line
                x1={PAD.left}
                x2={PAD.left + plotWidth}
                y1={y(tick)}
                y2={y(tick)}
                stroke="var(--line)"
                strokeWidth={1}
              />
              <text
                x={PAD.left - 8}
                y={y(tick)}
                textAnchor="end"
                dominantBaseline="middle"
                className="fill-[var(--faint)] font-mono text-[9px]"
              >
                {tick * 100}
              </text>
            </g>
          ))}

          {/* The suspicious window the backend reported. */}
          {hasBand ? (
            <g>
              <rect
                x={x(suspiciousStart)}
                y={PAD.top}
                width={Math.max(2, x(suspiciousEnd) - x(suspiciousStart))}
                height={plotHeight}
                fill="rgba(10,10,10,0.06)"
              />
              <line
                x1={x(suspiciousStart)}
                x2={x(suspiciousStart)}
                y1={PAD.top}
                y2={PAD.top + plotHeight}
                stroke="var(--line-strong)"
                strokeWidth={1}
              />
              <line
                x1={x(suspiciousEnd)}
                x2={x(suspiciousEnd)}
                y1={PAD.top}
                y2={PAD.top + plotHeight}
                stroke="var(--line-strong)"
                strokeWidth={1}
              />
            </g>
          ) : null}

          {/* Risk thresholds. */}
          {[
            { value: RISK_SUSPICIOUS, label: "suspicious 0.40" },
            { value: FRAME_THRESHOLD, label: "high risk 0.70" },
          ].map((reference) => (
            <g key={reference.label}>
              <line
                x1={PAD.left}
                x2={PAD.left + plotWidth}
                y1={y(reference.value)}
                y2={y(reference.value)}
                stroke="var(--line-strong)"
                strokeWidth={1}
                strokeDasharray="3 3"
              />
              {/* Halo so the label stays legible where the series runs under it. */}
              <text
                x={PAD.left + plotWidth}
                y={y(reference.value) - 5}
                textAnchor="end"
                stroke="var(--background)"
                strokeWidth={3}
                paintOrder="stroke"
                className="fill-[var(--muted)] text-[9px]"
              >
                {reference.label}
              </text>
            </g>
          ))}

          <defs>
            <linearGradient id="df-area" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="rgba(10,10,10,0.16)" />
              <stop offset="100%" stopColor="rgba(10,10,10,0)" />
            </linearGradient>
          </defs>
          <path d={area} fill="url(#df-area)" />
          <path
            d={line}
            fill="none"
            stroke="var(--foreground)"
            strokeWidth={2}
            strokeLinejoin="round"
            strokeLinecap="round"
          />

          {/* Peak frame, direct-labelled — the one point worth naming. */}
          <circle
            cx={x(peak.timestamp)}
            cy={y(peak.fake_probability)}
            r={4}
            fill="var(--background)"
            stroke="var(--foreground)"
            strokeWidth={2}
          />

          {/* Baseline + time axis. */}
          <line
            x1={PAD.left}
            x2={PAD.left + plotWidth}
            y1={y(0)}
            y2={y(0)}
            stroke="var(--line-strong)"
            strokeWidth={1}
          />
          {timeTicks(scores).map((score) => (
            <text
              key={score.timestamp}
              x={x(score.timestamp)}
              y={HEIGHT - 10}
              textAnchor="middle"
              className="fill-[var(--faint)] font-mono text-[9px]"
            >
              {seconds(score.timestamp)}
            </text>
          ))}

          {/* Hover crosshair. */}
          {active ? (
            <g>
              <line
                x1={x(active.timestamp)}
                x2={x(active.timestamp)}
                y1={PAD.top}
                y2={PAD.top + plotHeight}
                stroke="var(--foreground)"
                strokeWidth={1}
                strokeDasharray="2 2"
              />
              <circle
                cx={x(active.timestamp)}
                cy={y(active.fake_probability)}
                r={5}
                fill="var(--foreground)"
                stroke="var(--background)"
                strokeWidth={2}
              />
            </g>
          ) : null}
        </svg>

        {active ? (
          <div
            className="pointer-events-none absolute top-1 -translate-x-1/2 rounded-lg border border-line bg-background px-2.5 py-1.5 text-[11px] shadow-[0_2px_8px_rgba(0,0,0,0.08)]"
            style={{ left: tooltipLeft }}
          >
            <span className="font-mono text-muted">{seconds(active.timestamp)}</span>
            <span className="mx-1.5 text-line-strong">|</span>
            <span className="font-medium tabular-nums">
              {percent(active.fake_probability)} fake
            </span>
          </div>
        ) : null}
      </div>

      <div className="mt-1 flex items-center justify-between gap-3 px-2">
        <p className="text-[11px] text-faint">
          {scores.length} frames scored · peak {percent(peak.fake_probability)} at{" "}
          {seconds(peak.timestamp)}
        </p>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setShowTable((open) => !open)}
          aria-expanded={showTable}
        >
          {showTable ? "Hide data" : "Show data"}
        </Button>
      </div>

      {showTable ? (
        <div className="mt-2 max-h-56 overflow-y-auto rounded-lg border border-line">
          <table className="w-full text-left text-xs">
            <thead className="sticky top-0 bg-sunken text-[10px] uppercase tracking-[0.09em] text-faint">
              <tr>
                <th scope="col" className="px-3 py-2 font-medium">
                  Timestamp
                </th>
                <th scope="col" className="px-3 py-2 text-right font-medium">
                  Fake probability
                </th>
              </tr>
            </thead>
            <tbody className="font-mono tabular-nums">
              {scores.map((score) => (
                <tr key={score.timestamp} className="border-t border-line">
                  <td className="px-3 py-1.5 text-muted">{seconds(score.timestamp)}</td>
                  <td className="px-3 py-1.5 text-right">
                    {score.fake_probability.toFixed(4)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
    </div>
  );
}
