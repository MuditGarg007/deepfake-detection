/** Formatting helpers shared by the screens. */

import type { RiskStatus } from "./types";

/** `0.9733` -> `"97.3%"`. */
export function percent(value: number, digits = 1): string {
  return `${(value * 100).toFixed(digits)}%`;
}

/** `10.2` -> `"10.2s"`; anything past a minute gets `m:ss`. */
export function seconds(value: number): string {
  if (value < 60) return `${value.toFixed(1)}s`;
  const minutes = Math.floor(value / 60);
  const rest = value - minutes * 60;
  return `${minutes}:${rest.toFixed(1).padStart(4, "0")}`;
}

export function fileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/**
 * The SQLite fallback stores naive timestamps, so a value with no zone marker is
 * read as UTC, matching what Postgres returns for the same row.
 */
export function parseTimestamp(value: string): Date {
  const hasZone = /(Z|[+-]\d{2}:?\d{2})$/.test(value);
  return new Date(hasZone ? value : `${value}Z`);
}

const RELATIVE = new Intl.RelativeTimeFormat("en", { numeric: "auto" });
const UNITS: [Intl.RelativeTimeFormatUnit, number][] = [
  ["second", 60],
  ["minute", 60],
  ["hour", 24],
  ["day", 7],
];

/** `"just now"`, `"4 minutes ago"`, or an absolute date past a week. */
export function relativeTime(value: string): string {
  const date = parseTimestamp(value);
  let delta = (date.getTime() - Date.now()) / 1000;
  if (Math.abs(delta) < 45) return "just now";

  for (const [unit, span] of UNITS) {
    if (Math.abs(delta) < span) return RELATIVE.format(Math.round(delta), unit);
    delta /= span;
  }
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

export function absoluteTime(value: string): string {
  return parseTimestamp(value).toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  });
}

export const STATUS_LABEL: Record<RiskStatus, string> = {
  REAL: "Likely authentic",
  SUSPICIOUS: "Suspicious",
  HIGH_RISK: "High risk",
};

export const STATUS_BLURB: Record<RiskStatus, string> = {
  REAL: "Frame scores stayed below the suspicion threshold across the clip.",
  SUSPICIOUS: "Some manipulation signals are present, but the result is not conclusive. Review the video manually.",
  HIGH_RISK: "Strong and sustained manipulation signal. Treat this video as fabricated.",
};

/** Risk thresholds mirrored from `backend/config.py`. */
export const RISK_SUSPICIOUS = 0.4;
export const RISK_HIGH = 0.7;
export const FRAME_THRESHOLD = 0.7;
