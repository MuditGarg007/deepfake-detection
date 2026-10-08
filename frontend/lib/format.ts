import type { RiskStatus } from "./types";

export function percent(value: number, digits = 1): string {
  return `${(value * 100).toFixed(digits)}%`;
}

export function fileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function seconds(value: number): string {
  if (value < 60) return `${Math.round(value)}s`;
  const minutes = Math.floor(value / 60);
  const rest = Math.round(value % 60);
  return `${minutes}m ${rest.toString().padStart(2, "0")}s`;
}

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

export const STATUS_LABEL: Record<RiskStatus, string> = {
  REAL: "Likely authentic",
  SUSPICIOUS: "Suspicious",
  HIGH_RISK: "High risk",
};
