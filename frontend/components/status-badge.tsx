/**
 * Risk state in a monochrome palette: fill weight ranks the three states, and a
 * glyph plus the written label carries the meaning so shade is never the only cue.
 */

import type { RiskStatus } from "@/lib/types";
import { STATUS_LABEL } from "@/lib/format";
import { AlertIcon, CheckIcon, WarningIcon } from "./icons";
import { cx } from "./ui";

const STYLES: Record<RiskStatus, string> = {
  REAL: "border-line-strong bg-background text-foreground",
  SUSPICIOUS: "border-line-strong bg-sunken text-foreground",
  HIGH_RISK: "border-foreground bg-foreground text-background",
};

const ICONS: Record<RiskStatus, typeof CheckIcon> = {
  REAL: CheckIcon,
  SUSPICIOUS: WarningIcon,
  HIGH_RISK: AlertIcon,
};

export function StatusBadge({
  status,
  size = "md",
}: {
  status: RiskStatus;
  size?: "sm" | "md";
}) {
  const Icon = ICONS[status];
  return (
    <span
      className={cx(
        "inline-flex items-center gap-1.5 rounded-full border font-medium",
        size === "sm" ? "px-2 py-0.5 text-[11px]" : "px-2.5 py-1 text-xs",
        STYLES[status],
      )}
    >
      <Icon className={size === "sm" ? "size-3" : "size-3.5"} />
      {STATUS_LABEL[status]}
    </span>
  );
}

/** Icon-only variant for dense rows; the label rides along for screen readers. */
export function StatusGlyph({ status }: { status: RiskStatus }) {
  const Icon = ICONS[status];
  return (
    <span
      title={STATUS_LABEL[status]}
      className={cx(
        "inline-flex size-6 shrink-0 items-center justify-center rounded-full border",
        STYLES[status],
      )}
    >
      <Icon className="size-3.5" />
      <span className="sr-only">{STATUS_LABEL[status]}</span>
    </span>
  );
}
