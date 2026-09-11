/** Risk state shown as a plain text label. */

import type { RiskStatus } from "@/lib/types";
import { STATUS_LABEL } from "@/lib/format";
import { cx } from "./ui";

const STYLES: Record<RiskStatus, string> = {
  REAL: "border-line-strong bg-surface text-muted",
  SUSPICIOUS: "border-brown bg-surface text-brown",
  HIGH_RISK: "border-brown bg-brown text-white",
};

export function StatusBadge({
  status,
  size = "md",
}: {
  status: RiskStatus;
  size?: "sm" | "md";
}) {
  return (
    <span
      className={cx(
        "inline-block rounded border font-medium",
        size === "sm" ? "px-2 py-0.5 text-xs" : "px-2.5 py-1 text-sm",
        STYLES[status],
      )}
    >
      {STATUS_LABEL[status]}
    </span>
  );
}
