import type { ComponentProps, ReactNode } from "react";

import { STATUS_LABEL, percent } from "@/lib/format";
import type { RiskStatus } from "@/lib/types";

export function cx(...parts: (string | false | null | undefined)[]): string {
  return parts.filter(Boolean).join(" ");
}

export function Card({
  className,
  children,
  ...props
}: ComponentProps<"section">) {
  return (
    <section
      {...props}
      className={cx("border border-line bg-background", className)}
    >
      {children}
    </section>
  );
}

export function CardHeader({
  title,
  hint,
  action,
}: {
  title: ReactNode;
  hint?: ReactNode;
  action?: ReactNode;
}) {
  return (
    <header className="flex items-center justify-between gap-3 border-b border-line bg-surface px-5 py-3">
      <div className="min-w-0">
        <h2 className="text-base font-semibold">{title}</h2>
        {hint ? <p className="mt-0.5 text-sm text-muted">{hint}</p> : null}
      </div>
      {action}
    </header>
  );
}

export function Label({ children }: { children: ReactNode }) {
  return <span className="text-sm text-muted">{children}</span>;
}

type ButtonProps = ComponentProps<"button"> & {
  variant?: "primary" | "secondary" | "ghost";
  size?: "sm" | "md";
};

const VARIANTS: Record<NonNullable<ButtonProps["variant"]>, string> = {
  primary:
    "bg-brown text-white border border-brown hover:bg-brown-dark hover:border-brown-dark disabled:bg-line-strong disabled:border-line-strong",
  secondary:
    "bg-background text-foreground border border-line-strong hover:bg-surface disabled:text-faint",
  ghost:
    "bg-transparent text-muted border border-transparent hover:bg-surface hover:text-brown",
};

export function Button({
  variant = "secondary",
  size = "md",
  className,
  ...props
}: ButtonProps) {
  return (
    <button
      {...props}
      className={cx(
        "inline-flex items-center justify-center gap-2 font-medium",
        "transition-colors disabled:cursor-not-allowed",
        size === "sm" ? "h-8 px-3 text-sm" : "h-10 px-4 text-base",
        VARIANTS[variant],
        className,
      )}
    />
  );
}

export function ErrorNote({ children }: { children: ReactNode }) {
  return (
    <p
      role="alert"
      className="border border-line-strong bg-surface px-3 py-2 text-sm leading-relaxed text-foreground"
    >
      {children}
    </p>
  );
}

export const TONE: Record<RiskStatus, string> = {
  REAL: "text-real",
  SUSPICIOUS: "text-suspicious",
  HIGH_RISK: "text-high",
};

export function Verdict({
  status,
  probability,
  placeholder = "Waiting for a face",
}: {
  status: RiskStatus | null;
  probability: number | null;
  placeholder?: string;
}) {
  if (!status || probability === null) {
    return <p className="text-2xl font-semibold text-faint">{placeholder}</p>;
  }
  return (
    <p className={cx("flex flex-wrap items-baseline gap-x-3", TONE[status])}>
      <span className="text-2xl font-semibold">{STATUS_LABEL[status]}</span>
      <span className="text-base tabular-nums">
        {percent(probability)} fake
      </span>
    </p>
  );
}

export function Stats({ items }: { items: [string, ReactNode][] }) {
  return (
    <dl className="grid grid-cols-3 gap-px border border-line bg-line">
      {items.map(([label, value]) => (
        <div key={label} className="bg-surface px-3 py-2">
          <dt className="text-xs text-muted">{label}</dt>
          <dd className="text-base font-medium tabular-nums">{value}</dd>
        </div>
      ))}
    </dl>
  );
}
