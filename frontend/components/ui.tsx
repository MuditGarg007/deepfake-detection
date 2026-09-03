/** Shared primitives — one card, one button, one label, used everywhere. */

import type { ComponentProps, ReactNode } from "react";

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
      className={cx(
        "rounded-xl border border-line bg-background",
        "shadow-[0_1px_2px_rgba(0,0,0,0.04)]",
        className,
      )}
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
    <header className="flex items-center justify-between gap-3 border-b border-line px-5 py-3.5">
      <div className="min-w-0">
        <h2 className="text-[13px] font-medium tracking-tight">{title}</h2>
        {hint ? <p className="mt-0.5 text-xs text-muted">{hint}</p> : null}
      </div>
      {action}
    </header>
  );
}

/** Small uppercase caption used above every metric. */
export function Label({ children }: { children: ReactNode }) {
  return (
    <span className="text-[10px] font-medium uppercase tracking-[0.09em] text-faint">
      {children}
    </span>
  );
}

type ButtonProps = ComponentProps<"button"> & {
  variant?: "primary" | "secondary" | "ghost";
  size?: "sm" | "md";
};

const VARIANTS: Record<NonNullable<ButtonProps["variant"]>, string> = {
  primary:
    "bg-foreground text-background border border-foreground hover:bg-[#333] hover:border-[#333] disabled:bg-[#c9c9c9] disabled:border-[#c9c9c9]",
  secondary:
    "bg-background text-foreground border border-line-strong hover:bg-sunken disabled:text-faint",
  ghost:
    "bg-transparent text-muted border border-transparent hover:bg-sunken hover:text-foreground",
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
        "inline-flex items-center justify-center gap-2 rounded-lg font-medium",
        "transition-colors duration-150 disabled:cursor-not-allowed",
        size === "sm" ? "h-8 px-3 text-xs" : "h-10 px-4 text-sm",
        VARIANTS[variant],
        className,
      )}
    />
  );
}

/** Inline error strip — a hairline rule plus the backend's `detail` text. */
export function ErrorNote({ children }: { children: ReactNode }) {
  return (
    <p
      role="alert"
      className="flex gap-2 rounded-lg border border-line-strong bg-sunken px-3 py-2.5 text-xs leading-relaxed text-foreground"
    >
      <span aria-hidden="true" className="font-mono">
        !
      </span>
      <span>{children}</span>
    </p>
  );
}
