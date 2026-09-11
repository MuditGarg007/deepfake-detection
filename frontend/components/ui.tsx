/** Shared building blocks used across the app. */

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
      className={cx("rounded-lg border border-line bg-background", className)}
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

/** Small caption placed above a number. */
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
        "inline-flex items-center justify-center gap-2 rounded-md font-medium",
        "transition-colors disabled:cursor-not-allowed",
        size === "sm" ? "h-8 px-3 text-sm" : "h-10 px-4 text-base",
        VARIANTS[variant],
        className,
      )}
    />
  );
}

/** Error message shown above the form it belongs to. */
export function ErrorNote({ children }: { children: ReactNode }) {
  return (
    <p
      role="alert"
      className="rounded-md border border-line-strong bg-surface px-3 py-2 text-sm leading-relaxed text-foreground"
    >
      {children}
    </p>
  );
}
