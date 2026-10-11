"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState, type ComponentProps, type FormEvent, type ReactNode } from "react";

import { Button, Card, CardHeader, ErrorNote } from "./ui";

export const AFTER_AUTH = "/";
export const SSO_CALLBACK = "/sso-callback";

export const PROVIDERS = [
  { strategy: "oauth_google", label: "Continue with Google" },
  { strategy: "oauth_github", label: "Continue with GitHub" },
] as const;

export type OAuthStrategy = (typeof PROVIDERS)[number]["strategy"];

type ClerkLikeError = {
  message?: string;
  longMessage?: string;
  errors?: { message?: string; longMessage?: string }[];
};

/** The human part of a Clerk error, without its `Clerk:` prefix and code. */
export function clerkMessage(error: unknown): string {
  const value = (error ?? {}) as ClerkLikeError;
  const first = value.errors?.[0];
  const text =
    first?.longMessage ??
    first?.message ??
    value.longMessage ??
    value.message ??
    "Something went wrong";
  return text.replace(/^Clerk:\s*/i, "").split("\n")[0];
}

/** `navigate` callback for `finalize()`: go to the app once the session is set. */
export function useFinish() {
  const router = useRouter();
  return ({ decorateUrl }: { decorateUrl: (url: string) => string }) => {
    const url = decorateUrl(AFTER_AUTH);
    if (url.startsWith("http")) window.location.href = url;
    else router.push(url);
  };
}

export function AuthShell({
  title,
  hint,
  footer,
  children,
}: {
  title: ReactNode;
  hint?: ReactNode;
  footer?: ReactNode;
  children: ReactNode;
}) {
  return (
    <div className="flex min-h-full flex-1 flex-col">
      <header className="border-b border-line bg-surface">
        <div className="mx-auto w-full max-w-3xl px-6 py-5">
          <h1 className="text-lg font-bold text-brown">Deepfake Detection</h1>
        </div>
      </header>

      <main className="mx-auto w-full max-w-sm flex-1 space-y-4 px-6 py-12">
        <Card>
          <CardHeader title={title} hint={hint} />
          <div className="space-y-4 p-5">{children}</div>
        </Card>
        {footer ? <p className="text-center text-sm text-muted">{footer}</p> : null}
      </main>
    </div>
  );
}

export function AuthLink({ href, children }: { href: string; children: ReactNode }) {
  return (
    <Link href={href} className="font-medium text-brown hover:underline">
      {children}
    </Link>
  );
}

export function Field({
  label,
  ...props
}: ComponentProps<"input"> & { label: string }) {
  return (
    <label className="block space-y-1">
      <span className="text-sm text-muted">{label}</span>
      <input
        {...props}
        className="h-10 w-full border border-line-strong bg-background px-3 text-base focus:border-brown"
      />
    </label>
  );
}

/** A `Field` for passwords, with an eye button that shows or hides the text. */
export function PasswordField({
  label,
  ...props
}: Omit<ComponentProps<"input">, "type"> & { label: string }) {
  const [visible, setVisible] = useState(false);
  return (
    <label className="block space-y-1">
      <span className="text-sm text-muted">{label}</span>
      <span className="relative block">
        <input
          {...props}
          type={visible ? "text" : "password"}
          className="h-10 w-full border border-line-strong bg-background pl-3 pr-10 text-base focus:border-brown"
        />
        <button
          type="button"
          onClick={() => setVisible((shown) => !shown)}
          aria-label={visible ? "Hide password" : "Show password"}
          aria-pressed={visible}
          className="absolute inset-y-0 right-0 flex w-10 items-center justify-center text-muted hover:text-brown"
        >
          <EyeIcon crossed={visible} />
        </button>
      </span>
    </label>
  );
}

function EyeIcon({ crossed }: { crossed: boolean }) {
  return (
    <svg
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z" />
      <circle cx="12" cy="12" r="3" />
      {crossed ? <path d="M3 3l18 18" /> : null}
    </svg>
  );
}

export function OAuthButtons({
  disabled,
  onPick,
}: {
  disabled: boolean;
  onPick: (strategy: OAuthStrategy) => void;
}) {
  return (
    <>
      <div className="space-y-2">
        {PROVIDERS.map(({ strategy, label }) => (
          <Button
            key={strategy}
            type="button"
            className="w-full"
            disabled={disabled}
            onClick={() => onPick(strategy)}
          >
            {label}
          </Button>
        ))}
      </div>
      <p className="flex items-center gap-3 text-xs text-faint">
        <span className="h-px flex-1 bg-line" />
        or
        <span className="h-px flex-1 bg-line" />
      </p>
    </>
  );
}

/** The one-time-code step shared by sign-up and sign-in. */
export function CodeForm({
  email,
  busy,
  error,
  onVerify,
  onResend,
  onBack,
}: {
  email: string | null;
  busy: boolean;
  error: string | null;
  onVerify: (code: string) => void;
  onResend: () => void;
  onBack: () => void;
}) {
  const [code, setCode] = useState("");

  function submit(event: FormEvent) {
    event.preventDefault();
    onVerify(code.trim());
  }

  return (
    <form onSubmit={submit} className="space-y-4">
      <p className="text-sm text-muted">
        We sent a 6-digit code to{" "}
        <span className="font-medium text-foreground">{email ?? "your email"}</span>.
      </p>
      <Field
        label="Verification code"
        name="code"
        inputMode="numeric"
        autoComplete="one-time-code"
        pattern="[0-9]{6}"
        maxLength={6}
        required
        autoFocus
        value={code}
        onChange={(event) => setCode(event.target.value.replace(/\D/g, ""))}
      />
      {error ? <ErrorNote>{error}</ErrorNote> : null}
      <Button type="submit" variant="primary" className="w-full" disabled={busy}>
        {busy ? <span className="df-spinner" aria-hidden /> : null}
        Verify
      </Button>
      <div className="flex justify-between">
        <Button type="button" variant="ghost" size="sm" onClick={onBack} disabled={busy}>
          Back
        </Button>
        <Button type="button" variant="ghost" size="sm" onClick={onResend} disabled={busy}>
          Resend code
        </Button>
      </div>
    </form>
  );
}
