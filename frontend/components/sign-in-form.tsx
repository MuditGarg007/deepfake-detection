"use client";

import { useAuth, useSignIn } from "@clerk/nextjs";
import { useState, type FormEvent } from "react";

import {
  AFTER_AUTH,
  CodeForm,
  Field,
  OAuthButtons,
  PasswordField,
  SSO_CALLBACK,
  clerkMessage,
  useFinish,
  type OAuthStrategy,
} from "./auth";
import { Button, ErrorNote } from "./ui";

// A new device ("client trust") or an account with email 2FA both finish with
// an emailed code.
const NEEDS_CODE = ["needs_second_factor", "needs_client_trust"];

export function SignInForm() {
  const { isLoaded } = useAuth();
  const { signIn, fetchStatus } = useSignIn();
  const finish = useFinish();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [step, setStep] = useState<"start" | "code">("start");
  const [error, setError] = useState<string | null>(null);
  // Sent back here from Google/GitHub with the device check still to do.
  const [resuming, setResuming] = useState(
    () => isLoaded && NEEDS_CODE.includes(signIn.status ?? ""),
  );

  const busy = !isLoaded || fetchStatus === "fetching";

  async function advance() {
    if (signIn.status === "complete") {
      const { error } = await signIn.finalize({ navigate: finish });
      if (error) setError(clerkMessage(error));
      return;
    }

    if (NEEDS_CODE.includes(signIn.status ?? "")) {
      const byEmail = signIn.supportedSecondFactors.some(
        (factor) => factor.strategy === "email_code",
      );
      if (!byEmail) {
        setError("This account needs a second factor that this page does not support.");
        return;
      }
      const { error } = await signIn.mfa.sendEmailCode();
      if (error) setError(clerkMessage(error));
      else setStep("code");
      return;
    }

    setError("Sign-in could not be completed. Try another method.");
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    setError(null);
    const { error } = await signIn.password({ emailAddress: email.trim(), password });
    if (error) setError(clerkMessage(error));
    else await advance();
  }

  async function verify(code: string) {
    setError(null);
    const { error } = await signIn.mfa.verifyEmailCode({ code });
    if (error) setError(clerkMessage(error));
    else await advance();
  }

  async function resend() {
    setError(null);
    const { error } = await signIn.mfa.sendEmailCode();
    if (error) setError(clerkMessage(error));
  }

  async function oauth(strategy: OAuthStrategy) {
    setError(null);
    const { error } = await signIn.sso({
      strategy,
      redirectUrl: AFTER_AUTH,
      redirectCallbackUrl: SSO_CALLBACK,
    });
    if (error) setError(clerkMessage(error));
  }

  if (step === "code") {
    return (
      <CodeForm
        email={signIn.identifier ?? email}
        busy={busy}
        error={error}
        onVerify={verify}
        onResend={resend}
        onBack={() => {
          void signIn.reset();
          setError(null);
          setResuming(false);
          setStep("start");
        }}
      />
    );
  }

  if (resuming) {
    return (
      <>
        <p className="text-sm text-muted">
          One more step: confirm it is you with a code sent to your email.
        </p>
        {error ? <ErrorNote>{error}</ErrorNote> : null}
        <Button
          variant="primary"
          className="w-full"
          disabled={busy}
          onClick={() => void advance()}
        >
          Email me a code
        </Button>
        <Button
          variant="ghost"
          size="sm"
          disabled={busy}
          onClick={() => {
            void signIn.reset();
            setError(null);
            setResuming(false);
          }}
        >
          Use another method
        </Button>
      </>
    );
  }

  return (
    <>
      <OAuthButtons disabled={busy} onPick={oauth} />
      <form onSubmit={submit} className="space-y-4">
        <Field
          label="Email"
          type="email"
          name="email"
          autoComplete="email"
          required
          value={email}
          onChange={(event) => setEmail(event.target.value)}
        />
        <PasswordField
          label="Password"
          name="password"
          autoComplete="current-password"
          required
          value={password}
          onChange={(event) => setPassword(event.target.value)}
        />
        {error ? <ErrorNote>{error}</ErrorNote> : null}
        <Button type="submit" variant="primary" className="w-full" disabled={busy}>
          {fetchStatus === "fetching" ? <span className="df-spinner" aria-hidden /> : null}
          Sign in
        </Button>
      </form>
    </>
  );
}
