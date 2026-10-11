"use client";

import { useAuth, useSignUp } from "@clerk/nextjs";
import { useState, type FormEvent } from "react";

import {
  AFTER_AUTH,
  CodeForm,
  Field,
  OAuthButtons,
  SSO_CALLBACK,
  clerkMessage,
  useFinish,
  type OAuthStrategy,
} from "./auth";
import { Button, ErrorNote } from "./ui";

export function SignUpForm() {
  const { isLoaded } = useAuth();
  const { signUp, fetchStatus } = useSignUp();
  const finish = useFinish();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [step, setStep] = useState<"start" | "code">("start");
  const [error, setError] = useState<string | null>(null);

  const busy = !isLoaded || fetchStatus === "fetching";

  async function advance() {
    if (signUp.status === "complete") {
      const { error } = await signUp.finalize({ navigate: finish });
      if (error) setError(clerkMessage(error));
      return;
    }

    if (signUp.unverifiedFields.includes("email_address")) {
      const { error } = await signUp.verifications.sendEmailCode();
      if (error) setError(clerkMessage(error));
      else setStep("code");
      return;
    }

    const missing = signUp.missingFields.join(", ").replaceAll("_", " ");
    setError(
      missing
        ? `Sign-up still needs: ${missing}.`
        : "Sign-up could not be completed. Try another method.",
    );
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    setError(null);
    const { error } = await signUp.password({ emailAddress: email.trim(), password });
    if (error) setError(clerkMessage(error));
    else await advance();
  }

  async function verify(code: string) {
    setError(null);
    const { error } = await signUp.verifications.verifyEmailCode({ code });
    if (error) setError(clerkMessage(error));
    else await advance();
  }

  async function resend() {
    setError(null);
    const { error } = await signUp.verifications.sendEmailCode();
    if (error) setError(clerkMessage(error));
  }

  async function oauth(strategy: OAuthStrategy) {
    setError(null);
    const { error } = await signUp.sso({
      strategy,
      redirectUrl: AFTER_AUTH,
      redirectCallbackUrl: SSO_CALLBACK,
    });
    if (error) setError(clerkMessage(error));
  }

  if (step === "code") {
    return (
      <CodeForm
        email={signUp.emailAddress ?? email}
        busy={busy}
        error={error}
        onVerify={verify}
        onResend={resend}
        onBack={() => {
          void signUp.reset();
          setError(null);
          setStep("start");
        }}
      />
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
        <Field
          label="Password"
          type="password"
          name="password"
          autoComplete="new-password"
          minLength={8}
          required
          value={password}
          onChange={(event) => setPassword(event.target.value)}
        />
        {error ? <ErrorNote>{error}</ErrorNote> : null}
        {/* Clerk mounts its bot-protection challenge here when it is enabled. */}
        <div id="clerk-captcha" />
        <Button type="submit" variant="primary" className="w-full" disabled={busy}>
          {fetchStatus === "fetching" ? <span className="df-spinner" aria-hidden /> : null}
          Create account
        </Button>
      </form>
    </>
  );
}
