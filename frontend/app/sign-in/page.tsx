import type { Metadata } from "next";

import { AuthLink, AuthShell } from "@/components/auth";
import { SignInForm } from "@/components/sign-in-form";

export const metadata: Metadata = { title: "Sign in · Deepfake Detection" };

export default function SignInPage() {
  return (
    <AuthShell
      title="Sign in"
      hint="Use your account to check videos and screen shares."
      footer={
        <>
          No account yet? <AuthLink href="/sign-up">Sign up</AuthLink>
        </>
      }
    >
      <SignInForm />
    </AuthShell>
  );
}
