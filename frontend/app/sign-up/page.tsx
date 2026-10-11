import type { Metadata } from "next";

import { AuthLink, AuthShell } from "@/components/auth";
import { SignUpForm } from "@/components/sign-up-form";

export const metadata: Metadata = { title: "Sign up · Deepfake Detection" };

export default function SignUpPage() {
  return (
    <AuthShell
      title="Create an account"
      hint="We will email you a code to confirm your address."
      footer={
        <>
          Already have an account? <AuthLink href="/sign-in">Sign in</AuthLink>
        </>
      }
    >
      <SignUpForm />
    </AuthShell>
  );
}
