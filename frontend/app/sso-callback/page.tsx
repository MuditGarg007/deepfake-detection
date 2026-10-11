"use client";

import { HandleSSOCallback } from "@clerk/nextjs";
import { useRouter } from "next/navigation";

import { AuthShell, useFinish } from "@/components/auth";

export default function SSOCallbackPage() {
  const router = useRouter();
  const finish = useFinish();

  return (
    <AuthShell title="Signing you in">
      <p className="flex items-center gap-2 text-sm text-muted">
        <span className="df-spinner" aria-hidden />
        Finishing up with your provider…
      </p>
      <HandleSSOCallback
        navigateToApp={finish}
        navigateToSignIn={() => router.push("/sign-in")}
        navigateToSignUp={() => router.push("/sign-up")}
      />
    </AuthShell>
  );
}
