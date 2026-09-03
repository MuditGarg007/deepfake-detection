"use client";

/** `GET /health` — reachability plus whether a real checkpoint is loaded. */

import { useEffect, useState } from "react";

import { getHealth } from "@/lib/api";
import { cx } from "./ui";

type State =
  | { kind: "loading" }
  | { kind: "online"; modelLoaded: boolean }
  | { kind: "offline" };

export function HealthBadge() {
  const [state, setState] = useState<State>({ kind: "loading" });

  useEffect(() => {
    let cancelled = false;
    getHealth()
      .then((health) => {
        if (!cancelled) setState({ kind: "online", modelLoaded: health.model_loaded });
      })
      .catch(() => {
        if (!cancelled) setState({ kind: "offline" });
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const text =
    state.kind === "loading"
      ? "Checking backend"
      : state.kind === "offline"
        ? "Backend unreachable"
        : state.modelLoaded
          ? "Model loaded"
          : "Stub mode — no checkpoint";

  return (
    <span
      className="inline-flex shrink-0 items-center gap-2 whitespace-nowrap rounded-full border border-line px-2.5 py-1 text-[11px] text-muted"
      title={
        state.kind === "online" && !state.modelLoaded
          ? "The backend booted without a checkpoint and returns a fixed 0.5 for every frame."
          : undefined
      }
    >
      <span
        aria-hidden="true"
        className={cx(
          "size-1.5 rounded-full",
          state.kind === "loading" && "df-pulse bg-faint",
          state.kind === "offline" && "bg-line-strong ring-1 ring-line-strong",
          state.kind === "online" && (state.modelLoaded ? "bg-foreground" : "bg-faint"),
        )}
      />
      {text}
    </span>
  );
}
