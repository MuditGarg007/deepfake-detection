"use client";

/**
 * Screen orchestrator: upload -> synchronous analysis -> result, with the
 * history rail kept in sync. All three Phase 4 endpoints are driven from here.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError, analyzeVideo, getAnalysis, getHistory } from "@/lib/api";
import type { Analysis, HistoryItem } from "@/lib/types";
import { HistoryPanel } from "./history-panel";
import { ResultPanel } from "./result-panel";
import { UploadCard } from "./upload-card";

function message(error: unknown): string {
  if (error instanceof ApiError) {
    return error.status === 0
      ? "Could not reach the backend. Is it running on the API URL this app is pointed at?"
      : error.message;
  }
  return error instanceof Error ? error.message : "Something went wrong";
}

export function Detector() {
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [busy, setBusy] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [historyLoading, setHistoryLoading] = useState(true);
  const [historyError, setHistoryError] = useState<string | null>(null);

  const startedAt = useRef(0);

  const refreshHistory = useCallback(async () => {
    try {
      setHistory(await getHistory(50));
      setHistoryError(null);
    } catch (cause) {
      setHistoryError(message(cause));
    } finally {
      setHistoryLoading(false);
    }
  }, []);

  // First paint: fill the history rail. Later refreshes go through
  // `refreshHistory` after an analysis lands.
  useEffect(() => {
    let cancelled = false;
    getHistory(50)
      .then((items) => {
        if (cancelled) return;
        setHistory(items);
        setHistoryError(null);
      })
      .catch((cause) => {
        if (!cancelled) setHistoryError(message(cause));
      })
      .finally(() => {
        if (!cancelled) setHistoryLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // A running clock while the pipeline works, so the wait is legible.
  useEffect(() => {
    if (!busy) return;
    const timer = window.setInterval(
      () => setElapsed((Date.now() - startedAt.current) / 1000),
      100,
    );
    return () => window.clearInterval(timer);
  }, [busy]);

  const onAnalyze = useCallback(
    async (file: File) => {
      startedAt.current = Date.now();
      setElapsed(0);
      setUploadProgress(0);
      setError(null);
      setBusy(true);
      try {
        const result = await analyzeVideo(file, { onProgress: setUploadProgress });
        setAnalysis(result);
        void refreshHistory();
      } catch (cause) {
        setError(message(cause));
      } finally {
        setBusy(false);
      }
    },
    [refreshHistory],
  );

  const onSelect = useCallback(async (id: number) => {
    setError(null);
    try {
      setAnalysis(await getAnalysis(id));
    } catch (cause) {
      setError(message(cause));
    }
  }, []);

  return (
    <div className="flex min-h-full flex-1 flex-col">
      <header className="border-b border-line bg-surface">
        <div className="mx-auto w-full max-w-3xl px-6 py-4">
          <h1 className="text-lg font-bold text-brown">
            Deepfake Detection &amp; Alert System
          </h1>
        </div>
      </header>

      <main className="mx-auto w-full max-w-3xl flex-1 px-6 py-8">
        <div className="space-y-6">
          {analysis ? (
            <ResultPanel analysis={analysis} onReset={() => setAnalysis(null)} />
          ) : (
            <UploadCard
              onAnalyze={onAnalyze}
              busy={busy}
              uploadProgress={uploadProgress}
              elapsed={elapsed}
              error={error}
              onDismissError={() => setError(null)}
            />
          )}

          <HistoryPanel
            items={history}
            loading={historyLoading}
            error={historyError}
            activeId={analysis?.id ?? null}
            onSelect={onSelect}
          />
        </div>
      </main>
    </div>
  );
}
