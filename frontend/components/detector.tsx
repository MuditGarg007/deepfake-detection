"use client";

import { useCallback, useEffect, useState } from "react";

import {
  API_URL,
  errorMessage,
  getAnalysis,
  getHealth,
  getHistory,
  getLiveSession,
} from "@/lib/api";
import type { Analysis, Health, HistoryItem, LiveSession } from "@/lib/types";
import { HistoryPanel } from "./history-panel";
import { LiveSummary } from "./live-summary";
import { ResultPanel } from "./result-panel";
import { ScreenShare } from "./screen-share";
import { Button, ErrorNote, cx } from "./ui";
import { UploadCard } from "./upload-card";

const TABS = [
  { id: "live", label: "Screen share" },
  { id: "upload", label: "Upload video" },
  { id: "history", label: "History" },
] as const;

type Tab = (typeof TABS)[number]["id"];

type Selected =
  | { kind: "upload"; analysis: Analysis }
  | { kind: "live"; session: LiveSession };

export function Detector() {
  const [tab, setTab] = useState<Tab>("live");
  const [health, setHealth] = useState<Health | null>(null);
  const [offline, setOffline] = useState(false);

  const [result, setResult] = useState<Analysis | null>(null);

  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [historyLoading, setHistoryLoading] = useState(true);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [selected, setSelected] = useState<Selected | null>(null);
  const [selectError, setSelectError] = useState<string | null>(null);

  const refreshHistory = useCallback(async () => {
    try {
      setHistory(await getHistory());
      setHistoryError(null);
    } catch (cause) {
      setHistoryError(errorMessage(cause));
    } finally {
      setHistoryLoading(false);
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    getHealth()
      .then((value) => !cancelled && setHealth(value))
      .catch(() => !cancelled && setOffline(true));
    getHistory()
      .then((items) => !cancelled && setHistory(items))
      .catch((cause) => !cancelled && setHistoryError(errorMessage(cause)))
      .finally(() => !cancelled && setHistoryLoading(false));
    return () => {
      cancelled = true;
    };
  }, []);

  const onSaved = useCallback(() => {
    void refreshHistory();
  }, [refreshHistory]);

  async function open(item: HistoryItem) {
    setSelectError(null);
    try {
      setSelected(
        item.kind === "live"
          ? { kind: "live", session: await getLiveSession(item.id) }
          : { kind: "upload", analysis: await getAnalysis(item.id) },
      );
    } catch (cause) {
      setSelectError(errorMessage(cause));
    }
  }

  function updateAnalysis(updated: Analysis) {
    if (result?.id === updated.id) setResult(updated);
    if (selected?.kind === "upload" && selected.analysis.id === updated.id) {
      setSelected({ kind: "upload", analysis: updated });
    }
    void refreshHistory();
  }

  const activeKey = selected
    ? selected.kind === "live"
      ? `live-${selected.session.id}`
      : `upload-${selected.analysis.id}`
    : null;

  const close = (
    <Button variant="ghost" size="sm" onClick={() => setSelected(null)}>
      Close
    </Button>
  );

  return (
    <div className="flex min-h-full flex-1 flex-col">
      <header className="border-b border-line bg-surface">
        <div className="mx-auto w-full max-w-3xl px-6 pt-5">
          <h1 className="text-lg font-bold text-brown">Deepfake Detection</h1>
          <nav role="tablist" className="mt-3 flex gap-1">
            {TABS.map(({ id, label }) => (
              <button
                key={id}
                type="button"
                role="tab"
                aria-selected={tab === id}
                onClick={() => setTab(id)}
                className={cx(
                  "-mb-px border-b-2 px-3 py-2 text-sm font-medium transition-colors",
                  tab === id
                    ? "border-brown text-brown"
                    : "border-transparent text-muted hover:text-foreground",
                )}
              >
                {label}
              </button>
            ))}
          </nav>
        </div>
      </header>

      <main className="mx-auto w-full max-w-3xl flex-1 space-y-6 px-6 py-8">
        {offline ? (
          <ErrorNote>
            Could not reach the backend at {API_URL}. Start it and reload the
            page.
          </ErrorNote>
        ) : health && !health.model_loaded ? (
          <ErrorNote>
            The backend has no checkpoint loaded, so every frame scores 50%.
          </ErrorNote>
        ) : null}

        <section hidden={tab !== "live"}>
          <ScreenShare onSaved={onSaved} />
        </section>

        <section hidden={tab !== "upload"} className="space-y-6">
          {result ? (
            <>
              <ResultPanel analysis={result} onUpdate={updateAnalysis} />
              <Button onClick={() => setResult(null)}>
                Check another video
              </Button>
            </>
          ) : (
            <UploadCard
              onDone={(analysis) => {
                setResult(analysis);
                void refreshHistory();
              }}
            />
          )}
        </section>

        <section hidden={tab !== "history"} className="space-y-6">
          {selectError ? <ErrorNote>{selectError}</ErrorNote> : null}
          {selected?.kind === "upload" ? (
            <ResultPanel
              analysis={selected.analysis}
              onUpdate={updateAnalysis}
              action={close}
            />
          ) : selected?.kind === "live" ? (
            <LiveSummary session={selected.session} action={close} />
          ) : null}
          <HistoryPanel
            items={history}
            loading={historyLoading}
            error={historyError}
            activeKey={activeKey}
            onSelect={open}
          />
        </section>
      </main>
    </div>
  );
}
