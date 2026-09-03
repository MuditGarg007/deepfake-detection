"use client";

/** `GET /history` — the recent analyses rail; a row loads its full result. */

import type { HistoryItem } from "@/lib/types";
import { percent, relativeTime } from "@/lib/format";
import { StatusGlyph } from "./status-badge";
import { FilmIcon } from "./icons";
import { Card, CardHeader, ErrorNote, cx } from "./ui";

export function HistoryPanel({
  items,
  loading,
  error,
  activeId,
  onSelect,
}: {
  items: HistoryItem[];
  loading: boolean;
  error: string | null;
  activeId: number | null;
  onSelect: (id: number) => void;
}) {
  return (
    <Card className="overflow-hidden">
      <CardHeader
        title="Recent analyses"
        hint={items.length ? `${items.length} stored` : undefined}
      />

      {error ? (
        <div className="p-4">
          <ErrorNote>{error}</ErrorNote>
        </div>
      ) : loading ? (
        <ul className="divide-y divide-line">
          {[0, 1, 2, 3].map((row) => (
            <li key={row} className="flex items-center gap-3 px-4 py-3">
              <span className="df-pulse size-6 rounded-full bg-sunken" />
              <span className="df-pulse h-3 flex-1 rounded bg-sunken" />
            </li>
          ))}
        </ul>
      ) : items.length === 0 ? (
        <p className="px-4 py-8 text-center text-xs text-muted">
          Nothing analyzed yet. Your results will collect here.
        </p>
      ) : (
        <ul className="max-h-[560px] divide-y divide-line overflow-y-auto">
          {items.map((item) => (
            <li key={item.id}>
              <button
                type="button"
                onClick={() => onSelect(item.id)}
                aria-current={item.id === activeId}
                className={cx(
                  "flex w-full items-center gap-3 px-4 py-3 text-left transition-colors duration-150",
                  item.id === activeId ? "bg-sunken" : "hover:bg-surface",
                )}
              >
                <StatusGlyph status={item.status} />
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-[13px] font-medium">
                    {item.filename}
                  </span>
                  <span className="flex items-center gap-1.5 text-[11px] text-faint">
                    {relativeTime(item.created_at)}
                    {item.has_video ? (
                      <>
                        <FilmIcon className="size-3" />
                        <span className="sr-only">video available</span>
                      </>
                    ) : null}
                  </span>
                </span>
                <span className="font-mono text-xs tabular-nums text-muted">
                  {percent(item.fake_probability, 0)}
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </Card>
  );
}
