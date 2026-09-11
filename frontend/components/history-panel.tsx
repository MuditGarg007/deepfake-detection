"use client";

/** `GET /history` shows the recent analyses; a row loads its full result. */

import type { HistoryItem } from "@/lib/types";
import { percent, relativeTime } from "@/lib/format";
import { StatusDot } from "./status-badge";
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
              <span className="df-pulse h-4 flex-1 rounded bg-sunken" />
            </li>
          ))}
        </ul>
      ) : items.length === 0 ? (
        <p className="px-4 py-8 text-center text-sm text-muted">
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
                  "flex w-full items-center gap-3 px-4 py-3 text-left transition-colors",
                  item.id === activeId ? "bg-sunken" : "hover:bg-surface",
                )}
              >
                <StatusDot status={item.status} />
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-sm font-medium">
                    {item.filename}
                  </span>
                  <span className="block text-xs text-muted">
                    {relativeTime(item.created_at)}
                    {item.has_video ? ", video saved" : ""}
                  </span>
                </span>
                <span className="text-sm font-medium text-brown">
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
