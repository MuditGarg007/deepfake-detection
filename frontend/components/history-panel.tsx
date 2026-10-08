"use client";

import type { HistoryItem } from "@/lib/types";
import { STATUS_LABEL, percent, relativeTime } from "@/lib/format";
import { Card, CardHeader, ErrorNote, TONE, cx } from "./ui";

export function HistoryPanel({
  items,
  loading,
  error,
  activeKey,
  onSelect,
}: {
  items: HistoryItem[];
  loading: boolean;
  error: string | null;
  activeKey: string | null;
  onSelect: (item: HistoryItem) => void;
}) {
  return (
    <Card className="overflow-hidden">
      <CardHeader
        title="History"
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
              <span className="df-pulse h-4 flex-1 bg-sunken" />
            </li>
          ))}
        </ul>
      ) : items.length === 0 ? (
        <p className="px-4 py-8 text-center text-sm text-muted">
          Nothing checked yet. Uploads and screen shares show up here.
        </p>
      ) : (
        <ul className="max-h-[560px] divide-y divide-line overflow-y-auto">
          {items.map((item) => {
            const key = `${item.kind}-${item.id}`;
            return (
              <li key={key}>
                <button
                  type="button"
                  onClick={() => onSelect(item)}
                  aria-current={key === activeKey}
                  className={cx(
                    "flex w-full items-center gap-3 px-4 py-3 text-left transition-colors",
                    key === activeKey ? "bg-sunken" : "hover:bg-surface",
                  )}
                >
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-sm font-medium">
                      {item.filename}
                    </span>
                    <span className="block text-xs text-muted">
                      {item.kind === "live" ? "Screen share" : "Upload"},{" "}
                      {relativeTime(item.created_at)}
                    </span>
                  </span>
                  <span className="shrink-0 text-right text-sm">
                    <span
                      className={cx("block font-medium", TONE[item.status])}
                    >
                      {STATUS_LABEL[item.status]}
                    </span>
                    <span className="block text-xs text-muted tabular-nums">
                      {percent(item.fake_probability)}
                    </span>
                  </span>
                </button>
              </li>
            );
          })}
        </ul>
      )}
    </Card>
  );
}
