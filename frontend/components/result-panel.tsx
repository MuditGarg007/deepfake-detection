"use client";

/** Result screen: the verdict and the aggregate score, nothing else. */

import type { Analysis } from "@/lib/types";
import { percent } from "@/lib/format";
import { StatusBadge } from "./status-badge";
import { Button, Card } from "./ui";

export function ResultPanel({
  analysis,
  onReset,
}: {
  analysis: Analysis;
  onReset: () => void;
}) {
  return (
    <Card>
      <div className="p-5">
        <h2 className="truncate text-xl font-semibold">{analysis.filename}</h2>

        <p className="mt-4 text-5xl font-bold text-brown">
          {percent(analysis.fake_probability)}
        </p>
        <p className="mt-1 text-sm text-muted">Fake probability</p>

        <div className="mt-4">
          <StatusBadge status={analysis.status} />
        </div>

        <div className="mt-6">
          <Button variant="secondary" onClick={onReset}>
            Analyze another video
          </Button>
        </div>
      </div>
    </Card>
  );
}
