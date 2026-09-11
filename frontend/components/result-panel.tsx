"use client";

/** Result screen: the verdict and the aggregate score, nothing else. */

import type { Analysis } from "@/lib/types";
import { STATUS_LABEL, percent } from "@/lib/format";
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
      <div className="p-5 text-base">
        <h2 className="truncate font-semibold">{analysis.filename}</h2>

        <p className="mt-4">
          Fake probability: {percent(analysis.fake_probability)}
        </p>
        <p className="mt-1">Verdict: {STATUS_LABEL[analysis.status]}</p>

        <div className="mt-6">
          <Button variant="secondary" onClick={onReset}>
            Analyze another video
          </Button>
        </div>
      </div>
    </Card>
  );
}
