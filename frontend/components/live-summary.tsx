import type { ReactNode } from "react";

import { percent, seconds } from "@/lib/format";
import type { LiveSession } from "@/lib/types";
import { Timeline } from "./timeline";
import { Card, CardHeader, Stats, Verdict } from "./ui";

export function LiveSummary({
  session,
  title = "Screen share",
  action,
}: {
  session: LiveSession;
  title?: string;
  action?: ReactNode;
}) {
  return (
    <Card>
      <CardHeader
        title={title}
        hint={`${session.frames_scored} frames with a face`}
        action={action}
      />
      <div className="space-y-4 p-5">
        <Verdict
          status={session.status}
          probability={session.mean_probability}
        />
        <Stats
          items={[
            ["Average", percent(session.mean_probability)],
            ["Peak", percent(session.peak_probability)],
            ["Duration", seconds(session.duration_seconds)],
          ]}
        />
        <Timeline points={session.timeline} />
      </div>
    </Card>
  );
}
