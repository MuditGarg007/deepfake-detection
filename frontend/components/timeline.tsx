import { seconds } from "@/lib/format";
import type { FrameScore } from "@/lib/types";

export function Timeline({
  points,
  highlight,
}: {
  points: FrameScore[];
  highlight?: [number, number] | null;
}) {
  if (points.length < 2) return null;

  const start = points[0].timestamp;
  const end = points[points.length - 1].timestamp;
  const span = Math.max(end - start, 0.001);
  const x = (time: number) => ((time - start) / span) * 100;
  const line = points
    .map(
      (point) =>
        `${x(point.timestamp).toFixed(2)},${((1 - point.fake_probability) * 100).toFixed(2)}`,
    )
    .join(" ");

  return (
    <figure>
      <div className="relative h-32 border border-line bg-surface">
        <svg
          viewBox="0 0 100 100"
          preserveAspectRatio="none"
          aria-hidden="true"
          className="absolute inset-0 h-full w-full"
        >
          {highlight ? (
            <rect
              x={x(highlight[0])}
              width={Math.max(x(highlight[1]) - x(highlight[0]), 0.5)}
              y={0}
              height={100}
              className="fill-sunken"
            />
          ) : null}
          <line
            x1={0}
            x2={100}
            y1={50}
            y2={50}
            strokeDasharray="4 4"
            vectorEffect="non-scaling-stroke"
            className="stroke-line-strong"
          />
          <polyline
            points={line}
            fill="none"
            strokeWidth={1.5}
            strokeLinejoin="round"
            vectorEffect="non-scaling-stroke"
            className="stroke-brown"
          />
        </svg>
        <span className="absolute left-1.5 top-1 text-xs text-faint">100%</span>
        <span className="absolute bottom-1 left-1.5 text-xs text-faint">
          0%
        </span>
      </div>
      <figcaption className="mt-1 flex justify-between text-xs text-muted">
        <span>{seconds(start)}</span>
        <span>Fake probability over time</span>
        <span>{seconds(end)}</span>
      </figcaption>
    </figure>
  );
}
