/**
 * Monochrome line icons. `currentColor` throughout so a glyph always inherits
 * the ink of whatever it sits in — and so status is never carried by shade alone.
 */

type IconProps = { className?: string };

const BASE = {
  viewBox: "0 0 16 16",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.5,
  strokeLinecap: "round",
  strokeLinejoin: "round",
} as const;

export function CheckIcon({ className }: IconProps) {
  return (
    <svg {...BASE} className={className} aria-hidden="true">
      <path d="M3 8.5 6.2 11.5 13 4.5" />
    </svg>
  );
}

export function WarningIcon({ className }: IconProps) {
  return (
    <svg {...BASE} className={className} aria-hidden="true">
      <path d="M8 2.5 14.5 13.5h-13z" />
      <path d="M8 6.5v3.2" />
      <path d="M8 12h.01" />
    </svg>
  );
}

export function AlertIcon({ className }: IconProps) {
  return (
    <svg {...BASE} className={className} aria-hidden="true">
      <circle cx="8" cy="8" r="6" />
      <path d="M8 4.8v3.6" />
      <path d="M8 11h.01" />
    </svg>
  );
}

export function UploadIcon({ className }: IconProps) {
  return (
    <svg {...BASE} className={className} aria-hidden="true">
      <path d="M8 11V2.5" />
      <path d="M4.8 5.7 8 2.5l3.2 3.2" />
      <path d="M2.5 10.5v2a1 1 0 0 0 1 1h9a1 1 0 0 0 1-1v-2" />
    </svg>
  );
}

export function FilmIcon({ className }: IconProps) {
  return (
    <svg {...BASE} className={className} aria-hidden="true">
      <rect x="1.75" y="3.25" width="12.5" height="9.5" rx="1.25" />
      <path d="M5 3.25v9.5M11 3.25v9.5M1.75 8h12.5" />
    </svg>
  );
}

export function CloseIcon({ className }: IconProps) {
  return (
    <svg {...BASE} className={className} aria-hidden="true">
      <path d="M4 4 12 12M12 4 4 12" />
    </svg>
  );
}

export function ClockIcon({ className }: IconProps) {
  return (
    <svg {...BASE} className={className} aria-hidden="true">
      <circle cx="8" cy="8" r="6" />
      <path d="M8 4.6V8l2.3 1.6" />
    </svg>
  );
}

export function ArrowIcon({ className }: IconProps) {
  return (
    <svg {...BASE} className={className} aria-hidden="true">
      <path d="M3 8h10" />
      <path d="M9.4 4.4 13 8l-3.6 3.6" />
    </svg>
  );
}

export function SpinnerIcon({ className }: IconProps) {
  return (
    <svg
      viewBox="0 0 16 16"
      className={className}
      aria-hidden="true"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.75}
      strokeLinecap="round"
    >
      <circle cx="8" cy="8" r="6" opacity={0.2} />
      <path d="M14 8a6 6 0 0 0-6-6">
        <animateTransform
          attributeName="transform"
          type="rotate"
          from="0 8 8"
          to="360 8 8"
          dur="0.8s"
          repeatCount="indefinite"
        />
      </path>
    </svg>
  );
}
