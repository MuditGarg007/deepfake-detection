"use client";

/**
 * Upload screen: pick or drop a video, then `POST /analyze` and hold until the
 * synchronous pipeline answers. Extension and size are checked here so the
 * obvious rejections never cost a round trip (the backend enforces both again).
 */

import { useCallback, useRef, useState } from "react";

import { fileSize } from "@/lib/format";
import { CloseIcon, FilmIcon, SpinnerIcon, UploadIcon } from "./icons";
import { Button, Card, ErrorNote, cx } from "./ui";

const ACCEPTED = [".mp4", ".avi", ".mov"];
const MAX_MB = 200;

const STAGES = [
  "Sampling frames at ~5 fps",
  "Detecting and cropping faces",
  "Scoring crops with the model",
  "Aggregating the verdict",
];

function validate(file: File): string | null {
  const extension = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
  if (!ACCEPTED.includes(extension)) {
    return `Unsupported file type — allowed: ${ACCEPTED.join(", ")}`;
  }
  if (file.size > MAX_MB * 1024 * 1024) {
    return `File is ${fileSize(file.size)} — the limit is ${MAX_MB} MB`;
  }
  if (file.size === 0) return "That file is empty";
  return null;
}

export function UploadCard({
  onAnalyze,
  busy,
  uploadProgress,
  elapsed,
  error,
  onDismissError,
}: {
  onAnalyze: (file: File) => void;
  busy: boolean;
  uploadProgress: number;
  elapsed: number;
  error: string | null;
  onDismissError: () => void;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [localError, setLocalError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);

  const accept = useCallback((candidate: File | undefined) => {
    if (!candidate) return;
    const problem = validate(candidate);
    setLocalError(problem);
    setFile(problem ? null : candidate);
  }, []);

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();
      setDragging(false);
      if (busy) return;
      accept(event.dataTransfer.files[0]);
    },
    [accept, busy],
  );

  const uploading = busy && uploadProgress < 1;
  const stage = STAGES[Math.min(STAGES.length - 1, Math.floor(elapsed / 3))];

  return (
    <Card className="overflow-hidden">
      <div className="px-5 pt-5">
        <h1 className="text-lg font-semibold tracking-tight">Analyze a video</h1>
        <p className="mt-1 text-[13px] leading-relaxed text-muted">
          Faces are sampled from the clip and scored frame by frame. Detection runs
          synchronously — a short clip takes a few seconds.
        </p>
      </div>

      <div className="p-5">
        <div
          onDragOver={(event) => {
            event.preventDefault();
            if (!busy) setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          className={cx(
            "rounded-xl border border-dashed transition-colors duration-150",
            dragging ? "border-foreground bg-sunken" : "border-line-strong bg-surface",
            busy && "opacity-60",
          )}
        >
          <input
            ref={inputRef}
            type="file"
            accept={ACCEPTED.join(",")}
            className="sr-only"
            disabled={busy}
            onChange={(event) => accept(event.target.files?.[0])}
          />

          {file ? (
            <div className="flex items-center gap-3 px-4 py-4">
              <span className="flex size-9 shrink-0 items-center justify-center rounded-lg border border-line bg-background">
                <FilmIcon className="size-4" />
              </span>
              <div className="min-w-0 flex-1">
                <p className="truncate text-[13px] font-medium">{file.name}</p>
                <p className="font-mono text-[11px] text-faint">{fileSize(file.size)}</p>
              </div>
              {!busy ? (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => {
                    setFile(null);
                    if (inputRef.current) inputRef.current.value = "";
                  }}
                  aria-label="Remove selected file"
                >
                  <CloseIcon className="size-3.5" />
                </Button>
              ) : null}
            </div>
          ) : (
            <button
              type="button"
              disabled={busy}
              onClick={() => inputRef.current?.click()}
              className="flex w-full flex-col items-center gap-2 px-4 py-10 text-center"
            >
              <span className="flex size-10 items-center justify-center rounded-full border border-line bg-background">
                <UploadIcon className="size-4" />
              </span>
              <span className="text-[13px] font-medium">
                Drop a video here, or <span className="underline underline-offset-2">browse</span>
              </span>
              <span className="font-mono text-[11px] text-faint">
                {ACCEPTED.join(" · ")} — up to {MAX_MB} MB
              </span>
            </button>
          )}
        </div>

        {localError ? (
          <div className="mt-3">
            <ErrorNote>{localError}</ErrorNote>
          </div>
        ) : null}

        {error ? (
          <div className="mt-3">
            <ErrorNote>
              {error}{" "}
              <button
                type="button"
                onClick={onDismissError}
                className="underline underline-offset-2"
              >
                Dismiss
              </button>
            </ErrorNote>
          </div>
        ) : null}

        {busy ? (
          <div className="mt-4 space-y-2">
            <div className="relative h-1 overflow-hidden rounded-full bg-sunken">
              {uploading ? (
                <div
                  className="h-full bg-foreground transition-[width] duration-150"
                  style={{ width: `${Math.round(uploadProgress * 100)}%` }}
                />
              ) : (
                <div className="df-sweep absolute inset-0" />
              )}
            </div>
            <p className="flex items-center gap-2 text-xs text-muted">
              <SpinnerIcon className="size-3.5" />
              {uploading
                ? `Uploading — ${Math.round(uploadProgress * 100)}%`
                : stage}
              <span className="ml-auto font-mono tabular-nums text-faint">
                {elapsed.toFixed(0)}s
              </span>
            </p>
          </div>
        ) : (
          <Button
            variant="primary"
            className="mt-4 w-full"
            disabled={!file}
            onClick={() => file && onAnalyze(file)}
          >
            Run detection
          </Button>
        )}
      </div>
    </Card>
  );
}
