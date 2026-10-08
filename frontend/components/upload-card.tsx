"use client";

import { useRef, useState } from "react";

import { analyzeVideo, errorMessage } from "@/lib/api";
import { fileSize } from "@/lib/format";
import type { Analysis } from "@/lib/types";
import { Button, Card, ErrorNote, cx } from "./ui";

const ACCEPTED = [".mp4", ".avi", ".mov"];
const MAX_MB = 200;

function validate(file: File): string | null {
  const extension = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
  if (!ACCEPTED.includes(extension)) {
    return `Unsupported file type. Allowed types are ${ACCEPTED.join(", ")}`;
  }
  if (file.size > MAX_MB * 1024 * 1024) {
    return `This file is ${fileSize(file.size)}. The limit is ${MAX_MB} MB.`;
  }
  if (file.size === 0) return "That file is empty";
  return null;
}

export function UploadCard({
  onDone,
}: {
  onDone: (analysis: Analysis) => void;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const [progress, setProgress] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  const busy = progress !== null;

  function pick(candidate: File | undefined) {
    if (!candidate || busy) return;
    const problem = validate(candidate);
    setError(problem);
    setFile(problem ? null : candidate);
  }

  async function run() {
    if (!file) return;
    setError(null);
    setProgress(0);
    try {
      const analysis = await analyzeVideo(file, setProgress);
      setFile(null);
      if (inputRef.current) inputRef.current.value = "";
      onDone(analysis);
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setProgress(null);
    }
  }

  return (
    <Card>
      <div className="p-5">
        <h2 className="text-xl font-semibold">Check a video file</h2>
        <p className="mt-1 text-sm text-muted">
          MP4, AVI or MOV, up to {MAX_MB} MB. Every frame with a face in it gets
          scored.
        </p>

        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED.join(",")}
          className="sr-only"
          disabled={busy}
          onChange={(event) => pick(event.target.files?.[0])}
        />

        <button
          type="button"
          disabled={busy}
          onClick={() => inputRef.current?.click()}
          onDragOver={(event) => {
            event.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(event) => {
            event.preventDefault();
            setDragging(false);
            pick(event.dataTransfer.files[0]);
          }}
          className={cx(
            "mt-4 flex w-full flex-col items-center justify-center gap-1 border border-dashed px-4 py-8 text-center transition-colors",
            dragging
              ? "border-brown bg-sunken"
              : "border-line-strong bg-surface hover:bg-sunken",
          )}
        >
          {file ? (
            <>
              <span className="max-w-full truncate font-medium">
                {file.name}
              </span>
              <span className="text-sm text-muted">{fileSize(file.size)}</span>
            </>
          ) : (
            <>
              <span className="font-medium">Drop a video here</span>
              <span className="text-sm text-muted">or click to choose one</span>
            </>
          )}
        </button>

        {error ? (
          <div className="mt-3">
            <ErrorNote>{error}</ErrorNote>
          </div>
        ) : null}

        <div className="mt-4 flex items-center gap-3">
          <Button variant="primary" disabled={!file || busy} onClick={run}>
            Run detection
          </Button>
          {busy ? (
            <span className="flex items-center gap-2 text-sm text-muted">
              <span className="df-spinner" />
              {progress < 1
                ? `Uploading ${Math.round(progress * 100)}%`
                : "Analyzing frame by frame"}
            </span>
          ) : null}
        </div>
      </div>
    </Card>
  );
}
