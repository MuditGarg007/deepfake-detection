"use client";

/**
 * Upload screen: pick a video, then `POST /analyze` and hold until the
 * synchronous pipeline answers. Extension and size are checked here so the
 * obvious rejections never cost a round trip (the backend enforces both again).
 */

import { useCallback, useRef, useState } from "react";

import { fileSize } from "@/lib/format";
import { Button, Card, ErrorNote } from "./ui";

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
  onAnalyze,
  busy,
  error,
  onDismissError,
}: {
  onAnalyze: (file: File) => void;
  busy: boolean;
  error: string | null;
  onDismissError: () => void;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [localError, setLocalError] = useState<string | null>(null);

  const accept = useCallback((candidate: File | undefined) => {
    if (!candidate) return;
    const problem = validate(candidate);
    setLocalError(problem);
    setFile(problem ? null : candidate);
  }, []);

  return (
    <Card>
      <div className="p-5">
        <h2 className="text-xl font-semibold">Analyze a video</h2>

        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED.join(",")}
          className="sr-only"
          disabled={busy}
          onChange={(event) => accept(event.target.files?.[0])}
        />

        <div className="mt-4 flex flex-wrap items-center gap-3">
          <Button
            variant="secondary"
            disabled={busy}
            onClick={() => inputRef.current?.click()}
          >
            Choose file
          </Button>
          <span className="min-w-0 flex-1 truncate text-base text-muted">
            {file ? `${file.name} (${fileSize(file.size)})` : "No file chosen"}
          </span>
          {file && !busy ? (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => {
                setFile(null);
                if (inputRef.current) inputRef.current.value = "";
              }}
            >
              Remove
            </Button>
          ) : null}
        </div>

        <p className="mt-2 text-sm text-muted">
          Upload a video for deepfake detection
        </p>

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
                className="text-brown underline"
              >
                Dismiss
              </button>
            </ErrorNote>
          </div>
        ) : null}

        {busy ? (
          <div className="mt-4 flex items-center gap-2 text-sm text-muted">
            <span className="df-spinner" />
            <span>Analyzing frame by frame</span>
          </div>
        ) : (
          <div className="mt-4">
            <Button
              variant="primary"
              disabled={!file}
              onClick={() => file && onAnalyze(file)}
            >
              Run detection
            </Button>
          </div>
        )}
      </div>
    </Card>
  );
}
