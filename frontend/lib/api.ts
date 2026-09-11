/** Thin client for the Phase 4 FastAPI backend (see backend/README.md). */

import type { Analysis, Health, HistoryItem } from "./types";

export const API_URL =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ?? "http://localhost:8000";

/** Raised for any non-2xx response, carrying FastAPI's `detail` when present. */
export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

/** Pull FastAPI's `{"detail": ...}` out of an error body, if it is there. */
function detailFrom(body: unknown, fallback: string): string {
  if (body && typeof body === "object" && "detail" in body) {
    const detail = (body as { detail: unknown }).detail;
    if (typeof detail === "string") return detail;
  }
  return fallback;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    cache: "no-store",
    ...init,
  });

  if (!response.ok) {
    let detail = `request failed with ${response.status}`;
    try {
      detail = detailFrom(await response.json(), detail);
    } catch {
      // non-JSON error body, keep the generic message
    }
    throw new ApiError(detail, response.status);
  }

  return response.json() as Promise<T>;
}

/**
 * Upload a video and wait for the full synchronous analysis.
 *
 * XHR rather than `fetch` so the upload half of the wait can be shown as a real
 * percentage; the detection half has no progress to report and is signalled by
 * `onProgress(1)`.
 */
export function analyzeVideo(
  file: File,
  options: { onProgress?: (fraction: number) => void; signal?: AbortSignal } = {},
): Promise<Analysis> {
  const { onProgress, signal } = options;

  return new Promise<Analysis>((resolve, reject) => {
    const body = new FormData();
    body.append("file", file);

    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_URL}/analyze`);
    xhr.responseType = "json";

    xhr.upload.addEventListener("progress", (event) => {
      if (event.lengthComputable) onProgress?.(event.loaded / event.total);
    });
    xhr.upload.addEventListener("load", () => onProgress?.(1));

    xhr.addEventListener("load", () => {
      const parsed: unknown =
        xhr.response ?? (xhr.responseText ? JSON.parse(xhr.responseText) : null);
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(parsed as Analysis);
      } else {
        reject(
          new ApiError(
            detailFrom(parsed, `request failed with ${xhr.status}`),
            xhr.status,
          ),
        );
      }
    });
    xhr.addEventListener("error", () =>
      reject(new ApiError("could not reach the backend", 0)),
    );
    xhr.addEventListener("abort", () =>
      reject(new DOMException("upload cancelled", "AbortError")),
    );

    signal?.addEventListener("abort", () => xhr.abort(), { once: true });
    xhr.send(body);
  });
}

export function getAnalysis(id: number): Promise<Analysis> {
  return request<Analysis>(`/analysis/${id}`);
}

/**
 * Source video for an analysis, served with Range support so the `<video>`
 * element can seek. Only meaningful when `analysis.has_video` is true.
 */
export function videoUrl(id: number): string {
  return `${API_URL}/analysis/${id}/video`;
}

export function getHistory(limit = 100): Promise<HistoryItem[]> {
  return request<HistoryItem[]>(`/history?limit=${limit}`);
}

export function getHealth(): Promise<Health> {
  return request<Health>("/health");
}
