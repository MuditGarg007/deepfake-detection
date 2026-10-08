import type {
  Analysis,
  Feedback,
  Health,
  HistoryItem,
  LiveFrame,
  LiveSession,
} from "./types";

export const API_URL =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ??
  "http://localhost:8000";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export function errorMessage(error: unknown): string {
  if (error instanceof ApiError && error.status === 0) {
    return `Could not reach the backend at ${API_URL}. Is it running?`;
  }
  return error instanceof Error ? error.message : "Something went wrong";
}

function detailFrom(body: unknown, fallback: string): string {
  if (body && typeof body === "object" && "detail" in body) {
    const detail = (body as { detail: unknown }).detail;
    if (typeof detail === "string") return detail;
  }
  return fallback;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_URL}${path}`, { cache: "no-store", ...init });
  } catch {
    throw new ApiError("could not reach the backend", 0);
  }

  if (!response.ok) {
    let detail = `request failed with ${response.status}`;
    try {
      detail = detailFrom(await response.json(), detail);
    } catch {}
    throw new ApiError(detail, response.status);
  }

  return response.json() as Promise<T>;
}

export function analyzeVideo(
  file: File,
  onProgress?: (fraction: number) => void,
): Promise<Analysis> {
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
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(xhr.response as Analysis);
      } else {
        reject(
          new ApiError(
            detailFrom(xhr.response, `request failed with ${xhr.status}`),
            xhr.status,
          ),
        );
      }
    });
    xhr.addEventListener("error", () =>
      reject(new ApiError("could not reach the backend", 0)),
    );

    xhr.send(body);
  });
}

export function getAnalysis(id: number): Promise<Analysis> {
  return request<Analysis>(`/analysis/${id}`);
}

export function frameUrl(id: number): string {
  return `${API_URL}/analysis/${id}/frame`;
}

export function sendFeedback(
  id: number,
  label: Feedback | null,
): Promise<Analysis> {
  return request<Analysis>(`/analysis/${id}/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ label }),
  });
}

export function rerunAnalysis(id: number): Promise<Analysis> {
  return request<Analysis>(`/analysis/${id}/rerun`, { method: "POST" });
}

export async function startLive(): Promise<string> {
  const { session_id } = await request<{ session_id: string }>("/live/start", {
    method: "POST",
  });
  return session_id;
}

export function sendLiveFrame(
  sessionId: string,
  frame: Blob,
): Promise<LiveFrame> {
  return request<LiveFrame>(`/live/${sessionId}/frame`, {
    method: "POST",
    headers: { "Content-Type": "image/jpeg" },
    body: frame,
  });
}

export function stopLive(sessionId: string): Promise<LiveSession> {
  return request<LiveSession>(`/live/${sessionId}/stop`, { method: "POST" });
}

export function getLiveSession(id: number): Promise<LiveSession> {
  return request<LiveSession>(`/live/sessions/${id}`);
}

export function getHistory(limit = 50): Promise<HistoryItem[]> {
  return request<HistoryItem[]>(`/history?limit=${limit}`);
}

export function getHealth(): Promise<Health> {
  return request<Health>("/health");
}
