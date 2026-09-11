/** Mirrors the Pydantic schemas in `backend/schemas.py`. */

export type RiskStatus = "REAL" | "SUSPICIOUS" | "HIGH_RISK";

/** What the viewer says a video really was, independent of the prediction. */
export type Feedback = "REAL" | "FAKE";

export interface FrameScore {
  timestamp: number;
  fake_probability: number;
}

/** `POST /analyze` and `GET /analysis/{id}`. */
export interface Analysis {
  id: number;
  filename: string;
  fake_probability: number;
  status: RiskStatus;
  suspicious_start: number | null;
  suspicious_end: number | null;
  frame_scores: FrameScore[];
  created_at: string;
  /** Null until the viewer reports what the video really was; optional. */
  user_feedback: Feedback | null;
  /** False when the source video was never kept, or is gone from disk. */
  has_video: boolean;
}

/** One row of `GET /history`, without frame scores. */
export interface HistoryItem {
  id: number;
  filename: string;
  fake_probability: number;
  status: RiskStatus;
  created_at: string;
  has_video: boolean;
}

export interface Health {
  status: string;
  model_loaded: boolean;
}
