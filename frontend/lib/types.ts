export type RiskStatus = "REAL" | "SUSPICIOUS" | "HIGH_RISK";

export type Feedback = "REAL" | "FAKE";

export interface FrameScore {
  timestamp: number;
  fake_probability: number;
}

export interface Analysis {
  id: number;
  filename: string;
  fake_probability: number;
  status: RiskStatus;
  suspicious_start: number | null;
  suspicious_end: number | null;
  frame_scores: FrameScore[];
  created_at: string;
  user_feedback: Feedback | null;
  has_video: boolean;
}

export interface HistoryItem {
  id: number;
  kind: "upload" | "live";
  filename: string;
  fake_probability: number;
  status: RiskStatus;
  created_at: string;
  has_video: boolean;
}

export interface LiveFrame {
  face_found: boolean;
  fake_probability: number | null;
  status: RiskStatus | "WAITING" | null;
  smoothed: number | null;
  frames_scored: number | null;
  frames_received: number | null;
  dropped: boolean;
  recent: FrameScore[];
}

export interface LiveSession {
  id: number;
  started_at: string;
  ended_at: string;
  duration_seconds: number;
  frames_scored: number;
  mean_probability: number;
  peak_probability: number;
  status: RiskStatus;
  timeline: FrameScore[];
  user_feedback: Feedback | null;
}

export interface Health {
  status: string;
  model_loaded: boolean;
}
