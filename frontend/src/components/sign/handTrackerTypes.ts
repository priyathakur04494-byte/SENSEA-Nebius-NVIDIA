import { CustomSign } from "@/services/customSignStore";

export interface DetectedSign {
  id: string;
  label: string;
  note?: string;
  confidence: number;
  custom?: boolean;
  motion?: boolean;
}

export type TrackerState = "idle" | "loading" | "running" | "denied" | "error";

/** Messages coming from the camera page. */
export type TrackerMessage =
  | { type: "ready" }
  | { type: "status"; state: TrackerState; message: string }
  | { type: "frame"; hands: { label: string | null; confidence: number; side?: string }[]; current: DetectedSign | null }
  | { type: "confirmed"; sign: DetectedSign }
  | { type: "teach-progress"; phase: "countdown" | "record"; take: number; takes: number; count?: number }
  | { type: "taught"; sign: CustomSign }
  | { type: "teach-failed"; reason: string };

/** Messages sent to the camera page. */
export type HostMessage =
  | { type: "init"; customSigns: CustomSign[]; facing: "user" | "environment" }
  | { type: "set-custom"; customSigns: CustomSign[] }
  | { type: "teach"; id: string; label: string }
  | { type: "cancel-teach" }
  | { type: "camera"; facing: "user" | "environment" };

export interface HandTrackerHandle {
  send: (m: HostMessage) => void;
}

export interface HandTrackerViewProps {
  onMessage: (m: TrackerMessage) => void;
  /** Called once with a function that sends messages to the camera page. */
  onHandle: (h: HandTrackerHandle) => void;
  height: number;
}
