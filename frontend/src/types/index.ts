// ---------- Core domain types ----------

export type AlertLevel = "normal" | "important" | "emergency";

export type SoundDirection = "left" | "center" | "right";

export type SupportedLanguage =
  | "en" // English
  | "hi" // Hindi
  | "ta" // Tamil
  | "te" // Telugu
  | "bn" // Bengali
  | "mr" // Marathi
  | "gu" // Gujarati
  | "kn" // Kannada
  | "ml" // Malayalam
  | "pa"; // Punjabi

export const LANGUAGE_LABELS: Record<SupportedLanguage, string> = {
  en: "English",
  hi: "हिन्दी (Hindi)",
  ta: "தமிழ் (Tamil)",
  te: "తెలుగు (Telugu)",
  bn: "বাংলা (Bengali)",
  mr: "मराठी (Marathi)",
  gu: "ગુજરાતી (Gujarati)",
  kn: "ಕನ್ನಡ (Kannada)",
  ml: "മലയാളം (Malayalam)",
  pa: "ਪੰਜਾਬੀ (Punjabi)",
};

/** A single utterance produced by the live captioning engine. */
export interface CaptionEntry {
  id: string;
  speakerId: string;
  originalText: string;
  simplifiedText?: string;
  translatedText?: string;
  language: SupportedLanguage;
  timestamp: number;
  isFinal: boolean;
}

/** A person the app has learned to recognize across sessions. */
export interface Speaker {
  id: string;
  label: string; // e.g. "Speaker 2" until named
  name?: string; // user-assigned name, e.g. "Mom"
  colorTag: string; // hex color used consistently in the UI
  isImportantPerson: boolean;
  notes?: string;
  lastSeen: number;
}

/** A remembered deadline extracted from conversation or added manually. */
export interface RememberedDeadline {
  id: string;
  description: string;
  dueDate: string; // ISO date
  sourceConversationId?: string;
  createdAt: number;
}

/** A user preference the memory system has learned. */
export interface RememberedPreference {
  id: string;
  category: string; // e.g. "food", "communication", "routine"
  detail: string;
  createdAt: number;
}

export interface EnvironmentalSoundEvent {
  id: string;
  soundType: "siren" | "alarm" | "horn" | "doorbell" | "knock" | "phone_ring" | "other";
  label: string;
  alertLevel: AlertLevel;
  direction: SoundDirection;
  confidence: number; // 0-1
  timestamp: number;
}

export interface ConversationSession {
  id: string;
  startedAt: number;
  endedAt?: number;
  captions: CaptionEntry[];
  participantIds: string[];
  summary?: string;
}
