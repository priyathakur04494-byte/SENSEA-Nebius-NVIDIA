/**
 * SHARED SPEECH SERVICE CONTRACT
 * --------------------------------
 * Both the mock and real implementations must satisfy this interface so
 * AppContext and any screen can swap between them without touching call sites.
 */

import { CaptionEntry, SupportedLanguage } from "@/types";

export type CaptionCallback = (entry: CaptionEntry) => void;

export interface SpeechService {
  /**
   * Start streaming captions. Calls onCaption with interim (isFinal: false)
   * and final (isFinal: true) results.
   */
  startStream(onCaption: CaptionCallback, language?: SupportedLanguage): void;

  /** Stop an active stream. Safe to call when no stream is running. */
  stopStream(): void;
}
