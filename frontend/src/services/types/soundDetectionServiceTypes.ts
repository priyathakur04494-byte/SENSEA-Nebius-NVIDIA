/**
 * SHARED SOUND DETECTION SERVICE CONTRACT
 * -----------------------------------------
 * Both the mock and real implementations must satisfy this interface so
 * AppContext can swap between them without touching call sites.
 */

import { AlertLevel, EnvironmentalSoundEvent } from "@/types";

export type SoundEventCallback = (event: EnvironmentalSoundEvent) => void;

export interface SoundDetectionService {
  /** Start listening for environmental sounds and call onEvent on each. */
  startDetection(onEvent: SoundEventCallback): void;

  /** Stop the active detection loop. Safe to call when not running. */
  stopDetection(): void;
}

/** Utility exported by both implementations. */
export function alertLevelWeight(level: AlertLevel): number {
  switch (level) {
    case "emergency": return 3;
    case "important": return 2;
    case "normal":    return 1;
  }
}
