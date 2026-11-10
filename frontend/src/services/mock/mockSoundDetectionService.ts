/**
 * MOCK SOUND DETECTION SERVICE
 * ------------------------------
 * Simulates detection events (siren, alarm, horn, doorbell) at random
 * intervals with a direction and alert level, so the Home screen's alert
 * banner and history list have real events to render and you can test the
 * full alert-level UI (🟢 normal / 🟡 important / 🔴 emergency) end to end.
 *
 * HOW TO SWITCH TO THE REAL IMPLEMENTATION:
 * In AppContext.tsx (or wherever you import this), change:
 *   import soundDetectionService from "@/services/mock/mockSoundDetectionService";
 * to:
 *   import soundDetectionService from "@/services/real/realSoundDetectionService";
 * No other files need changing — both satisfy the SoundDetectionService interface.
 *
 * WHAT YOU NEED FOR A REAL DEVICE:
 * Sound *classification* needs an audio ML model (YAMNet / TensorFlow Lite)
 * run on-device via a native module (requires Expo Development Build, not
 * Expo Go — e.g. `react-native-fast-tflite`).
 *
 * Sound *direction* needs stereo microphone input. See the full notes in
 * the real implementation stub at @/services/real/realSoundDetectionService.ts.
 */

import { SoundDetectionService, alertLevelWeight } from "@/services/types/soundDetectionServiceTypes";
import { AlertLevel, EnvironmentalSoundEvent, SoundDirection } from "@/types";

export { alertLevelWeight };

const SOUND_LIBRARY: {
  soundType: EnvironmentalSoundEvent["soundType"];
  label: string;
  alertLevel: AlertLevel;
}[] = [
  { soundType: "doorbell",   label: "Doorbell",                    alertLevel: "normal" },
  { soundType: "knock",      label: "Knock at the door",           alertLevel: "normal" },
  { soundType: "phone_ring", label: "Phone ringing",               alertLevel: "normal" },
  { soundType: "horn",       label: "Car horn nearby",             alertLevel: "important" },
  { soundType: "alarm",      label: "Alarm sounding",              alertLevel: "important" },
  { soundType: "siren",      label: "Possible emergency: siren detected", alertLevel: "emergency" },
];

const DIRECTIONS: SoundDirection[] = ["left", "center", "right"];

let intervalHandle: ReturnType<typeof setInterval> | null = null;

function makeId(): string {
  return `sound_${Date.now()}_${Math.floor(Math.random() * 10000)}`;
}

const mockSoundDetectionService: SoundDetectionService = {
  startDetection(onEvent) {
    mockSoundDetectionService.stopDetection();

    intervalHandle = setInterval(() => {
      const template = SOUND_LIBRARY[Math.floor(Math.random() * SOUND_LIBRARY.length)];
      const direction = DIRECTIONS[Math.floor(Math.random() * DIRECTIONS.length)];

      onEvent({
        id: makeId(),
        soundType: template.soundType,
        label: template.label,
        alertLevel: template.alertLevel,
        direction,
        confidence: 0.75 + Math.random() * 0.24,
        timestamp: Date.now(),
      });
    }, 9000);
  },

  stopDetection() {
    if (intervalHandle) {
      clearInterval(intervalHandle);
      intervalHandle = null;
    }
  },
};

export default mockSoundDetectionService;
