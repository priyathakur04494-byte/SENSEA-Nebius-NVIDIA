/**
 * ENVIRONMENTAL SOUND DETECTION SERVICE
 * --------------------------------------
 * WHAT THIS FILE DOES RIGHT NOW:
 * Simulates detection events (siren, alarm, horn, doorbell) at random
 * intervals with a direction and alert level, so the Home screen's alert
 * banner and history list have real events to render and you can test the
 * full alert-level UI (🟢 normal / 🟡 important / 🔴 emergency) end to end.
 *
 * WHAT YOU NEED TO PLUG IN FOR A REAL DEVICE:
 * Sound *classification* (recognizing "this is a siren" vs "this is a horn")
 * needs an audio ML model — Google's YAMNet or a custom TensorFlow Lite model
 * are the standard choices, run on-device via a native module (this requires
 * an Expo Development Build, not Expo Go, since it needs native TFLite
 * bindings — e.g. `react-native-fast-tflite`).
 *
 * Sound *direction* (left/right) needs stereo microphone input and an
 * amplitude/phase comparison between channels — a single phone mic cannot
 * do this reliably by itself; this typically requires either (a) a stereo
 * external mic accessory, or (b) two-mic hardware with a manufacturer API
 * (some Android phones expose this), or (c) approximating direction from
 * a connected pair of hearing aids / earbuds with directional mics, if the
 * target device supports it. Treat true stereo direction detection as a
 * hardware-dependent stretch goal, and be upfront with users about that
 * dependency rather than silently guessing.
 *
 * Swap `startMockDetection` for a real classifier pipeline that calls
 * `onEvent` with the same `EnvironmentalSoundEvent` shape.
 */

import { AlertLevel, EnvironmentalSoundEvent, SoundDirection } from "@/types";

type SoundEventCallback = (event: EnvironmentalSoundEvent) => void;

const SOUND_LIBRARY: {
  soundType: EnvironmentalSoundEvent["soundType"];
  label: string;
  alertLevel: AlertLevel;
}[] = [
  { soundType: "doorbell", label: "Doorbell", alertLevel: "normal" },
  { soundType: "knock", label: "Knock at the door", alertLevel: "normal" },
  { soundType: "phone_ring", label: "Phone ringing", alertLevel: "normal" },
  { soundType: "horn", label: "Car horn nearby", alertLevel: "important" },
  { soundType: "alarm", label: "Alarm sounding", alertLevel: "important" },
  { soundType: "siren", label: "Possible emergency: siren detected", alertLevel: "emergency" },
];

const DIRECTIONS: SoundDirection[] = ["left", "center", "right"];

let intervalHandle: ReturnType<typeof setInterval> | null = null;

function makeId(): string {
  return `sound_${Date.now()}_${Math.floor(Math.random() * 10000)}`;
}

export function startMockDetection(onEvent: SoundEventCallback) {
  stopMockDetection();

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
}

export function stopMockDetection() {
  if (intervalHandle) {
    clearInterval(intervalHandle);
    intervalHandle = null;
  }
}

export function alertLevelWeight(level: AlertLevel): number {
  switch (level) {
    case "emergency":
      return 3;
    case "important":
      return 2;
    case "normal":
      return 1;
  }
}
