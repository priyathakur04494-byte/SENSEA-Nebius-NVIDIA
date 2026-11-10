/**
 * REAL SOUND DETECTION SERVICE — Stub / Native Placeholder
 * ----------------------------------------------------------
 * This file is a placeholder for a hardware-backed environmental sound
 * classifier. The mock (mockSoundDetectionService.ts) is used by default.
 *
 * WHAT YOU NEED TO IMPLEMENT THIS:
 *
 * Sound *classification* (recognising "this is a siren" vs "this is a horn")
 * needs an audio ML model — Google's YAMNet or a custom TensorFlow Lite model
 * are the standard choices, run on-device via a native module. This requires
 * an Expo Development Build (not Expo Go) since it needs native TFLite
 * bindings — e.g. `react-native-fast-tflite`.
 *
 * Sound *direction* (left/right) needs stereo microphone input and an
 * amplitude/phase comparison between channels. A single phone mic cannot
 * do this reliably; this typically requires either:
 *   (a) A stereo external mic accessory
 *   (b) Two-mic hardware with a manufacturer API (some Android phones)
 *   (c) A connected pair of hearing aids / earbuds with directional mics
 * Treat true stereo direction detection as a hardware-dependent stretch goal.
 *
 * HOW TO ACTIVATE:
 * In AppContext.tsx swap the import:
 *   import soundDetectionService from "@/services/real/realSoundDetectionService";
 */

import { SoundDetectionService } from "@/services/types/soundDetectionServiceTypes";

const realSoundDetectionService: SoundDetectionService = {
  startDetection(_onEvent) {
    // TODO: initialise TFLite model, start microphone capture, and call
    // _onEvent with real EnvironmentalSoundEvent objects.
    console.warn(
      "[realSoundDetectionService] Not implemented — falling back to mock. " +
      "See src/services/real/realSoundDetectionService.ts for integration notes."
    );
  },

  stopDetection() {
    // TODO: stop mic capture and unload model.
  },
};

export default realSoundDetectionService;
