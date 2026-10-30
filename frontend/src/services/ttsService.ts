/**
 * TEXT-TO-SPEECH SERVICE
 * ------------------------
 * This one works for real out of the box — `expo-speech` speaks on-device
 * with no API key or network needed, which is exactly right for the
 * "type a message, app speaks it aloud" two-way communication feature.
 */

import * as Speech from "expo-speech";
import { SupportedLanguage } from "@/types";

// Maps our app's language codes to BCP-47 locale codes expo-speech expects.
const LOCALE_MAP: Record<SupportedLanguage, string> = {
  en: "en-IN",
  hi: "hi-IN",
  ta: "ta-IN",
  te: "te-IN",
  bn: "bn-IN",
  mr: "mr-IN",
  gu: "gu-IN",
  kn: "kn-IN",
  ml: "ml-IN",
  pa: "pa-IN",
};

export function speak(text: string, language: SupportedLanguage = "en") {
  if (!text.trim()) return;
  Speech.stop();
  Speech.speak(text, {
    language: LOCALE_MAP[language],
    pitch: 1.0,
    rate: 0.95,
  });
}

export function stopSpeaking() {
  Speech.stop();
}
