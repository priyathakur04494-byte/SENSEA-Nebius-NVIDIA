/**
 * PER-MODE CONTENT
 * ----------------
 * SENSEA adapts to three contexts the user can pick at sign-up/log-in or
 * change later from Profile: Hospital, Workplace, Classroom (see AppMode in
 * @/types). This file is the single place that content lives, so every
 * screen/service that adapts imports from here instead of hard-coding its
 * own copy per mode.
 *
 * Nothing here changes layout, navigation, or which features exist — only
 * what those features show/listen for, per the request to keep the app
 * itself the same and have it "adapt" underneath.
 */

import type { Ionicons } from "@expo/vector-icons";
import { AlertLevel, AppMode, EnvironmentalSoundEvent } from "@/types";

export const MODES: AppMode[] = ["hospital", "workplace", "classroom"];

export const MODE_ICONS: Record<AppMode, keyof typeof Ionicons.glyphMap> = {
  hospital: "medkit-outline",
  workplace: "briefcase-outline",
  classroom: "school-outline",
};

export const MODE_DESCRIPTIONS: Record<AppMode, string> = {
  hospital: "Appointments and care team conversations — watches for medication, dosage, and emergency alarms.",
  workplace: "Meetings and calls — watches for deadlines, decisions, and action items.",
  classroom: "Lectures and classes — watches for assignments, exams, and announcements.",
};

export const MODE_TAGLINES: Record<AppMode, string> = {
  hospital: "Sense every conversation with your care team.",
  workplace: "Sense every conversation — what's happening, what matters, and what you missed.",
  classroom: "Sense every conversation in class, from lecture to homework.",
};

// ---------------------------------------------------------------------------
// Mock live-caption scripts (speechService.startMockStream). Same mechanism
// as before — just a different sample conversation per mode so the demo
// data actually matches the context the user picked.
// ---------------------------------------------------------------------------

export interface ModeUtterance {
  speakerId: string;
  text: string;
}

export const MODE_SAMPLE_UTTERANCES: Record<AppMode, ModeUtterance[]> = {
  workplace: [
    { speakerId: "speaker-1", text: "Hey, did you get a chance to look at the report I sent?" },
    { speakerId: "speaker-2", text: "Not yet, I'll review it after lunch, around 2pm today." },
    { speakerId: "speaker-1", text: "Okay, no rush, but the client meeting is Friday morning." },
    { speakerId: "speaker-2", text: "Got it, I'll have comments back to you by Thursday." },
    { speakerId: "speaker-1", text: "Sounds good. Also, happy birthday for next week, by the way!" },
  ],
  hospital: [
    { speakerId: "speaker-1", text: "Good morning, I'm Dr. Patel. How are you feeling today?" },
    { speakerId: "speaker-2", text: "A bit better, but I still have some pain, around a 3 out of 10." },
    { speakerId: "speaker-1", text: "Okay, we'll keep you on the current medication and check again this afternoon." },
    { speakerId: "speaker-2", text: "Should I take the next dose before or after lunch?" },
    { speakerId: "speaker-1", text: "Take it after lunch, around 2pm, and tell the nurse if the pain gets worse." },
  ],
  classroom: [
    { speakerId: "speaker-1", text: "Alright everyone, today we're covering chapter 7 on cell division." },
    { speakerId: "speaker-2", text: "Is the homework from chapter 6 still due tomorrow?" },
    { speakerId: "speaker-1", text: "Yes, hand it in at the start of class tomorrow, no extensions." },
    { speakerId: "speaker-2", text: "Will this be on the midterm next week?" },
    { speakerId: "speaker-1", text: "Good question, sections 7.1 through 7.3 will be on the midterm Friday." },
  ],
};

// ---------------------------------------------------------------------------
// Mock environmental sound library (soundDetectionService.startMockDetection).
// Reuses the existing EnvironmentalSoundEvent["soundType"] union — only the
// label/alert level differ per mode — so AlertBanner and the rest of the
// Sound Alerts UI need no changes at all.
// ---------------------------------------------------------------------------

export interface ModeSoundTemplate {
  soundType: EnvironmentalSoundEvent["soundType"];
  label: string;
  alertLevel: AlertLevel;
}

export const MODE_SOUND_LIBRARY: Record<AppMode, ModeSoundTemplate[]> = {
  workplace: [
    { soundType: "doorbell", label: "Doorbell", alertLevel: "normal" },
    { soundType: "knock", label: "Knock at the door", alertLevel: "normal" },
    { soundType: "phone_ring", label: "Phone ringing", alertLevel: "normal" },
    { soundType: "horn", label: "Car horn nearby", alertLevel: "important" },
    { soundType: "alarm", label: "Alarm sounding", alertLevel: "important" },
    { soundType: "siren", label: "Possible emergency: siren detected", alertLevel: "emergency" },
  ],
  hospital: [
    { soundType: "alarm", label: "IV pump beeping", alertLevel: "normal" },
    { soundType: "knock", label: "Knock — someone at the door", alertLevel: "normal" },
    { soundType: "phone_ring", label: "Nurse call button", alertLevel: "important" },
    { soundType: "alarm", label: "Vitals monitor alarm", alertLevel: "important" },
    { soundType: "siren", label: "Code alarm — possible emergency", alertLevel: "emergency" },
  ],
  classroom: [
    { soundType: "alarm", label: "Class bell", alertLevel: "normal" },
    { soundType: "knock", label: "Knock at the classroom door", alertLevel: "normal" },
    { soundType: "phone_ring", label: "Phone ringing nearby", alertLevel: "normal" },
    { soundType: "horn", label: "Bus horn outside", alertLevel: "important" },
    { soundType: "siren", label: "Fire alarm", alertLevel: "emergency" },
  ],
};

// ---------------------------------------------------------------------------
// Extra "important" keywords layered onto the base set in contextInsights.ts,
// so the AI Context panel flags the things that actually matter per context.
// ---------------------------------------------------------------------------

export const MODE_IMPORTANT_KEYWORDS: Record<AppMode, string[]> = {
  workplace: ["budget", "client", "proposal", "invoice"],
  hospital: [
    "medication",
    "dose",
    "dosage",
    "prescription",
    "appointment",
    "surgery",
    "discharge",
    "allergic",
    "pain",
    "vitals",
    "doctor",
    "nurse",
  ],
  classroom: ["homework", "assignment", "exam", "quiz", "midterm", "final", "due", "project", "grade", "syllabus"],
};

// ---------------------------------------------------------------------------
// "Ask SENSEA" suggested questions, per mode.
// ---------------------------------------------------------------------------

export const MODE_SUGGESTED_QUESTIONS: Record<AppMode, string[]> = {
  workplace: [
    "What did I miss?",
    "What decisions were made?",
    "Did anyone mention my name?",
    "Summarize the last few minutes.",
  ],
  hospital: [
    "What did I miss?",
    "What did the doctor say to do?",
    "When is my next dose?",
    "Summarize what was said.",
  ],
  classroom: [
    "What did I miss?",
    "What's the homework?",
    "Is anything due this week?",
    "Summarize the lecture so far.",
  ],
};

// ---------------------------------------------------------------------------
// Sound Alerts screen copy — legend built from the mode's own sound library
// above, so it always describes what's actually in the mock stream.
// ---------------------------------------------------------------------------

export const MODE_ALERT_SUBTITLE: Record<AppMode, string> = {
  hospital: "Your device listens for important sounds around you in care settings — monitors, call buttons, and alarms — and alerts you here with an urgency level and the direction it came from.",
  workplace: "Your device listens for important sounds around you — sirens, alarms, horns, doorbells — and alerts you here with an urgency level and the direction it came from.",
  classroom: "Your device listens for important sounds around you in class — bells, alarms, knocks — and alerts you here with an urgency level and the direction it came from.",
};

export function legendExamplesForLevel(mode: AppMode, level: AlertLevel): string {
  const examples = MODE_SOUND_LIBRARY[mode]
    .filter((s) => s.alertLevel === level)
    .map((s) => s.label.toLowerCase());
  return examples.length ? examples.join(", ") : "—";
}
