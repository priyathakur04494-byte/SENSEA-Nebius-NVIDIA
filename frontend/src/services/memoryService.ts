/**
 * PERSONAL ACCESSIBILITY MEMORY SERVICE
 * ---------------------------------------
 * Persists speakers, important people, deadlines, and preferences to the
 * device using AsyncStorage, so this genuinely works offline with no backend
 * required. If you later want memory to sync across a user's devices, swap
 * the AsyncStorage calls for a backend (Firebase, Supabase, etc.) behind the
 * same function signatures.
 */

import AsyncStorage from "@react-native-async-storage/async-storage";
import {
  RememberedDeadline,
  RememberedPreference,
  Speaker,
} from "@/types";
import { colors } from "@/theme/colors";

const KEYS = {
  speakers: "memory:speakers",
  deadlines: "memory:deadlines",
  preferences: "memory:preferences",
};

async function readList<T>(key: string): Promise<T[]> {
  try {
    const raw = await AsyncStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T[]) : [];
  } catch {
    return [];
  }
}

async function writeList<T>(key: string, list: T[]): Promise<void> {
  await AsyncStorage.setItem(key, JSON.stringify(list));
}

// ---------- Speakers ----------

export async function getSpeakers(): Promise<Speaker[]> {
  return readList<Speaker>(KEYS.speakers);
}

export async function upsertSpeaker(speaker: Speaker): Promise<void> {
  const list = await getSpeakers();
  const idx = list.findIndex((s) => s.id === speaker.id);
  if (idx >= 0) list[idx] = speaker;
  else list.push(speaker);
  await writeList(KEYS.speakers, list);
}

export async function getOrCreateSpeaker(speakerId: string): Promise<Speaker> {
  const list = await getSpeakers();
  const existing = list.find((s) => s.id === speakerId);
  if (existing) return existing;

  const newSpeaker: Speaker = {
    id: speakerId,
    label: `Speaker ${list.length + 1}`,
    colorTag: colors.speakerPalette[list.length % colors.speakerPalette.length],
    isImportantPerson: false,
    lastSeen: Date.now(),
  };
  await upsertSpeaker(newSpeaker);
  return newSpeaker;
}

// ---------- Deadlines ----------

export async function getDeadlines(): Promise<RememberedDeadline[]> {
  const list = await readList<RememberedDeadline>(KEYS.deadlines);
  return list.sort((a, b) => new Date(a.dueDate).getTime() - new Date(b.dueDate).getTime());
}

export async function addDeadline(deadline: RememberedDeadline): Promise<void> {
  const list = await getDeadlines();
  list.push(deadline);
  await writeList(KEYS.deadlines, list);
}

export async function removeDeadline(id: string): Promise<void> {
  const list = await getDeadlines();
  await writeList(KEYS.deadlines, list.filter((d) => d.id !== id));
}

// ---------- Preferences ----------

export async function getPreferences(): Promise<RememberedPreference[]> {
  return readList<RememberedPreference>(KEYS.preferences);
}

export async function addPreference(pref: RememberedPreference): Promise<void> {
  const list = await getPreferences();
  list.push(pref);
  await writeList(KEYS.preferences, list);
}

export async function removePreference(id: string): Promise<void> {
  const list = await getPreferences();
  await writeList(KEYS.preferences, list.filter((p) => p.id !== id));
}

/**
 * Very lightweight deadline extraction from caption text — looks for common
 * date/day phrases. Good enough to demonstrate the "remembers deadlines"
 * feature; for production, replace with an LLM call that extracts
 * structured {description, dueDate} pairs from a transcript.
 */
export function extractPossibleDeadline(text: string): string | null {
  const pattern = /\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|tonight|next week)\b/i;
  return pattern.test(text) ? text : null;
}
