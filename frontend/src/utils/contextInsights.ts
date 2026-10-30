/**
 * UI-LAYER CONTEXT INSIGHTS
 * -------------------------
 * These are lightweight, local, extractive helpers — the same demo-quality
 * approach already used by the app's summarize()/answerQuestion() logic —
 * that turn the existing caption transcript into the sections the "AI
 * Context" panel displays (Live Summary, Important Points, Decisions,
 * You May Have Missed, Ask SENSEA).
 *
 * They run entirely on-device against data already produced by the real
 * speech/translation/memory services and do NOT call any network service or
 * invent new backend behavior. For production, swap the bodies of these
 * functions for calls to an LLM (e.g. the Anthropic API), sending the
 * transcript as context — every screen that consumes this module can stay
 * exactly as it is.
 */

import { CaptionEntry, Speaker } from "@/types";

export interface TranscriptLine {
  id: string;
  speakerLabel: string;
  text: string;
  timestamp: number;
}

const DECISION_PATTERN =
  /\b(agreed|agree|let'?s (do|go with|submit|move)|sounds good|deal|confirmed|decided|works for (everyone|me|us)|okay,? let'?s)\b/i;

const IMPORTANT_PATTERN =
  /\b(\d{1,2}(:\d{2})?\s?(am|pm)|deadline|friday|monday|tuesday|wednesday|thursday|saturday|sunday|tomorrow|tonight|next week|meeting|submit|report|urgent|asap)\b/i;

export function toTranscriptLines(captions: CaptionEntry[], speakers: Record<string, Speaker>): TranscriptLine[] {
  return captions
    .filter((c) => c.isFinal)
    .map((c) => ({
      id: c.id,
      speakerLabel: speakers[c.speakerId]?.name ?? speakers[c.speakerId]?.label ?? "Speaker",
      text: c.originalText,
      timestamp: c.timestamp,
    }));
}

/** One short sentence describing what the conversation currently looks like. */
export function buildLiveSummary(lines: TranscriptLine[]): string {
  if (lines.length === 0) return "Waiting for the conversation to start.";
  const speakerCount = new Set(lines.map((l) => l.speakerLabel)).size;
  const latest = lines[lines.length - 1];
  return `${speakerCount > 1 ? `${speakerCount} people are` : `${latest.speakerLabel} is`} talking. Most recent: "${latest.text}"`;
}

/** Lines that mention a date, time, or otherwise sound worth flagging. */
export function buildImportantPoints(lines: TranscriptLine[]): TranscriptLine[] {
  return lines.filter((l) => IMPORTANT_PATTERN.test(l.text));
}

/** Lines that sound like the group settled on something. */
export function buildDecisions(lines: TranscriptLine[]): TranscriptLine[] {
  return lines.filter((l) => DECISION_PATTERN.test(l.text));
}

/** Lines the user hasn't acknowledged yet (see MissedContext's "Mark as caught up"). */
export function buildMissed(lines: TranscriptLine[], acknowledgedCount: number): TranscriptLine[] {
  if (acknowledgedCount >= lines.length) return [];
  // Leave the single most recent line out of "missed" — that's simply the
  // live conversation, not something the user failed to catch.
  return lines.slice(acknowledgedCount, Math.max(acknowledgedCount, lines.length - 1));
}

export function answerFromTranscript(question: string, lines: TranscriptLine[]): string {
  const q = question.toLowerCase();

  if (/mention(ed)? my name|about me/.test(q)) {
    return "No mention of your name was picked up in this conversation so far.";
  }
  if (/what.*(miss|missed)/.test(q)) {
    const decisions = buildDecisions(lines);
    if (decisions.length) return `You may have missed: "${decisions[decisions.length - 1].text}"`;
    return "Nothing important looks like it slipped by — you're caught up.";
  }
  if (/decision|decide|agreed/.test(q)) {
    const decisions = buildDecisions(lines);
    return decisions.length
      ? decisions.map((d) => `${d.speakerLabel}: "${d.text}"`).join("\n")
      : "No clear decision has come up yet.";
  }

  const keywords = q.split(/\s+/).filter((w) => w.length > 3);
  const match = [...lines].reverse().find((l) => keywords.some((k) => l.text.toLowerCase().includes(k)));
  if (match) return `${match.speakerLabel} said: "${match.text}"`;
  return "I couldn't find anything in the conversation so far that answers that. Try asking again once more has been said.";
}

export const SUGGESTED_QUESTIONS = [
  "What did I miss?",
  "What decisions were made?",
  "Did anyone mention my name?",
  "Summarize the last few minutes.",
];
