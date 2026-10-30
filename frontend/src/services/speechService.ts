/**
 * SPEECH-TO-TEXT SERVICE
 * ----------------------
 * WHAT THIS FILE DOES RIGHT NOW:
 * Provides a working streaming-caption simulation so every screen in the app
 * (captions, memory, summary, Q&A) has real data to operate on out of the box.
 *
 * WHAT YOU NEED TO PLUG IN FOR A REAL DEVICE:
 * True real-time speech-to-text requires either:
 *   (a) A cloud streaming API — Google Cloud Speech-to-Text (streaming),
 *       Azure Speech SDK, or AssemblyAI real-time — reached over a WebSocket,
 *       fed by microphone PCM frames captured via `expo-audio`'s
 *       AudioRecorder in streaming mode (the SDK 55+ replacement for the
 *       now-removed `expo-av`), OR
 *   (b) An on-device model (e.g. whisper.cpp via a native module, or
 *       Android's on-device SpeechRecognizer / iOS's SFSpeechRecognizer
 *       through a native bridge — Expo's managed workflow can't call these
 *       directly, so this would need an Expo Development Build with a
 *       custom native module).
 *
 * Speaker diarization (telling speakers apart) additionally needs either a
 * diarization-capable cloud API (Google/Azure both support this) or a
 * separate on-device embedding model — it's not something a phone mic alone
 * gives you for free.
 *
 * Swap `startMockStream` below for a real implementation that calls your
 * chosen provider and forwards results through the same `onCaption` callback
 * shape so the rest of the app needs no changes.
 */

import { CaptionEntry, SupportedLanguage } from "@/types";

type CaptionCallback = (entry: CaptionEntry) => void;

const SAMPLE_UTTERANCES: { speakerId: string; text: string }[] = [
  { speakerId: "speaker-1", text: "Hey, did you get a chance to look at the report I sent?" },
  { speakerId: "speaker-2", text: "Not yet, I'll review it after lunch, around 2pm today." },
  { speakerId: "speaker-1", text: "Okay, no rush, but the client meeting is Friday morning." },
  { speakerId: "speaker-2", text: "Got it, I'll have comments back to you by Thursday." },
  { speakerId: "speaker-1", text: "Sounds good. Also, happy birthday for next week, by the way!" },
];

let intervalHandle: ReturnType<typeof setInterval> | null = null;
let utteranceIndex = 0;

function makeId(): string {
  return `cap_${Date.now()}_${Math.floor(Math.random() * 10000)}`;
}

/**
 * Starts a mock live-caption stream, emitting one utterance every few
 * seconds. Mirrors the shape a real streaming transcription callback would
 * have: interim (isFinal: false) followed by a final revision.
 */
export function startMockStream(onCaption: CaptionCallback, language: SupportedLanguage = "en") {
  stopStream();
  utteranceIndex = 0;

  intervalHandle = setInterval(() => {
    if (utteranceIndex >= SAMPLE_UTTERANCES.length) {
      utteranceIndex = 0; // loop for demo purposes
    }
    const sample = SAMPLE_UTTERANCES[utteranceIndex];
    const id = makeId();
    const timestamp = Date.now();

    // Emit an interim result first (simulates partial recognition)
    onCaption({
      id,
      speakerId: sample.speakerId,
      originalText: sample.text.slice(0, Math.ceil(sample.text.length * 0.5)),
      language,
      timestamp,
      isFinal: false,
    });

    // Then the final result shortly after
    setTimeout(() => {
      onCaption({
        id,
        speakerId: sample.speakerId,
        originalText: sample.text,
        language,
        timestamp,
        isFinal: true,
      });
    }, 700);

    utteranceIndex += 1;
  }, 3200);
}

export function stopStream() {
  if (intervalHandle) {
    clearInterval(intervalHandle);
    intervalHandle = null;
  }
}

/**
 * Text simplification for complex speech. This mock version applies
 * heuristics — plain-word substitution, dropping filler openers, and
 * splitting long/compound sentences into short ones — so the "Simplify"
 * toggle produces a visibly different, shorter sentence for typical speech,
 * not just for a narrow set of formal phrases. For production, swap this for
 * a call to an LLM (e.g. Claude via the Anthropic API) with a prompt like:
 * "Rewrite this sentence in simple, plain language for someone who benefits
 * from reduced sentence complexity. Keep the meaning exact."
 */
const WORD_SUBSTITUTIONS: [RegExp, string][] = [
  [/\bapproximately\b/gi, "about"],
  [/\butilize[sd]?\b/gi, "use"],
  [/\bprior to\b/gi, "before"],
  [/\bin order to\b/gi, "to"],
  [/\bin the event that\b/gi, "if"],
  [/\bdue to the fact that\b/gi, "because"],
  [/\bat this point in time\b/gi, "now"],
  [/\bin the near future\b/gi, "soon"],
  [/\bwith regard(?:s|ing) to\b/gi, "about"],
  [/\bsubsequent to\b/gi, "after"],
  [/\ba chance to\b/gi, "time to"],
  [/\badditionally\b/gi, "also"],
  [/\bhowever\b/gi, "but"],
  [/\btherefore\b/gi, "so"],
  [/\bnevertheless\b/gi, "still"],
  [/\bfurthermore\b/gi, "also"],
  [/\bconsequently\b/gi, "so"],
  [/\bcommence[sd]?\b/gi, "start"],
  [/\bterminate[sd]?\b/gi, "end"],
  [/\bpurchase[sd]?\b/gi, "buy"],
  [/\bassist(?:ance)?\b/gi, "help"],
  [/\bobtain(?:ed)?\b/gi, "get"],
  [/\bsufficient\b/gi, "enough"],
  [/\bdemonstrate[sd]?\b/gi, "show"],
  [/\bfacilitate[sd]?\b/gi, "help"],
  [/\bendeavor(?:ed)?\b/gi, "try"],
  [/\bnumerous\b/gi, "many"],
  [/\bregarding\b/gi, "about"],
];

// Filler openers that add no meaning — dropped so the sentence starts on the
// actual content.
const FILLER_OPENERS = /^(well|so|actually|basically|you know|okay|ok|got it|no rush|sounds good)\s*,\s*/i;

// Splits a sentence into shorter ones at a comma immediately followed by a
// coordinating conjunction (", but" / ", and" / ", so" / ", because"), which
// is usually where a compound sentence can cleanly become two simple ones.
function splitCompoundSentence(text: string): string {
  return text.replace(/,\s+(but|and|so|because)\s+/gi, (_match, conj: string) => {
    const capitalized = conj.charAt(0).toUpperCase() + conj.slice(1).toLowerCase();
    return `. ${capitalized === "And" || capitalized === "So" ? capitalized : capitalized} `;
  });
}

export function simplifyText(text: string): string {
  let result = text;

  for (const [pattern, replacement] of WORD_SUBSTITUTIONS) {
    result = result.replace(pattern, replacement);
  }

  result = result.replace(/,\s*(which|who|that)\s/gi, ". This ");
  result = splitCompoundSentence(result);
  result = result.replace(FILLER_OPENERS, "");
  result = result.replace(/\s{2,}/g, " ").trim();

  // Capitalize the first letter after any splits/trims above.
  if (result.length) {
    result = result.charAt(0).toUpperCase() + result.slice(1);
  }

  return result;
}
