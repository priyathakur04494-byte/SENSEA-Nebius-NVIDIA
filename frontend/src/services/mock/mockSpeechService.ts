/**
 * MOCK SPEECH SERVICE
 * --------------------
 * Simulates a live-caption stream by emitting pre-canned utterances at a
 * fixed interval, so every screen in the app has real data to operate on
 * out of the box — no backend, mic permission, or network needed.
 *
 * HOW TO SWITCH TO THE REAL IMPLEMENTATION:
 * In AppContext.tsx (or wherever you import this), change:
 *   import speechService from "@/services/mock/mockSpeechService";
 * to:
 *   import speechService from "@/services/real/realSpeechService";
 * No other files need changing — both satisfy the SpeechService interface.
 *
 * TEXT SIMPLIFICATION is kept here (not in the interface) because it's a
 * pure text utility used by the UI independently of the stream provider.
 */

import { SpeechService } from "@/services/types/speechServiceTypes";
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

const mockSpeechService: SpeechService = {
  startStream(onCaption: CaptionCallback, _language: SupportedLanguage = "en") {
    mockSpeechService.stopStream();
    utteranceIndex = 0;

    intervalHandle = setInterval(() => {
      if (utteranceIndex >= SAMPLE_UTTERANCES.length) {
        utteranceIndex = 0;
      }
      const sample = SAMPLE_UTTERANCES[utteranceIndex];
      const id = makeId();
      const timestamp = Date.now();

      // Emit interim result (simulates partial recognition)
      onCaption({
        id,
        speakerId: sample.speakerId,
        originalText: sample.text.slice(0, Math.ceil(sample.text.length * 0.5)),
        language: _language,
        timestamp,
        isFinal: false,
      });

      // Then the final result shortly after
      setTimeout(() => {
        onCaption({
          id,
          speakerId: sample.speakerId,
          originalText: sample.text,
          language: _language,
          timestamp,
          isFinal: true,
        });
      }, 700);

      utteranceIndex += 1;
    }, 3200);
  },

  stopStream() {
    if (intervalHandle) {
      clearInterval(intervalHandle);
      intervalHandle = null;
    }
  },
};

export default mockSpeechService;

// ── Text simplification (pure utility, not part of the stream interface) ───

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

const FILLER_OPENERS =
  /^(well|so|actually|basically|you know|okay|ok|got it|no rush|sounds good)\s*,\s*/i;

function splitCompoundSentence(text: string): string {
  return text.replace(/,\s+(but|and|so|because)\s+/gi, (_match, conj: string) => {
    const capitalized = conj.charAt(0).toUpperCase() + conj.slice(1).toLowerCase();
    return `. ${capitalized} `;
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
  if (result.length) {
    result = result.charAt(0).toUpperCase() + result.slice(1);
  }
  return result;
}
