/**
 * TRANSLATION SERVICE
 * --------------------
 * Tries the real Bhashini API first (see bhashiniClient.ts and
 * src/config/bhashiniConfig.ts for setup). If Bhashini isn't configured yet,
 * or a call fails (no network, bad credentials, API hiccup), this falls back
 * to a tiny built-in phrase dictionary so the app still demonstrates
 * translation working end to end, plus a clearly-marked placeholder for
 * anything outside that dictionary.
 *
 * TO GO LIVE: fill in real credentials in src/config/bhashiniConfig.ts.
 * That's the only file you need to touch — this service and everything
 * that calls it (captions, type-to-speak) needs no further changes.
 */

import { SupportedLanguage } from "@/types";
import { bhashiniTranslate } from "@/services/bhashiniClient";
import { isBhashiniConfigured } from "@/config/bhashiniConfig";

// A small on-device phrase table used only as a fallback when Bhashini
// isn't configured or a call fails, so the demo still works with zero
// network access.
const MOCK_DICTIONARY: Record<string, Partial<Record<SupportedLanguage, string>>> = {
  "Hey, did you get a chance to look at the report I sent?": {
    hi: "अरे, क्या आपने मेरी भेजी रिपोर्ट देखी?",
    ta: "ஏய், நான் அனுப்பிய அறிக்கையை பார்த்தீர்களா?",
    te: "హే, నేను పంపిన నివేదికను చూశారా?",
  },
  "Not yet, I'll review it after lunch, around 2pm today.": {
    hi: "अभी नहीं, मैं आज दोपहर के भोजन के बाद, लगभग 2 बजे इसे देखूंगा।",
    ta: "இன்னும் இல்லை, மதிய உணவுக்குப் பிறகு, சுமார் 2 மணிக்கு பார்ப்பேன்.",
    te: "ఇంకా లేదు, భోజనం తర్వాత, దాదాపు 2 గంటలకు చూస్తాను.",
  },
  "Okay, no rush, but the client meeting is Friday morning.": {
    hi: "ठीक है, जल्दी नहीं है, लेकिन क्लाइंट मीटिंग शुक्रवार सुबह है।",
    ta: "சரி, அவசரம் இல்லை, ஆனால் வாடிக்கையாளர் கூட்டம் வெள்ளி காலை.",
    te: "సరే, తొందర లేదు, కానీ క్లయింట్ మీటింగ్ శుక్రవారం ఉదయం.",
  },
  "Got it, I'll have comments back to you by Thursday.": {
    hi: "समझ गया, मैं गुरुवार तक आपको टिप्पणियाँ भेज दूंगा।",
    ta: "சரி, வியாழக்கிழமைக்குள் கருத்துகளை அனுப்புவேன்.",
    te: "అర్థమైంది, గురువారం లోపు కామెంట్స్ పంపుతాను.",
  },
  "Sounds good. Also, happy birthday for next week, by the way!": {
    hi: "ठीक है। और हाँ, अगले हफ्ते जन्मदिन की अग्रिम शुभकामनाएं!",
    ta: "சரி. அடுத்த வாரம் பிறந்தநாள் வாழ்த்துக்கள் முன்கூட்டியே!",
    te: "సరే. వచ్చే వారం పుట్టినరోజు శుభాకాంక్షలు ముందుగానే!",
  },
};

/**
 * True if `translateText` actually reached a real translation (Bhashini,
 * or a mock-dictionary hit) rather than returning the "not available yet"
 * placeholder. Callers (like the type-to-speak screen) use this to decide
 * whether it's safe to speak the result aloud.
 */
export function isPendingPlaceholder(text: string): boolean {
  return /^\[[A-Z]{2} translation pending API integration\]/.test(text);
}

export async function translateText(
  text: string,
  targetLanguage: SupportedLanguage,
  sourceLanguage: SupportedLanguage = "en"
): Promise<string> {
  if (targetLanguage === sourceLanguage) return text;

  if (isBhashiniConfigured()) {
    try {
      return await bhashiniTranslate(text, sourceLanguage, targetLanguage);
    } catch (err) {
      // Fall through to the mock dictionary below rather than breaking the
      // caption/speech flow over a network hiccup.
      console.warn("Bhashini translation failed, falling back to mock:", err);
    }
  }

  const entry = MOCK_DICTIONARY[text];
  if (entry && entry[targetLanguage]) {
    return entry[targetLanguage] as string;
  }

  // Fallback for text outside the demo dictionary and without a configured
  // API — clearly marked so it's never mistaken for a real translation.
  return `[${targetLanguage.toUpperCase()} translation pending API integration] ${text}`;
}
