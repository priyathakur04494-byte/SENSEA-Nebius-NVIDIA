/**
 * TRANSLATION SERVICE
 * -------------------
 * Bridges live-caption / type-to-speak text to real translated text.
 *
 * Priority chain:
 *   1. SENSEA backend  → POST /api/translation (IndicTrans2 / Nemotron)
 *   2. Free web translation → direct translation fallback (no API key needed)
 *   3. Bhashini API    → if configured with credentials
 *   4. Mock dictionary → phrase-level fallback for demo phrases
 *   5. Placeholder     → clearly-marked "[XX translation pending…]" string
 */

import { SupportedLanguage } from "@/types";
import { translateTextBackend } from "@/services/backendApi";
import { bhashiniTranslate } from "@/config/bhashiniClient";
import { isBhashiniConfigured } from "@/config/bhashiniConfig";

// ── UI code → IndicTrans2 code ─────────────────────────────────────────────

const UI_TO_INDIC: Record<SupportedLanguage, string> = {
  en: "eng_Latn",
  hi: "hin_Deva",
  ta: "tam_Taml",
  te: "tel_Telu",
  bn: "ben_Beng",
  mr: "mar_Deva",
  gu: "guj_Gujr",
  kn: "kan_Knda",
  ml: "mal_Mlym",
  pa: "pan_Guru",
};

// ── Phrase-level mock dictionary (demo fallback) ───────────────────────────

const MOCK_DICTIONARY: Record<string, Partial<Record<SupportedLanguage, string>>> = {
  "Hey, did you get a chance to look at the report I sent?": {
    hi: "अरे, क्या आपने मेरी भेजी रिपोर्ट देखी?",
    ta: "ஏய், நான் அனுப்பிய அறிக்கையைப் பார்க்க வாய்ப்பு கிடைத்ததா?",
    te: "హే, నేను పంపిన నివేదికను చూసే అవకాశం మీకు దొరికిందా?",
  },
  "Not yet, I'll review it after lunch, around 2pm today.": {
    hi: "अभी नहीं, मैं दोपहर के भोजन के बाद लगभग 2 बजे देखूंगा।",
    ta: "இன்னும் இல்லை, மதிய உணவுக்குப் பிறகு சுமார் 2 மணிக்கு மதிப்பாய்வு செய்கிறேன்.",
    te: "ఇంకా లేదు, లంచ్ తర్వాత మధ్యాహ్నం 2 గంటల ప్రాంతంలో రివ్యూ చేస్తాను.",
  },
  "Okay, no rush, but the client meeting is Friday morning.": {
    hi: "ठीक है, कोई जल्दी नहीं, लेकिन क्लाइंट मीटिंग शुक्रवार सुबह है।",
    ta: "சரி, அவசரமில்லை, ஆனால் வாடிக்கையாளர் சந்திப்பு வெள்ளிக்கிழமை காலை.",
    te: "సరే, తొందరేం లేదు, కానీ క్లయింట్ మీటింగ్ శుక్రవారం ఉదయం ఉంది.",
  },
  "Got it, I'll have comments back to you by Thursday.": {
    hi: "समझ गया, मैं गुरुवार तक टिप्पणियाँ वापस दे दूँगा।",
    ta: "புரிந்தது, வியாழக்கிழமைக்குள் உங்களுக்குக் கருத்துகளைத் தருகிறேன்.",
    te: "అర్థమైంది, గురువారానికల్లా కామెంట్స్ మీకు అందిస్తాను.",
  },
};

// ── Free web translation fallback ─────────────────────────────────────────

async function freeWebTranslate(
  text: string,
  sourceLang: string,
  targetLang: string
): Promise<string> {
  const url = `https://translate.googleapis.com/translate_a/single?client=gtx&sl=${encodeURIComponent(
    sourceLang
  )}&tl=${encodeURIComponent(targetLang)}&dt=t&q=${encodeURIComponent(text)}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const data = await res.json();
  if (Array.isArray(data) && Array.isArray(data[0])) {
    const translated = data[0].map((item: any) => item[0]).join("");
    if (translated.trim()) return translated;
  }
  throw new Error("Empty translation result");
}

// ── Public API ─────────────────────────────────────────────────────────────

export function isPendingPlaceholder(text: string): boolean {
  return /^\[[A-Z]{2} translation pending API integration\]/.test(text);
}

export async function translateText(
  text: string,
  targetLanguage: SupportedLanguage,
  sourceLanguage: SupportedLanguage = "en"
): Promise<string> {
  if (!text.trim() || targetLanguage === sourceLanguage) return text;

  const sourceIndic = UI_TO_INDIC[sourceLanguage];
  const targetIndic = UI_TO_INDIC[targetLanguage];

  // 1. Try SENSEA FastAPI backend
  if (sourceIndic && targetIndic) {
    try {
      const result = await translateTextBackend(text, sourceIndic, targetIndic);
      if (result.translation && result.translation.trim()) {
        return result.translation;
      }
    } catch (err) {
      console.warn("[translationService] Backend translation failed, trying free fallback:", err);
    }
  }

  // 2. Try free web translation fallback (instant, works anywhere in browser)
  try {
    const webResult = await freeWebTranslate(text, sourceLanguage, targetLanguage);
    if (webResult) return webResult;
  } catch (err) {
    console.warn("[translationService] Free web translation failed:", err);
  }

  // 3. Try Bhashini (if credentials configured)
  if (isBhashiniConfigured()) {
    try {
      return await bhashiniTranslate(text, sourceLanguage, targetLanguage);
    } catch (err) {
      console.warn("[translationService] Bhashini translation failed, using mock dict:", err);
    }
  }

  // 4. Mock dictionary (demo phrases)
  const entry = MOCK_DICTIONARY[text];
  if (entry && entry[targetLanguage]) {
    return entry[targetLanguage] as string;
  }

  // 5. Clearly-marked placeholder (last resort)
  return `[${targetLanguage.toUpperCase()} translation pending API integration] ${text}`;
}
