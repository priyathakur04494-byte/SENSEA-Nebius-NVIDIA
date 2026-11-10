/**
 * SENSEA BACKEND API CLIENT
 * --------------------------
 * Thin typed wrappers around every endpoint exposed by the FastAPI backend
 * (sensea/api/). All functions throw on non-OK responses so callers can
 * catch and decide whether to fall back to a mock.
 *
 * Base URL is read from the EXPO_PUBLIC_API_URL env var so you can point at
 * a local dev server, a hosted instance, or a tunnel without touching code:
 *
 *   EXPO_PUBLIC_API_URL=http://192.168.1.42:8000   # LAN dev
 *   EXPO_PUBLIC_API_URL=https://sensea.example.com # production
 *
 * Falls back to http://127.0.0.1:8000 when the var is not set.
 */

const API_BASE_URL =
  (process.env.EXPO_PUBLIC_API_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");

// ── Helpers ────────────────────────────────────────────────────────────────

async function checkResponse(res: Response): Promise<Response> {
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`API ${res.status}: ${body || res.statusText}`);
  }
  return res;
}

// ── Speech ─────────────────────────────────────────────────────────────────

export interface TranscribeResult {
  transcript: string;
  language: string;
  whisper_lang: string;
}

/**
 * POST /api/speech/transcribe
 * Upload an audio Blob; returns a transcript.
 * language: 'eng_Latn' | 'hin_Deva' | 'hinglish'
 */
export async function transcribeAudio(
  audio: Blob,
  language = "eng_Latn"
): Promise<TranscribeResult> {
  const extension = audio.type.includes("webm") ? "webm" : "wav";
  const formData = new FormData();
  formData.append("audio", audio, `recording.${extension}`);
  formData.append("language", language);

  const res = await fetch(`${API_BASE_URL}/api/speech/transcribe`, {
    method: "POST",
    body: formData,
  });
  await checkResponse(res);
  return res.json();
}

export interface SpeakerSegment {
  speaker_id: string;
  is_new: boolean;
  confidence: number;
  duration_seconds: number;
}

export interface DiarizeResult {
  speakers: SpeakerSegment[];
  total_speakers: number;
}

/**
 * POST /api/speech/diarize
 * Run speaker diarization on an audio Blob.
 */
export async function diarizeAudio(audio: Blob): Promise<DiarizeResult> {
  const formData = new FormData();
  formData.append("audio", audio, "recording.wav");

  const res = await fetch(`${API_BASE_URL}/api/speech/diarize`, {
    method: "POST",
    body: formData,
  });
  await checkResponse(res);
  return res.json();
}

// ── Translation ────────────────────────────────────────────────────────────

export interface TranslationResult {
  translation: string;
  source_language: string;
  target_language: string;
}

/**
 * POST /api/translation
 * Translate text using IndicTrans2 on the backend.
 * Language codes: IndicTrans2 format, e.g. 'eng_Latn', 'hin_Deva'.
 */
export async function translateTextBackend(
  text: string,
  sourceLanguage: string,
  targetLanguage: string
): Promise<TranslationResult> {
  const res = await fetch(`${API_BASE_URL}/api/translation`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, source_language: sourceLanguage, target_language: targetLanguage }),
  });
  await checkResponse(res);
  return res.json();
}

// ── TTS ────────────────────────────────────────────────────────────────────

/**
 * POST /api/tts
 * Convert text to speech using Kokoro on the backend.
 * Returns a WAV Blob (24 kHz mono) ready to pass to the Audio API.
 */
export async function textToSpeechBackend(text: string): Promise<Blob> {
  const res = await fetch(`${API_BASE_URL}/api/tts`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  await checkResponse(res);
  return res.blob();
}

// ── Languages ──────────────────────────────────────────────────────────────

export interface LanguageInfo {
  code: string;
  name: string;
}

export interface LanguagesResult {
  speech_languages: LanguageInfo[];
  translation_targets: LanguageInfo[];
}

/**
 * GET /api/languages
 * Fetch supported speech & translation language codes from the backend.
 */
export async function fetchSupportedLanguages(): Promise<LanguagesResult> {
  const res = await fetch(`${API_BASE_URL}/api/languages`);
  await checkResponse(res);
  return res.json();
}

// ── Chat / Memory ──────────────────────────────────────────────────────────

export interface ChatResult {
  answer: string;
  mode: string;
}

/**
 * POST /api/chat
 * Answer a question using RAG over stored memories (Nemotron).
 */
export async function chatQuestion(
  question: string,
  mode = "general",
  userId?: string
): Promise<ChatResult> {
  const res = await fetch(`${API_BASE_URL}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, mode, user_id: userId ?? null }),
  });
  await checkResponse(res);
  return res.json();
}

export interface MemoryItem {
  text: string;
  memory_type: string;
  importance?: number;
  date?: string;
  time?: string;
}

export interface AddMemoriesResult {
  added: number;
  mode: string;
}

/**
 * POST /api/memories
 * Store normalized memory objects extracted from a conversation.
 */
export async function addMemories(
  memories: MemoryItem[],
  mode = "general",
  userId?: string
): Promise<AddMemoriesResult> {
  const res = await fetch(`${API_BASE_URL}/api/memories`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ memories, mode, user_id: userId ?? null }),
  });
  await checkResponse(res);
  return res.json();
}

export interface GetMemoriesResult {
  memories: Record<string, unknown>[];
  count: number;
  mode: string | null;
}

/**
 * GET /api/memories
 * Retrieve stored memories, optionally filtered by mode / user.
 */
export async function getMemories(
  mode?: string,
  userId?: string
): Promise<GetMemoriesResult> {
  const params = new URLSearchParams();
  if (mode) params.set("mode", mode);
  if (userId) params.set("user_id", userId);

  const res = await fetch(`${API_BASE_URL}/api/memories?${params}`);
  await checkResponse(res);
  return res.json();
}

/**
 * DELETE /api/memories
 * Clear all stored memories for a user (or all memories if userId is omitted).
 */
export async function clearMemories(userId?: string): Promise<void> {
  const params = new URLSearchParams();
  if (userId) params.set("user_id", userId);

  const res = await fetch(`${API_BASE_URL}/api/memories?${params}`, {
    method: "DELETE",
  });
  await checkResponse(res);
}

// ── Conversation Analysis ──────────────────────────────────────────────────

export interface ConversationSegment {
  speaker_id: string;
  text: string;
  translation?: string;
}

export interface AnalyzeResult {
  mode: string;
  result: Record<string, unknown>;
}

/**
 * POST /api/conversation/analyze
 * Analyze a conversation with NVIDIA Nemotron.
 * mode: 'general' | 'hospital' | 'meeting' | 'classroom'
 */
export async function analyzeConversation(
  mode: string,
  segments: ConversationSegment[]
): Promise<AnalyzeResult> {
  const res = await fetch(`${API_BASE_URL}/api/conversation/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode, segments }),
  });
  await checkResponse(res);
  return res.json();
}

// ── Full Pipeline ──────────────────────────────────────────────────────────

export interface PipelineSegment {
  speaker_id: string;
  text: string;
  translation: string;
  confidence: number;
  duration_seconds: number;
}

export interface PipelineResult {
  segments: PipelineSegment[];
  total_speakers: number;
}

/**
 * POST /api/pipeline
 * Run the full diarize → transcribe → (optional) translate pipeline in one call.
 */
export async function runPipeline(
  audio: Blob,
  speechLanguage = "eng_Latn",
  translateTo?: string
): Promise<PipelineResult> {
  const formData = new FormData();
  formData.append("audio", audio, "recording.wav");
  formData.append("speech_language", speechLanguage);
  if (translateTo) formData.append("translate_to", translateTo);

  const res = await fetch(`${API_BASE_URL}/api/pipeline`, {
    method: "POST",
    body: formData,
  });
  await checkResponse(res);
  return res.json();
}

// ── Health ─────────────────────────────────────────────────────────────────

/**
 * GET /api/health
 * Returns true if the backend is reachable.
 */
export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/health`, { method: "GET" });
    return res.ok;
  } catch {
    return false;
  }
}


// ── Sign Language Model ────────────────────────────────────────────────────

export interface SignPrediction {
  label: string;
  confidence: number;
  all_scores?: Record<string, number>;
}

export interface SignPredictResult {
  prediction: SignPrediction;
  frames_used: number;
  orientation: string;
}

/**
 * POST /api/sign-language/predict
 * Upload a video of a sign language gesture; returns PyTorch model predictions.
 */
export async function predictSignFromVideo(videoBlob: Blob): Promise<SignPredictResult> {
  const formData = new FormData();
  formData.append("video", videoBlob, "gesture.webm");

  const res = await fetch(`${API_BASE_URL}/api/sign-language/predict`, {
    method: "POST",
    body: formData,
  });
  await checkResponse(res);
  return res.json();
}
