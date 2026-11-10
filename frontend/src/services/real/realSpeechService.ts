/**
 * REAL SPEECH SERVICE
 * -------------------
 * Real microphone live-captioning service:
 * 1. Uses the browser's Web Speech API (SpeechRecognition / webkitSpeechRecognition)
 *    when available for instant, low-latency live streaming transcription.
 * 2. Falls back to MediaRecorder + SENSEA FastAPI backend (POST /api/speech/transcribe)
 *    for server-side Whisper transcription.
 */

import { SpeechService } from "@/services/types/speechServiceTypes";
import { transcribeAudio } from "@/services/backendApi";
import { CaptionEntry, SupportedLanguage } from "@/types";

type CaptionCallback = (entry: CaptionEntry) => void;

/** Map UI languages to BCP-47 codes for browser Web Speech API. */
const BCP47_LANG_MAP: Record<SupportedLanguage, string> = {
  en: "en-US",
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

/** Map UI language codes to IndicTrans2 speech codes for FastAPI Whisper. */
const BACKEND_LANG_MAP: Partial<Record<SupportedLanguage, string>> = {
  en: "eng_Latn",
  hi: "hin_Deva",
};

function makeId(): string {
  return `cap_${Date.now()}_${Math.floor(Math.random() * 10000)}`;
}

let activeRecognition: any = null;
let mediaRecorder: MediaRecorder | null = null;
let activeStream: MediaStream | null = null;
let chunkInterval: ReturnType<typeof setInterval> | null = null;
let isStopping = false;

const realSpeechService: SpeechService = {
  startStream(onCaption: CaptionCallback, language: SupportedLanguage = "en") {
    realSpeechService.stopStream();
    isStopping = false;

    // Check for Browser Web Speech API (Chrome, Edge, Safari, Android)
    const SpeechRecognitionAPI =
      typeof window !== "undefined"
        ? (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition
        : null;

    if (SpeechRecognitionAPI) {
      try {
        const recognition = new SpeechRecognitionAPI();
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.lang = BCP47_LANG_MAP[language] || "en-US";

        let currentId = makeId();

        recognition.onresult = (event: any) => {
          for (let i = event.resultIndex; i < event.results.length; ++i) {
            const transcript = event.results[i][0].transcript;
            const isFinal = event.results[i].isFinal;

            if (transcript.trim()) {
              onCaption({
                id: currentId,
                speakerId: "speaker-1",
                originalText: transcript.trim(),
                language,
                timestamp: Date.now(),
                isFinal,
              });

              if (isFinal) {
                // Prepare a new ID for the next utterance
                currentId = makeId();
              }
            }
          }
        };

        recognition.onerror = (event: any) => {
          console.warn("[realSpeechService] Web Speech error:", event.error);
          if (event.error === "not-allowed") {
            alert("Microphone permission was denied. Please allow microphone access in your browser.");
          }
        };

        recognition.onend = () => {
          // Restart if still supposed to be listening (browser auto-stops on long pauses)
          if (!isStopping && activeRecognition === recognition) {
            try {
              recognition.start();
            } catch {
              // already started
            }
          }
        };

        recognition.start();
        activeRecognition = recognition;
        console.log("[realSpeechService] Using Web Speech API for live transcription.");
        return;
      } catch (err) {
        console.warn("[realSpeechService] Web Speech API failed, falling back to MediaRecorder:", err);
      }
    }

    // Fallback: MediaRecorder -> Backend API (Whisper.cpp)
    if (typeof navigator !== "undefined" && navigator.mediaDevices?.getUserMedia) {
      const backendLang = BACKEND_LANG_MAP[language] ?? "eng_Latn";
      const CHUNK_MS = 6000;

      navigator.mediaDevices
        .getUserMedia({ audio: true })
        .then((stream) => {
          if (isStopping) {
            stream.getTracks().forEach((t) => t.stop());
            return;
          }

          activeStream = stream;
          const mimeType = MediaRecorder.isTypeSupported("audio/webm") ? "audio/webm" : "";
          const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
          mediaRecorder = recorder;

          let chunks: BlobPart[] = [];
          recorder.ondataavailable = (e) => {
            if (e.data.size > 0) chunks.push(e.data);
          };

          const transcribeChunk = async () => {
            if (chunks.length === 0) return;
            const blob = new Blob(chunks, { type: recorder.mimeType || "audio/webm" });
            chunks = [];

            const id = makeId();
            const timestamp = Date.now();

            onCaption({
              id,
              speakerId: "speaker-1",
              originalText: "…",
              language,
              timestamp,
              isFinal: false,
            });

            try {
              const result = await transcribeAudio(blob, backendLang);
              if (result.transcript?.trim()) {
                onCaption({
                  id,
                  speakerId: "speaker-1",
                  originalText: result.transcript,
                  language,
                  timestamp,
                  isFinal: true,
                });
              }
            } catch (err) {
              console.warn("[realSpeechService] Backend transcription failed:", err);
              onCaption({
                id,
                speakerId: "speaker-1",
                originalText: "",
                language,
                timestamp,
                isFinal: true,
              });
            }
          };

          recorder.start(CHUNK_MS);
          chunkInterval = setInterval(() => {
            if (recorder.state === "recording") {
              recorder.requestData();
            }
            transcribeChunk();
          }, CHUNK_MS);
        })
        .catch((err) => {
          console.error("[realSpeechService] Microphone access error:", err);
          alert("Could not access microphone. Please grant microphone permission.");
        });
    }
  },

  stopStream() {
    isStopping = true;

    if (activeRecognition) {
      try {
        activeRecognition.stop();
      } catch {}
      activeRecognition = null;
    }

    if (chunkInterval) {
      clearInterval(chunkInterval);
      chunkInterval = null;
    }

    if (mediaRecorder && mediaRecorder.state !== "inactive") {
      try {
        mediaRecorder.stop();
      } catch {}
    }
    mediaRecorder = null;

    if (activeStream) {
      activeStream.getTracks().forEach((t) => t.stop());
      activeStream = null;
    }
  },
};

export default realSpeechService;
