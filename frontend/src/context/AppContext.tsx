import React, { createContext, useCallback, useContext, useEffect, useRef, useState } from "react";
import {
  CaptionEntry,
  EnvironmentalSoundEvent,
  RememberedDeadline,
  Speaker,
  SupportedLanguage,
} from "@/types";
import { simplifyText, stopStream } from "@/services/speechService";
import { transcribeAudio } from "@/services/backendApi";
import { translateText } from "@/services/translationService";
import { startMockDetection, stopMockDetection } from "@/services/soundDetectionService";
import {
  addDeadline,
  extractPossibleDeadline,
  getOrCreateSpeaker,
  getSpeakers,
} from "@/services/memoryService";

interface AppState {
  isListening: boolean;
toggleListening: () => Promise<void>;
  captions: CaptionEntry[];
  speakers: Record<string, Speaker>;
  targetLanguage: SupportedLanguage;
  setTargetLanguage: (lang: SupportedLanguage) => void;
  showTranslation: boolean;
  setShowTranslation: (v: boolean) => void;
  showSimplified: boolean;
  setShowSimplified: (v: boolean) => void;
  soundEvents: EnvironmentalSoundEvent[];
  latestSoundEvent: EnvironmentalSoundEvent | null;
  dismissLatestSoundEvent: () => void;
}

const AppContext = createContext<AppState | undefined>(undefined);

export function AppProvider({ children }: { children: React.ReactNode }) {
  const [isListening, setIsListening] = useState(false);
  const [captions, setCaptions] = useState<CaptionEntry[]>([]);
  const [speakers, setSpeakers] = useState<Record<string, Speaker>>({});
  const [targetLanguage, setTargetLanguage] = useState<SupportedLanguage>("hi");
  const [showTranslation, setShowTranslation] = useState(false);
  const [showSimplified, setShowSimplified] = useState(false);
  const [soundEvents, setSoundEvents] = useState<EnvironmentalSoundEvent[]>([]);
  const [latestSoundEvent, setLatestSoundEvent] = useState<EnvironmentalSoundEvent | null>(null);

  const targetLanguageRef = useRef(targetLanguage);
  targetLanguageRef.current = targetLanguage;

  const captionsRef = useRef(captions);
  captionsRef.current = captions;
 
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  useEffect(() => {
    getSpeakers().then((list) => {
      const map: Record<string, Speaker> = {};
      list.forEach((s) => (map[s.id] = s));
      setSpeakers(map);
    });
  }, []);

  const handleCaption = useCallback(async (entry: CaptionEntry) => {
    // Ensure we have a known speaker record (memory: "remembers speakers")
    const speaker = await getOrCreateSpeaker(entry.speakerId);
    setSpeakers((prev) => ({ ...prev, [speaker.id]: speaker }));

    let enriched = entry;

    if (entry.isFinal) {
      const simplifiedText = simplifyText(entry.originalText);
      const translatedText = await translateText(entry.originalText, targetLanguageRef.current);
      enriched = { ...entry, simplifiedText, translatedText };

      // Memory: passively capture possible deadlines mentioned in speech
      const possibleDeadline = extractPossibleDeadline(entry.originalText);
      if (possibleDeadline) {
        const deadline: RememberedDeadline = {
          id: `deadline_${entry.id}`,
          description: possibleDeadline,
          dueDate: new Date(Date.now() + 3 * 24 * 60 * 60 * 1000).toISOString(),
          sourceConversationId: entry.id,
          createdAt: Date.now(),
        };
        addDeadline(deadline).catch(() => {});
      }
    }

    setCaptions((prev) => {
      const idx = prev.findIndex((c) => c.id === entry.id);
      if (idx >= 0) {
        const next = [...prev];
        next[idx] = enriched;
        return next;
      }
      return [...prev, enriched];
    });
  }, []);

  


const toggleListening = useCallback(async () => {
  if (isListening) {
    mediaRecorderRef.current?.stop();
    setIsListening(false);
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: true,
    });
    streamRef.current = stream;

    const mimeType = MediaRecorder.isTypeSupported("audio/webm")
      ? "audio/webm"
      : "";

    const recorder = new MediaRecorder(
      stream,
      mimeType ? { mimeType } : undefined
    );
    mediaRecorderRef.current = recorder;

    const chunks: BlobPart[] = [];

    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) chunks.push(event.data);
    };

    recorder.onstop = async () => {
      try {
        const blob = new Blob(chunks, {
          type: recorder.mimeType || "audio/webm",
        });

        const result = await transcribeAudio(blob, "eng_Latn");

        if (result.transcript?.trim()) {
          await handleCaption({
            id: String(Date.now()),
            speakerId: "speaker-1",
            originalText: result.transcript,
            language: "en",
            timestamp: Date.now(),
            isFinal: true,
          });
        }
      } catch (error) {
        alert(
          error instanceof Error
            ? error.message
            : "Transcription failed."
        );
      } finally {
        stream.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
        mediaRecorderRef.current = null;
        setIsListening(false);
      }
    };

    recorder.start();
    setIsListening(true);
  } catch (error) {
    console.error(error);
    alert("Could not access the microphone. Please allow microphone access.");
    setIsListening(false);
  }
}, [isListening, handleCaption]);




  useEffect(() => {
    // Environmental sound detection runs independently of the caption stream —
    // a user should be alerted to a siren whether or not they're actively
    // captioning a conversation.
    startMockDetection((event) => {
      setSoundEvents((prev) => [event, ...prev].slice(0, 50));
      setLatestSoundEvent(event);
    });
    return () => stopMockDetection();
  }, []);

  const dismissLatestSoundEvent = useCallback(() => setLatestSoundEvent(null), []);

  // Bug fix: captions were only translated once, at the moment they arrived,
  // using whatever language was selected at that instant. Switching the
  // language chip afterward left already-received captions showing a stale
  // translation from the old language. Re-translate every final caption
  // whenever the target language changes so the displayed text always
  // matches the currently selected chip.
  useEffect(() => {
    let cancelled = false;
    const finals = captionsRef.current.filter((c) => c.isFinal);
    if (finals.length === 0) return;

    Promise.all(finals.map((c) => translateText(c.originalText, targetLanguage))).then(
      (translations) => {
        if (cancelled) return;
        setCaptions((current) =>
          current.map((c) => {
            const idx = finals.findIndex((f) => f.id === c.id);
            return idx >= 0 ? { ...c, translatedText: translations[idx] } : c;
          })
        );
      }
    );

    return () => {
      cancelled = true;
    };
  }, [targetLanguage]);

  return (
    <AppContext.Provider
      value={{
        isListening,
        toggleListening,
        captions,
        speakers,
        targetLanguage,
        setTargetLanguage,
        showTranslation,
        setShowTranslation,
        showSimplified,
        setShowSimplified,
        soundEvents,
        latestSoundEvent,
        dismissLatestSoundEvent,
      }}
    >
      {children}
    </AppContext.Provider>
  );
}

export function useAppState(): AppState {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error("useAppState must be used within AppProvider");
  return ctx;
}
