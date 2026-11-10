import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Pressable, StyleSheet, Text, TextInput, useWindowDimensions, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import HandTrackerView from "./HandTrackerView";
import SignMlPredictor from "./SignMlPredictor";
import {
  DetectedSign,
  HandTrackerHandle,
  TrackerMessage,
  TrackerState,
} from "./handTrackerTypes";
import { RECOGNIZABLE_SIGNS } from "@/config/recognizedSigns";
import { CustomSign, loadCustomSigns, saveCustomSigns } from "@/services/customSignStore";
import { speak, stopSpeaking } from "@/services/ttsService";
import { breakpoints, colors, radius, shadow, spacing } from "@/theme/colors";

/**
 * RECOGNISE — point the camera at your hands, sign, and SENSEA shows and says
 * what you are signing. Everything runs on the device; no video leaves it.
 */

interface Taught {
  phase: "idle" | "running";
  take: number;
  takes: number;
  count?: number;
  recording?: boolean;
}

export default function SignRecognizer() {
  const { width } = useWindowDimensions();
  const isWide = width >= breakpoints.tablet;

  const handle = useRef<HandTrackerHandle | null>(null);
  const [session, setSession] = useState(0); // bump to restart the camera
  const [state, setState] = useState<TrackerState>("loading");
  const [statusText, setStatusText] = useState("Loading hand model…");
  const [facing, setFacing] = useState<"user" | "environment">("user");
  const [handsSeen, setHandsSeen] = useState(0);
  const [current, setCurrent] = useState<DetectedSign | null>(null);
  const [transcript, setTranscript] = useState<DetectedSign[]>([]);
  const [voice, setVoice] = useState(false);
  const voiceRef = useRef(false);
  voiceRef.current = voice;

  const [custom, setCustom] = useState<CustomSign[]>([]);
  const customRef = useRef<CustomSign[]>([]);
  customRef.current = custom;
  const [loadedCustom, setLoadedCustom] = useState(false);
  const [teachName, setTeachName] = useState("");
  const [teaching, setTeaching] = useState<Taught>({ phase: "idle", take: 0, takes: 3 });
  const [teachNote, setTeachNote] = useState<string | null>(null);
  const [showVocab, setShowVocab] = useState(false);
  const [pageReady, setPageReady] = useState(false);
  const [recognizerMode, setRecognizerMode] = useState<"ml" | "tracker">("ml");

  useEffect(() => {
    loadCustomSigns().then((list: CustomSign[]) => {
      setCustom(list);
      setLoadedCustom(true);
    });
    return () => stopSpeaking();
  }, []);

  // ---- Messages from the camera page -------------------------------------
  const onMessage = useCallback((m: TrackerMessage) => {
    switch (m.type) {
      case "ready":
        setPageReady(true);
        break;
      case "status":
        setState(m.state);
        setStatusText(m.message);
        break;
      case "frame":
        setHandsSeen(m.hands.length);
        setCurrent(m.current);
        break;
      case "confirmed":
        setTranscript((t) => [...t.slice(-39), m.sign]);
        if (voiceRef.current) speak(m.sign.label, "en");
        break;
      case "teach-progress":
        setTeaching({
          phase: "running",
          take: m.take,
          takes: m.takes,
          count: m.count,
          recording: m.phase === "record",
        });
        break;
      case "taught": {
        const next = [...customRef.current.filter((s) => s.id !== m.sign.id), m.sign];
        setCustom(next);
        saveCustomSigns(next);
        setTeaching({ phase: "idle", take: 0, takes: 3 });
        setTeachName("");
        setTeachNote(`Saved “${m.sign.label}”. Sign it any time and SENSEA will say it.`);
        break;
      }
      case "teach-failed":
        setTeaching({ phase: "idle", take: 0, takes: 3 });
        setTeachNote(m.reason);
        break;
    }
  }, []);

  // ---- Start the page once the saved signs are loaded --------------------
  const onHandle = useCallback((h: HandTrackerHandle) => {
    handle.current = h;
  }, []);
  // Once the camera page says it is ready AND the saved signs are loaded, start it.
  useEffect(() => {
    if (!pageReady || !loadedCustom) return;
    handle.current?.send({ type: "init", customSigns: customRef.current, facing });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pageReady, loadedCustom]);

  const restart = () => {
    setState("loading");
    setStatusText("Loading hand model…");
    setPageReady(false);
    setSession((s) => s + 1);
  };

  const flipCamera = () => {
    const next = facing === "user" ? "environment" : "user";
    setFacing(next);
    handle.current?.send({ type: "camera", facing: next });
  };

  // ---- Teaching ------------------------------------------------------------
  const startTeach = () => {
    const name = teachName.trim();
    if (!name) {
      setTeachNote("Type what this sign means first (for example “Thank you”).");
      return;
    }
    setTeachNote(null);
    const id = `c${Date.now().toString(36)}`;
    setTeaching({ phase: "running", take: 1, takes: 3, count: 3 });
    handle.current?.send({ type: "teach", id, label: name });
  };
  const cancelTeach = () => {
    handle.current?.send({ type: "cancel-teach" });
    setTeaching({ phase: "idle", take: 0, takes: 3 });
  };
  const removeCustom = (id: string) => {
    const next = custom.filter((s) => s.id !== id);
    setCustom(next);
    saveCustomSigns(next);
    handle.current?.send({ type: "set-custom", customSigns: next });
  };

  const sentence = useMemo(() => transcript.map((t) => t.label).join("  ·  "), [transcript]);

  // ---- Layout -----------------------------------------------------------------
  const camWidth = isWide ? 420 : Math.min(width - spacing.md * 2, 520);
  const camHeight = Math.round(camWidth * 0.78);

  const camera = (
    <View style={[styles.cameraCard, shadow.md]}>
      <View key={session}>
        <HandTrackerView onMessage={onMessage} onHandle={onHandle} height={camHeight} />
      </View>

      {(state === "denied" || state === "error") && (
        <View style={styles.errorBox} accessibilityRole="alert">
          <Ionicons name="videocam-off-outline" size={20} color={colors.critical} />
          <View style={{ flex: 1 }}>
            <Text style={styles.errorTitle}>{state === "denied" ? "Camera is blocked" : "Camera could not start"}</Text>
            <Text style={styles.errorText}>{statusText}</Text>
          </View>
          <Pressable onPress={restart} style={styles.retryButton} accessibilityRole="button" accessibilityLabel="Try the camera again">
            <Text style={styles.retryText}>Try again</Text>
          </Pressable>
        </View>
      )}

      <View style={styles.cameraBar}>
        <View style={styles.statusPill}>
          <Ionicons
            name={state === "running" ? (handsSeen ? "hand-left" : "hand-left-outline") : "time-outline"}
            size={14}
            color={colors.accentStrong}
          />
          <Text style={styles.statusText}>
            {state === "running"
              ? handsSeen === 0
                ? "No hand in view"
                : handsSeen === 1
                  ? "1 hand tracked"
                  : "2 hands tracked"
              : state === "loading"
                ? "Starting…"
                : "Camera off"}
          </Text>
        </View>
        <Pressable onPress={flipCamera} style={styles.optionChip} accessibilityRole="button" accessibilityLabel="Switch between front and back camera">
          <Ionicons name="camera-reverse-outline" size={15} color={colors.accentStrong} />
          <Text style={styles.optionText}>Flip camera</Text>
        </Pressable>
      </View>
    </View>
  );

  const confidencePct = current ? Math.round(current.confidence * 100) : 0;

  const result = (
    <View style={[styles.resultCard, shadow.md]}>
      <Text style={styles.fieldLabel}>YOU ARE SIGNING</Text>
      <View accessibilityLiveRegion="polite" accessible accessibilityLabel={current ? `Sign recognised: ${current.label}` : "No sign recognised yet"}>
        {current ? (
          <>
            <Text style={styles.resultWord}>{current.label}</Text>
            <View style={styles.badgeRow}>
              {current.custom && <Badge icon="person" text="Your own sign" />}
              {current.motion && <Badge icon="swap-horizontal" text="Moving sign" />}
              {!!current.note && !current.custom && <Text style={styles.resultNote}>{current.note}</Text>}
            </View>
            <View style={styles.confRow}>
              <View style={styles.confTrack}>
                <View style={[styles.confFill, { width: `${confidencePct}%` }]} />
              </View>
              <Text style={styles.confText}>{confidencePct}% match</Text>
            </View>
          </>
        ) : (
          <Text style={styles.placeholder}>
            {state === "running"
              ? handsSeen
                ? "Hold the sign steady for a moment…"
                : "Show your hand to the camera and make a sign."
              : "Waiting for the camera…"}
          </Text>
        )}
      </View>

      <View style={styles.divider} />
      <View style={styles.rowBetween}>
        <Text style={styles.fieldLabel}>WHAT YOU SIGNED</Text>
        <Pressable
          onPress={() => {
            if (voice) stopSpeaking();
            setVoice((v) => !v);
          }}
          accessibilityRole="switch"
          accessibilityState={{ checked: voice }}
          accessibilityLabel="Say each recognised sign aloud"
          style={[styles.optionChip, voice && styles.optionChipOn]}
        >
          <Ionicons name={voice ? "volume-high" : "volume-mute-outline"} size={15} color={colors.accentStrong} />
          <Text style={styles.optionText}>{voice ? "Voice on" : "Voice off"}</Text>
        </Pressable>
      </View>
      <View style={styles.transcriptBox}>
        {transcript.length === 0 ? (
          <Text style={styles.placeholder}>Signs you hold steady appear here, in order.</Text>
        ) : (
          <View style={styles.chipWrap}>
            {transcript.map((t, i) => (
              <View key={i} style={[styles.signChip, t.custom && styles.signChipCustom]}>
                <Text style={styles.signChipText}>{t.label}</Text>
              </View>
            ))}
          </View>
        )}
      </View>
      <View style={styles.actionRow}>
        <Pressable
          onPress={() => speak(transcript.map((t) => t.label).join(". "), "en")}
          disabled={!transcript.length}
          accessibilityRole="button"
          accessibilityLabel="Speak everything signed"
          style={[styles.primaryButton, !transcript.length && styles.disabled]}
        >
          <Ionicons name="volume-high" size={16} color={colors.textOnAccent} />
          <Text style={styles.primaryText}>Say it aloud</Text>
        </Pressable>
        <Pressable
          onPress={() => setTranscript([])}
          disabled={!transcript.length}
          accessibilityRole="button"
          accessibilityLabel="Clear what you signed"
          style={[styles.optionChip, !transcript.length && styles.disabled]}
        >
          <Ionicons name="trash-outline" size={15} color={colors.accentStrong} />
          <Text style={styles.optionText}>Clear</Text>
        </Pressable>
      </View>
      {!!sentence && <Text style={styles.sentence} selectable>{sentence}</Text>}
    </View>
  );

  const teach = (
    <View style={[styles.card, shadow.sm]}>
      <View style={styles.rowBetween}>
        <Text style={styles.cardTitle}>Teach SENSEA a sign</Text>
        <Ionicons name="school-outline" size={18} color={colors.accentStrong} />
      </View>
      <Text style={styles.cardBody}>
        Use any sign language — ISL, BSL, ASL, or signs you and your family made up. Type what the sign means, then
        do the sign three times when the countdown ends. SENSEA will recognise it from then on.
      </Text>

      {teaching.phase === "idle" ? (
        <View style={styles.teachRow}>
          <TextInput
            value={teachName}
            onChangeText={setTeachName}
            placeholder="What does it mean? e.g. Thank you"
            placeholderTextColor={colors.textMuted}
            style={styles.input}
            accessibilityLabel="Meaning of the sign you want to teach"
            maxLength={40}
          />
          <Pressable
            onPress={startTeach}
            disabled={state !== "running"}
            accessibilityRole="button"
            accessibilityLabel="Start recording the sign"
            style={[styles.primaryButton, state !== "running" && styles.disabled]}
          >
            <Ionicons name="radio-button-on" size={16} color={colors.textOnAccent} />
            <Text style={styles.primaryText}>Record</Text>
          </Pressable>
        </View>
      ) : (
        <View style={styles.teachProgress} accessibilityLiveRegion="polite">
          <Text style={styles.teachProgressTitle}>
            Take {teaching.take} of {teaching.takes}
          </Text>
          <Text style={styles.cardBody}>
            {teaching.recording ? "Sign now — keep your hand in view." : `Get ready… ${teaching.count ?? ""}`}
          </Text>
          <Pressable onPress={cancelTeach} style={styles.optionChip} accessibilityRole="button" accessibilityLabel="Cancel recording">
            <Ionicons name="close" size={15} color={colors.accentStrong} />
            <Text style={styles.optionText}>Cancel</Text>
          </Pressable>
        </View>
      )}
      {!!teachNote && <Text style={styles.teachNote}>{teachNote}</Text>}

      {custom.length > 0 && (
        <View style={{ marginTop: spacing.md }}>
          <Text style={styles.fieldLabel}>YOUR TAUGHT SIGNS</Text>
          {custom.map((s) => (
            <View key={s.id} style={styles.customRow}>
              <Text style={styles.customLabel}>{s.label}</Text>
              <Pressable onPress={() => removeCustom(s.id)} accessibilityRole="button" accessibilityLabel={`Delete taught sign ${s.label}`} style={styles.iconButton}>
                <Ionicons name="trash-outline" size={16} color={colors.critical} />
              </Pressable>
            </View>
          ))}
        </View>
      )}
    </View>
  );

  const vocab = (
    <View style={[styles.card, shadow.sm]}>
      <Pressable
        onPress={() => setShowVocab((v) => !v)}
        accessibilityRole="button"
        accessibilityState={{ expanded: showVocab }}
        style={styles.rowBetween}
      >
        <Text style={styles.cardTitle}>Signs it knows already ({RECOGNIZABLE_SIGNS.length})</Text>
        <Ionicons name={showVocab ? "chevron-up" : "chevron-down"} size={18} color={colors.accentStrong} />
      </Pressable>
      {showVocab &&
        RECOGNIZABLE_SIGNS.map((s) => (
          <View key={s.label} style={styles.vocabRow}>
            <Text style={styles.vocabLabel}>{s.label}</Text>
            <Text style={styles.vocabHow}>
              {s.how}
              {s.moving ? "  (moving)" : ""}
            </Text>
          </View>
        ))}
      <Text style={styles.honest}>
        These are common handshapes and movements, not a full sign language. Real signing also uses the face, body and
        speed, which this cannot read yet — for anything else, teach it above. Good light, a plain background and one
        arm’s length from the camera help most.
      </Text>
    </View>
  );

  return (
    <View>
      <Text style={styles.title}>Sign to text</Text>
      <Text style={styles.subtitle}>
        Recognize sign language gestures using either your trained PyTorch ISL deep learning model, or instant on-device hand tracking.
      </Text>

      <View style={styles.modeToggleBar}>
        <Pressable
          onPress={() => setRecognizerMode("ml")}
          style={[styles.modeToggleTab, recognizerMode === "ml" && styles.modeToggleTabActive]}
          accessibilityRole="tab"
          accessibilityState={{ selected: recognizerMode === "ml" }}
        >
          <Ionicons
            name="hardware-chip"
            size={16}
            color={recognizerMode === "ml" ? colors.textOnAccent : colors.accentStrong}
          />
          <Text style={[styles.modeToggleText, recognizerMode === "ml" && styles.modeToggleTextActive]}>
            ISL Neural Model (Backend ML)
          </Text>
        </Pressable>

        <Pressable
          onPress={() => setRecognizerMode("tracker")}
          style={[styles.modeToggleTab, recognizerMode === "tracker" && styles.modeToggleTabActive]}
          accessibilityRole="tab"
          accessibilityState={{ selected: recognizerMode === "tracker" }}
        >
          <Ionicons
            name="hand-left-outline"
            size={16}
            color={recognizerMode === "tracker" ? colors.textOnAccent : colors.accentStrong}
          />
          <Text style={[styles.modeToggleText, recognizerMode === "tracker" && styles.modeToggleTextActive]}>
            On-Device Tracker (18 Signs)
          </Text>
        </Pressable>
      </View>

      {recognizerMode === "ml" ? (
        <SignMlPredictor
          onSignRecognized={(sign) => {
            setTranscript((t) => [...t.slice(-39), { id: 'ml_' + String(Date.now()), label: sign, confidence: 0.95 }]);
          }}
        />
      ) : null}

      {recognizerMode === "tracker" && (
        isWide ? (
          <View style={styles.wideRow}>
            <View style={{ width: camWidth + 2 }}>
              {camera}
              {vocab}
            </View>
            <View style={{ flex: 1 }}>
              {result}
              {teach}
            </View>
          </View>
        ) : (
          <>
            {camera}
            {result}
            {teach}
            {vocab}
          </>
        )
      )}
    </View>
  );
}

function Badge({ icon, text }: { icon: keyof typeof Ionicons.glyphMap; text: string }) {
  return (
    <View style={styles.badge}>
      <Ionicons name={icon} size={12} color={colors.accentStrong} />
      <Text style={styles.badgeText}>{text}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  modeToggleBar: {
    flexDirection: "row",
    gap: spacing.sm,
    backgroundColor: colors.surface,
    padding: 6,
    borderRadius: radius.pill,
    borderWidth: 1,
    borderColor: colors.border,
    marginBottom: spacing.md,
  },
  modeToggleTab: {
    flex: 1,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 6,
    paddingVertical: 10,
    borderRadius: radius.pill,
    backgroundColor: "transparent",
  },
  modeToggleTabActive: {
    backgroundColor: colors.accent,
  },
  modeToggleText: {
    fontSize: 13,
    fontWeight: "700",
    color: colors.textSecondary,
  },
  modeToggleTextActive: {
    color: colors.textOnAccent,
  },
  title: { fontSize: 22, fontWeight: "800", color: colors.textPrimary },
  subtitle: { fontSize: 14, lineHeight: 20, color: colors.textSecondary, marginTop: 4, marginBottom: spacing.md },
  wideRow: { flexDirection: "row", gap: spacing.md, alignItems: "flex-start" },

  cameraCard: {
    backgroundColor: colors.surface, borderRadius: radius.lg, borderWidth: 1, borderColor: colors.border,
    padding: spacing.sm, marginBottom: spacing.md,
  },
  cameraBar: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", marginTop: spacing.sm },
  statusPill: {
    flexDirection: "row", alignItems: "center", gap: 6, backgroundColor: colors.accentSoft,
    paddingHorizontal: 10, paddingVertical: 6, borderRadius: radius.pill,
  },
  statusText: { fontSize: 12, fontWeight: "700", color: colors.accentStrong },
  errorBox: {
    flexDirection: "row", gap: spacing.sm, alignItems: "center", backgroundColor: colors.criticalSoft,
    borderRadius: radius.md, padding: spacing.sm, marginTop: spacing.sm,
  },
  errorTitle: { fontSize: 14, fontWeight: "800", color: colors.critical },
  errorText: { fontSize: 13, color: colors.textSecondary, marginTop: 2 },
  retryButton: { backgroundColor: colors.critical, paddingHorizontal: 12, paddingVertical: 8, borderRadius: radius.pill },
  retryText: { color: "#fff", fontWeight: "700", fontSize: 13 },

  resultCard: {
    backgroundColor: colors.surface, borderRadius: radius.lg, borderWidth: 1, borderColor: colors.border,
    padding: spacing.md, marginBottom: spacing.md,
  },
  fieldLabel: { fontSize: 11, fontWeight: "800", letterSpacing: 0.8, color: colors.textMuted },
  resultWord: { fontSize: 38, lineHeight: 46, fontWeight: "800", color: colors.primary, marginTop: 4 },
  resultNote: { fontSize: 13, color: colors.textSecondary, flexShrink: 1 },
  badgeRow: { flexDirection: "row", flexWrap: "wrap", alignItems: "center", gap: 8, marginTop: 2 },
  badge: {
    flexDirection: "row", alignItems: "center", gap: 4, backgroundColor: colors.accentSoft,
    paddingHorizontal: 8, paddingVertical: 3, borderRadius: radius.pill,
  },
  badgeText: { fontSize: 11, fontWeight: "700", color: colors.accentStrong },
  confRow: { flexDirection: "row", alignItems: "center", gap: 10, marginTop: spacing.sm },
  confTrack: { flex: 1, height: 8, borderRadius: 4, backgroundColor: colors.surfaceMuted, overflow: "hidden" },
  confFill: { height: 8, backgroundColor: colors.accent },
  confText: { fontSize: 12, fontWeight: "700", color: colors.textSecondary, minWidth: 74, textAlign: "right" },
  placeholder: { fontSize: 15, lineHeight: 22, color: colors.textMuted, marginTop: 6 },
  divider: { height: 1, backgroundColor: colors.border, marginVertical: spacing.md },
  rowBetween: { flexDirection: "row", justifyContent: "space-between", alignItems: "center" },
  transcriptBox: { minHeight: 52, marginTop: spacing.sm },
  chipWrap: { flexDirection: "row", flexWrap: "wrap", gap: 6 },
  signChip: { backgroundColor: colors.primarySoft, borderRadius: radius.pill, paddingHorizontal: 12, paddingVertical: 6 },
  signChipCustom: { backgroundColor: colors.accentSoft },
  signChipText: { fontSize: 14, fontWeight: "700", color: colors.primary },
  actionRow: { flexDirection: "row", alignItems: "center", gap: spacing.sm, marginTop: spacing.sm },
  sentence: { fontSize: 12, color: colors.textMuted, marginTop: spacing.sm },

  card: {
    backgroundColor: colors.surface, borderRadius: radius.lg, borderWidth: 1, borderColor: colors.border,
    padding: spacing.md, marginBottom: spacing.md,
  },
  cardTitle: { fontSize: 16, fontWeight: "800", color: colors.textPrimary, flexShrink: 1 },
  cardBody: { fontSize: 14, lineHeight: 20, color: colors.textSecondary, marginTop: 6 },
  teachRow: { flexDirection: "row", gap: spacing.sm, alignItems: "center", marginTop: spacing.md, flexWrap: "wrap" },
  input: {
    flex: 1, minWidth: 180, borderWidth: 1, borderColor: colors.borderStrong, borderRadius: radius.md,
    paddingHorizontal: 12, paddingVertical: 10, fontSize: 15, color: colors.textPrimary, backgroundColor: colors.background,
  },
  teachProgress: { marginTop: spacing.md, gap: 6, alignItems: "flex-start" },
  teachProgressTitle: { fontSize: 18, fontWeight: "800", color: colors.accentStrong },
  teachNote: { fontSize: 13, lineHeight: 19, color: colors.textSecondary, marginTop: spacing.sm },
  customRow: {
    flexDirection: "row", alignItems: "center", justifyContent: "space-between",
    paddingVertical: 8, borderBottomWidth: 1, borderBottomColor: colors.border,
  },
  customLabel: { fontSize: 15, fontWeight: "700", color: colors.textPrimary },
  iconButton: { width: 34, height: 34, alignItems: "center", justifyContent: "center", borderRadius: radius.pill },
  vocabRow: { paddingVertical: 7, borderBottomWidth: 1, borderBottomColor: colors.border },
  vocabLabel: { fontSize: 14, fontWeight: "800", color: colors.textPrimary },
  vocabHow: { fontSize: 13, color: colors.textSecondary, marginTop: 1 },
  honest: { fontSize: 12, lineHeight: 18, color: colors.textMuted, marginTop: spacing.md },

  primaryButton: {
    flexDirection: "row", alignItems: "center", gap: 6, backgroundColor: colors.accent,
    paddingHorizontal: 14, paddingVertical: 10, borderRadius: radius.pill,
  },
  primaryText: { color: colors.textOnAccent, fontWeight: "800", fontSize: 14 },
  optionChip: {
    flexDirection: "row", alignItems: "center", gap: 6, backgroundColor: colors.surface, borderWidth: 1,
    borderColor: colors.border, paddingHorizontal: 12, paddingVertical: 8, borderRadius: radius.pill,
  },
  optionChipOn: { backgroundColor: colors.accentSoft, borderColor: colors.accent },
  optionText: { fontSize: 13, fontWeight: "700", color: colors.accentStrong },
  disabled: { opacity: 0.45 },
});
