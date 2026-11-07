import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  AccessibilityInfo,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  useWindowDimensions,
  View,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import SignStage from "@/components/sign/SignStage";
import SignRecognizer from "@/components/sign/SignRecognizer";
import HandSvg from "@/components/sign/HandSvg";
import {
  ASL_SIGNS,
  SIGN_CATEGORIES,
  SIGN_LANGUAGES,
  SignCategory,
  signsForCategory,
} from "@/config/signLanguageData";
import { speak, stopSpeaking } from "@/services/ttsService";
import { breakpoints, colors, radius, shadow, spacing, typography } from "@/theme/colors";

/**
 * SIGNS — a visual sign-language guide for communication.
 *
 * Signs play "turn by turn": the signer shows one sign step by step, then the
 * next sign takes its turn, with the meaning displayed alongside. The user can
 * pause, step manually, change speed, filter by topic, jump to any sign, and
 * (optionally) have the meaning spoken aloud — handy when signing with
 * someone who hears.
 */

const SPEEDS = [
  { label: "0.5×", factor: 0.5 },
  { label: "1×", factor: 1 },
  { label: "1.5×", factor: 1.5 },
];

const FRAME_HOLD_MS = 1700; // how long each step stays on screen at 1×
const SIGN_PAUSE_MS = 900; // extra beat on the last step before the next sign's turn

type Filter = SignCategory | "all";

function SignGuide() {
  const { width } = useWindowDimensions();
  const isWide = width >= breakpoints.tablet;

  const language = SIGN_LANGUAGES[0];
  const [filter, setFilter] = useState<Filter>("all");
  const [signIdx, setSignIdx] = useState(0);
  const [frameIdx, setFrameIdx] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [speedIdx, setSpeedIdx] = useState(1);
  const [voice, setVoice] = useState(false);
  const [reduceMotion, setReduceMotion] = useState(false);

  useEffect(() => {
    AccessibilityInfo.isReduceMotionEnabled?.().then(setReduceMotion).catch(() => {});
    const sub = AccessibilityInfo.addEventListener?.("reduceMotionChanged", setReduceMotion);
    return () => sub?.remove?.();
  }, []);

  const signs = useMemo(() => signsForCategory(filter), [filter]);
  const sign = signs[Math.min(signIdx, signs.length - 1)];
  const frame = sign.frames[Math.min(frameIdx, sign.frames.length - 1)];
  const speed = SPEEDS[speedIdx].factor;

  // ---- Turn-by-turn playback -------------------------------------------------
  const goToSign = useCallback(
    (index: number) => {
      const n = signs.length;
      setSignIdx(((index % n) + n) % n);
      setFrameIdx(0);
    },
    [signs.length]
  );

  useEffect(() => {
    if (!playing) return;
    const isLastFrame = frameIdx >= sign.frames.length - 1;
    const wait = (FRAME_HOLD_MS + (isLastFrame ? SIGN_PAUSE_MS : 0)) / speed;
    const t = setTimeout(() => {
      if (isLastFrame) {
        goToSign(signIdx + 1);
      } else {
        setFrameIdx(frameIdx + 1);
      }
    }, wait);
    return () => clearTimeout(t);
  }, [playing, frameIdx, signIdx, sign, speed, goToSign]);

  // Speak each sign's meaning when its turn starts (opt-in).
  useEffect(() => {
    if (voice && frameIdx === 0) speak(`${sign.word}. ${sign.meaning}`, "en");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [voice, sign.id]);

  useEffect(() => () => stopSpeaking(), []);

  const chooseFilter = (next: Filter) => {
    setFilter(next);
    setSignIdx(0);
    setFrameIdx(0);
  };

  const pickFrame = (i: number) => {
    setPlaying(false);
    setFrameIdx(i);
  };

  // ---- Layout ----------------------------------------------------------------
  const stageWidth = isWide ? 340 : Math.min(width - spacing.md * 2 - 2, 360);
  const durationMs = reduceMotion ? 0 : 650 / speed;

  const stage = (
    <View style={[styles.stageCard, shadow.md]}>
      <View style={styles.stageTopRow}>
        <View style={styles.counterPill}>
          <Ionicons name="hand-left-outline" size={13} color={colors.accentStrong} />
          <Text style={styles.counterText}>
            Sign {signIdx + 1} of {signs.length}
          </Text>
        </View>
        <View style={styles.counterPill}>
          <Text style={styles.counterText}>
            Step {frameIdx + 1} of {sign.frames.length}
          </Text>
        </View>
      </View>

      <SignStage
        frame={frame}
        signKey={sign.id}
        width={stageWidth}
        durationMs={durationMs}
        reduceMotion={reduceMotion}
        label={`Sign for ${sign.word}. Step ${frameIdx + 1} of ${sign.frames.length}: ${frame.instruction}`}
      />

      <View style={styles.captionBar} accessibilityLiveRegion="polite">
        <Text style={styles.captionText}>{frame.instruction}</Text>
      </View>

      {/* Transport controls */}
      <View style={styles.controls}>
        <ControlButton icon="play-skip-back" label="Previous sign" onPress={() => goToSign(signIdx - 1)} />
        <Pressable
          onPress={() => setPlaying((p) => !p)}
          accessibilityRole="button"
          accessibilityLabel={playing ? "Pause" : "Play"}
          style={[styles.playButton, shadow.md]}
        >
          <Ionicons name={playing ? "pause" : "play"} size={26} color={colors.textOnAccent} />
        </Pressable>
        <ControlButton icon="play-skip-forward" label="Next sign" onPress={() => goToSign(signIdx + 1)} />
      </View>

      <View style={styles.optionsRow}>
        <Pressable
          onPress={() => setSpeedIdx((i) => (i + 1) % SPEEDS.length)}
          accessibilityRole="button"
          accessibilityLabel={`Playback speed ${SPEEDS[speedIdx].label}. Tap to change.`}
          style={styles.optionChip}
        >
          <Ionicons name="speedometer-outline" size={15} color={colors.accentStrong} />
          <Text style={styles.optionText}>{SPEEDS[speedIdx].label}</Text>
        </Pressable>
        <Pressable
          onPress={() => {
            setFrameIdx(0);
            setPlaying(false);
            setTimeout(() => setPlaying(true), 60);
          }}
          accessibilityRole="button"
          accessibilityLabel="Replay this sign"
          style={styles.optionChip}
        >
          <Ionicons name="refresh" size={15} color={colors.accentStrong} />
          <Text style={styles.optionText}>Replay</Text>
        </Pressable>
        <Pressable
          onPress={() => {
            if (voice) stopSpeaking();
            setVoice((v) => !v);
          }}
          accessibilityRole="switch"
          accessibilityState={{ checked: voice }}
          accessibilityLabel="Speak each sign's meaning aloud"
          style={[styles.optionChip, voice && styles.optionChipOn]}
        >
          <Ionicons name={voice ? "volume-high" : "volume-mute-outline"} size={15} color={colors.accentStrong} />
          <Text style={styles.optionText}>{voice ? "Voice on" : "Voice off"}</Text>
        </Pressable>
      </View>
    </View>
  );

  const details = (
    <View>
      <View style={[styles.meaningCard, shadow.md]}>
        <View style={styles.meaningHeader}>
          <Text style={styles.signWord} accessibilityRole="header">
            {sign.word}
          </Text>
          <Pressable
            onPress={() => speak(`${sign.word}. ${sign.meaning}`, "en")}
            accessibilityRole="button"
            accessibilityLabel={`Speak the meaning of ${sign.word}`}
            style={styles.speakButton}
          >
            <Ionicons name="volume-high-outline" size={18} color={colors.accentStrong} />
          </Pressable>
        </View>

        <Text style={styles.fieldLabel}>MEANING</Text>
        <Text style={styles.meaningText}>{sign.meaning}</Text>

        <Text style={[styles.fieldLabel, { marginTop: spacing.md }]}>USE IT WHEN</Text>
        <Text style={styles.useText}>{sign.useWhen}</Text>

        <Text style={[styles.fieldLabel, { marginTop: spacing.md }]}>HOW TO SIGN IT</Text>
        {sign.frames.map((f, i) => {
          const active = i === frameIdx;
          return (
            <Pressable
              key={i}
              onPress={() => pickFrame(i)}
              accessibilityRole="button"
              accessibilityState={{ selected: active }}
              accessibilityLabel={`Step ${i + 1}: ${f.instruction}`}
              style={[styles.stepRow, active && styles.stepRowActive]}
            >
              <View style={[styles.stepBadge, active && styles.stepBadgeActive]}>
                {active ? (
                  <Ionicons name="hand-left" size={13} color={colors.textOnAccent} />
                ) : (
                  <Text style={styles.stepBadgeText}>{i + 1}</Text>
                )}
              </View>
              <Text style={[styles.stepText, active && styles.stepTextActive]}>{f.instruction}</Text>
            </Pressable>
          );
        })}
      </View>
    </View>
  );

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <Text style={styles.title}>Sign language</Text>
      <Text style={styles.subtitle}>
        Watch each sign turn by turn, with what it means and when to use it.
      </Text>

      <View style={styles.languagePill}>
        <Ionicons name="globe-outline" size={14} color={colors.supportBlue} />
        <Text style={styles.languagePillText}>{language.name}</Text>
      </View>

      <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.filterRow}>
        {[{ id: "all" as Filter, label: "All signs" }, ...SIGN_CATEGORIES].map((c) => {
          const active = filter === c.id;
          return (
            <Pressable
              key={c.id}
              onPress={() => chooseFilter(c.id as Filter)}
              accessibilityRole="button"
              accessibilityState={{ selected: active }}
              style={[styles.filterChip, active && styles.filterChipActive]}
            >
              {active && <Ionicons name="checkmark" size={14} color={colors.accentStrong} />}
              <Text style={[styles.filterText, active && styles.filterTextActive]}>{c.label}</Text>
            </Pressable>
          );
        })}
      </ScrollView>

      {isWide ? (
        <View style={styles.wideRow}>
          <View style={{ width: stageWidth + spacing.md * 2 + 2 }}>{stage}</View>
          <View style={{ flex: 1 }}>{details}</View>
        </View>
      ) : (
        <>
          {stage}
          {details}
        </>
      )}

      <Text style={styles.sectionLabel}>ALL SIGNS — TAP TO WATCH</Text>
      <View style={styles.grid}>
        {signs.map((s, i) => {
          const active = i === signIdx;
          return (
            <Pressable
              key={s.id}
              onPress={() => goToSign(i)}
              accessibilityRole="button"
              accessibilityState={{ selected: active }}
              accessibilityLabel={`${s.word}. ${s.meaning}`}
              style={[styles.gridCard, active && styles.gridCardActive]}
            >
              <View style={styles.gridHands}>
                {s.frames[0].hands.map((h, hi) => (
                  <HandSvg key={hi} shape={h.shape} mirrored={!!h.left} width={26} height={42} forearm={false} />
                ))}
              </View>
              <Text style={[styles.gridWord, active && styles.gridWordActive]} numberOfLines={2}>
                {s.word}
              </Text>
              {active && <Ionicons name="radio-button-on" size={12} color={colors.accentStrong} />}
            </Pressable>
          );
        })}
      </View>

      <View style={styles.noteBox}>
        <Ionicons name="information-circle-outline" size={18} color={colors.supportBlue} />
        <Text style={styles.noteText}>{language.note}</Text>
      </View>
    </ScrollView>
  );
}

function ControlButton({
  icon,
  label,
  onPress,
}: {
  icon: keyof typeof Ionicons.glyphMap;
  label: string;
  onPress: () => void;
}) {
  return (
    <Pressable onPress={onPress} accessibilityRole="button" accessibilityLabel={label} style={styles.controlButton}>
      <Ionicons name={icon} size={22} color={colors.accentStrong} />
    </Pressable>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.background },
  content: { padding: spacing.md, paddingBottom: spacing.xl * 2 },
  title: { ...typography.heading, color: colors.textPrimary },
  subtitle: { ...typography.body, color: colors.textSecondary, marginTop: spacing.xs, marginBottom: spacing.sm },

  languagePill: {
    flexDirection: "row",
    alignItems: "center",
    alignSelf: "flex-start",
    gap: 6,
    backgroundColor: colors.supportBlueSoft,
    borderRadius: radius.pill,
    paddingVertical: 4,
    paddingHorizontal: spacing.sm + 2,
    marginBottom: spacing.sm,
  },
  languagePillText: { ...typography.label, color: colors.supportBlue },

  filterRow: { marginBottom: spacing.md, flexGrow: 0 },
  filterChip: {
    flexDirection: "row",
    alignItems: "center",
    gap: 4,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    borderRadius: radius.pill,
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.border,
    marginRight: spacing.xs,
  },
  filterChipActive: { backgroundColor: colors.accentSoft, borderColor: colors.accent },
  filterText: { ...typography.body, color: colors.textSecondary },
  filterTextActive: { color: colors.accentStrong, fontWeight: "700" },

  wideRow: { flexDirection: "row", gap: spacing.md, alignItems: "flex-start" },

  stageCard: {
    backgroundColor: colors.surface,
    borderRadius: radius.lg,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.md,
    marginBottom: spacing.md,
  },
  stageTopRow: { flexDirection: "row", justifyContent: "space-between", marginBottom: spacing.sm },
  counterPill: {
    flexDirection: "row",
    alignItems: "center",
    gap: 5,
    backgroundColor: colors.accentSoft,
    borderRadius: radius.pill,
    paddingVertical: 3,
    paddingHorizontal: spacing.sm + 2,
  },
  counterText: { ...typography.label, fontSize: 12, color: colors.accentStrong },
  captionBar: {
    backgroundColor: colors.primary,
    borderRadius: radius.md,
    paddingVertical: spacing.sm + 2,
    paddingHorizontal: spacing.md,
    marginTop: spacing.sm,
    minHeight: 64,
    justifyContent: "center",
  },
  captionText: { ...typography.body, color: colors.textOnDark, fontWeight: "600" },

  controls: { flexDirection: "row", alignItems: "center", justifyContent: "center", gap: spacing.lg, marginTop: spacing.md },
  controlButton: {
    width: 48,
    height: 48,
    borderRadius: radius.pill,
    backgroundColor: colors.surfaceMuted,
    alignItems: "center",
    justifyContent: "center",
  },
  playButton: {
    width: 62,
    height: 62,
    borderRadius: radius.pill,
    backgroundColor: colors.accent,
    alignItems: "center",
    justifyContent: "center",
  },
  optionsRow: { flexDirection: "row", flexWrap: "wrap", justifyContent: "center", gap: spacing.sm, marginTop: spacing.md },
  optionChip: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.md,
    borderRadius: radius.pill,
    backgroundColor: colors.surfaceMuted,
    borderWidth: 1,
    borderColor: colors.border,
  },
  optionChipOn: { backgroundColor: colors.accentSoft, borderColor: colors.accent },
  optionText: { ...typography.label, color: colors.accentStrong },

  meaningCard: {
    backgroundColor: colors.surface,
    borderRadius: radius.lg,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.md,
    marginBottom: spacing.md,
  },
  meaningHeader: { flexDirection: "row", alignItems: "center", justifyContent: "space-between", marginBottom: spacing.sm },
  signWord: { ...typography.heading, fontSize: 30, lineHeight: 36, color: colors.textPrimary, flex: 1 },
  speakButton: {
    width: 40,
    height: 40,
    borderRadius: radius.pill,
    backgroundColor: colors.accentSoft,
    alignItems: "center",
    justifyContent: "center",
  },
  fieldLabel: { ...typography.label, fontSize: 11, color: colors.textMuted, letterSpacing: 1, marginBottom: 2 },
  meaningText: { ...typography.caption, fontSize: 20, lineHeight: 28, color: colors.textPrimary },
  useText: { ...typography.body, color: colors.textSecondary },

  stepRow: {
    flexDirection: "row",
    alignItems: "flex-start",
    gap: spacing.sm,
    padding: spacing.sm,
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: "transparent",
    marginTop: 4,
  },
  stepRowActive: { backgroundColor: colors.accentSoft, borderColor: colors.accent },
  stepBadge: {
    width: 24,
    height: 24,
    borderRadius: radius.pill,
    backgroundColor: colors.surfaceMuted,
    alignItems: "center",
    justifyContent: "center",
    marginTop: 1,
  },
  stepBadgeActive: { backgroundColor: colors.accent },
  stepBadgeText: { ...typography.label, fontSize: 12, color: colors.textSecondary },
  stepText: { ...typography.body, flex: 1, color: colors.textSecondary },
  stepTextActive: { color: colors.textPrimary, fontWeight: "600" },

  sectionLabel: {
    ...typography.label,
    color: colors.textMuted,
    letterSpacing: 1,
    marginTop: spacing.sm,
    marginBottom: spacing.sm,
  },
  grid: { flexDirection: "row", flexWrap: "wrap", gap: spacing.sm },
  gridCard: {
    width: 104,
    alignItems: "center",
    gap: 6,
    backgroundColor: colors.surface,
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: colors.border,
    paddingVertical: spacing.sm + 2,
    paddingHorizontal: spacing.xs,
  },
  gridCardActive: { backgroundColor: colors.accentSoft, borderColor: colors.accent, borderWidth: 2 },
  gridHands: { flexDirection: "row", gap: 2, height: 44, alignItems: "center" },
  gridWord: { ...typography.label, color: colors.textSecondary, textAlign: "center", minHeight: 34 },
  gridWordActive: { color: colors.accentStrong },

  noteBox: {
    flexDirection: "row",
    gap: spacing.sm,
    alignItems: "flex-start",
    backgroundColor: colors.supportBlueSoft,
    borderRadius: radius.md,
    padding: spacing.md,
    marginTop: spacing.lg,
  },
  noteText: { ...typography.label, fontWeight: "400", color: colors.textSecondary, flex: 1, lineHeight: 19 },
});


/**
 * The Signs tab has two modes:
 *  - Learn: the illustrated guide (what this screen always did)
 *  - Recognise: point the camera at your hands and SENSEA says what you sign
 */
type Mode = "learn" | "recognise";

export default function SignLanguageScreen() {
  const [mode, setMode] = useState<Mode>("learn");
  return (
    <View style={{ flex: 1, backgroundColor: colors.background }}>
      <View style={modeStyles.bar} accessibilityRole="tablist">
        {([
          { id: "learn" as Mode, label: "Learn signs", icon: "play-circle-outline" as const },
          { id: "recognise" as Mode, label: "Sign to text", icon: "videocam-outline" as const },
        ]).map((m) => {
          const active = mode === m.id;
          return (
            <Pressable
              key={m.id}
              onPress={() => setMode(m.id)}
              accessibilityRole="tab"
              accessibilityState={{ selected: active }}
              accessibilityLabel={m.id === "learn" ? "Learn signs" : "Sign to text using the camera"}
              style={[modeStyles.tab, active && modeStyles.tabActive]}
            >
              <Ionicons name={m.icon} size={17} color={active ? colors.textOnAccent : colors.accentStrong} />
              <Text style={[modeStyles.tabText, active && modeStyles.tabTextActive]}>{m.label}</Text>
            </Pressable>
          );
        })}
      </View>
      {mode === "learn" ? (
        <SignGuide />
      ) : (
        <ScrollView style={{ flex: 1 }} contentContainerStyle={{ padding: spacing.md, paddingBottom: spacing.xl }}>
          <SignRecognizer />
        </ScrollView>
      )}
    </View>
  );
}

const modeStyles = StyleSheet.create({
  bar: {
    flexDirection: "row", gap: spacing.sm, padding: spacing.sm, paddingHorizontal: spacing.md,
    backgroundColor: colors.surface, borderBottomWidth: 1, borderBottomColor: colors.border,
  },
  tab: {
    flex: 1, flexDirection: "row", alignItems: "center", justifyContent: "center", gap: 6,
    paddingVertical: 10, borderRadius: radius.pill, backgroundColor: colors.accentSoft,
  },
  tabActive: { backgroundColor: colors.accent },
  tabText: { fontSize: 14, fontWeight: "800", color: colors.accentStrong },
  tabTextActive: { color: colors.textOnAccent },
});
