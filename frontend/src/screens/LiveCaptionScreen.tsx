import React, { useRef, useEffect } from "react";
import {
  FlatList,
  Pressable,
  ScrollView,
  StyleSheet,
  Switch,
  Text,
  View,
  useWindowDimensions,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { useAppState } from "@/context/AppContext";
import CaptionBubble from "@/components/CaptionBubble";
import AlertBanner from "@/components/AlertBanner";
import ContextPanel from "@/components/ContextPanel";
import { breakpoints, colors, radius, shadow, spacing, typography } from "@/theme/colors";
import { LANGUAGE_LABELS, SupportedLanguage } from "@/types";

const LANGUAGES = Object.keys(LANGUAGE_LABELS) as SupportedLanguage[];

export default function LiveCaptionScreen() {
  const {
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
    latestSoundEvent,
    dismissLatestSoundEvent,
  } = useAppState();

  const listRef = useRef<FlatList>(null);
  const { width } = useWindowDimensions();
  // Wide viewports (tablet landscape / laptop / desktop) get the two-column
  // "Live Captions | AI Context" workspace layout from the spec; phones get
  // a single-column, caption-first view with Context reachable via its own
  // tab (see AppNavigator).
  const isWideLayout = width >= breakpoints.tablet;

  useEffect(() => {
    if (captions.length) listRef.current?.scrollToEnd({ animated: true });
  }, [captions.length]);

  const captionColumn = (
    <View style={[styles.captionColumn, isWideLayout && styles.captionColumnWide]}>
      {latestSoundEvent && (
        <Pressable onPress={dismissLatestSoundEvent} style={styles.bannerWrap}>
          <AlertBanner
            label={latestSoundEvent.label}
            level={latestSoundEvent.alertLevel}
            direction={latestSoundEvent.direction}
            timestamp={latestSoundEvent.timestamp}
          />
        </Pressable>
      )}

      <View style={styles.toolbar}>
        <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.langRow}>
          {LANGUAGES.map((lang) => (
            <Pressable
              key={lang}
              onPress={() => setTargetLanguage(lang)}
              style={[styles.langChip, targetLanguage === lang && styles.langChipActive]}
            >
              <Text style={[styles.langChipText, targetLanguage === lang && styles.langChipTextActive]}>
                {LANGUAGE_LABELS[lang]}
              </Text>
            </Pressable>
          ))}
        </ScrollView>

        <View style={styles.toggleRow}>
          <ToggleOption label="Translate" value={showTranslation} onChange={setShowTranslation} />
          <ToggleOption label="Simplify" value={showSimplified} onChange={setShowSimplified} />
        </View>
      </View>

      {captions.length === 0 ? (
        <View style={styles.emptyState}>
          <Ionicons name="mic-outline" size={26} color={colors.textMuted} />
          <Text style={styles.emptyTitle}>No captions yet</Text>
          <Text style={styles.emptyBody}>
            Tap "Start listening" below to begin live captioning the conversation around you.
          </Text>
        </View>
      ) : (
        <FlatList
          ref={listRef}
          data={captions}
          keyExtractor={(item) => item.id}
          contentContainerStyle={styles.list}
          renderItem={({ item }) => (
            <CaptionBubble
              caption={item}
              speaker={speakers[item.speakerId]}
              showSimplified={showSimplified}
              showTranslation={showTranslation}
            />
          )}
        />
      )}

      <Pressable
        onPress={toggleListening}
        style={[styles.micButton, shadow.sm, isListening && styles.micButtonActive]}
        accessibilityRole="button"
        accessibilityLabel={isListening ? "Stop listening" : "Start listening"}
      >
        <Ionicons name={isListening ? "stop-circle" : "mic"} size={20} color="#fff" style={{ marginRight: 8 }} />
        <Text style={styles.micButtonText}>{isListening ? "Stop listening" : "Start listening"}</Text>
      </Pressable>

      <View style={styles.statusRow}>
        <View style={styles.statusItem}>
          <View style={[styles.statusDot, isListening && styles.statusDotActive]} />
          <Text style={styles.statusText}>{isListening ? "Listening" : "Not listening"}</Text>
        </View>
      </View>
    </View>
  );

  if (!isWideLayout) {
    return <View style={styles.container}>{captionColumn}</View>;
  }

  return (
    <View style={[styles.container, styles.wideContainer]}>
      {captionColumn}
      <View style={styles.contextColumn}>
        <ContextPanel compact />
      </View>
    </View>
  );
}

function ToggleOption({
  label,
  value,
  onChange,
}: {
  label: string;
  value: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <View style={styles.toggleOption}>
      <Text style={styles.toggleLabel}>{label}</Text>
      <Switch
        value={value}
        onValueChange={onChange}
        trackColor={{ false: colors.border, true: colors.accentSoft }}
        thumbColor={value ? colors.accent : colors.textMuted}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.background },
  wideContainer: { flexDirection: "row" },
  captionColumn: { flex: 1, padding: spacing.md },
  captionColumnWide: {
    maxWidth: 720,
    borderRightWidth: 1,
    borderRightColor: colors.border,
  },
  contextColumn: { flex: 1, minWidth: 320, maxWidth: 420, backgroundColor: colors.background },
  bannerWrap: { marginBottom: spacing.md },
  toolbar: { marginBottom: spacing.sm },
  langRow: { gap: spacing.xs, paddingBottom: spacing.sm },
  langChip: {
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    borderRadius: radius.pill,
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.border,
    marginRight: spacing.xs,
  },
  langChipActive: { backgroundColor: colors.accentSoft, borderColor: colors.accent },
  langChipText: { ...typography.body, color: colors.textSecondary },
  langChipTextActive: { color: colors.accentStrong, fontWeight: "700" },
  toggleRow: { flexDirection: "row", gap: spacing.lg, marginTop: spacing.xs },
  toggleOption: { flexDirection: "row", alignItems: "center", gap: spacing.xs },
  toggleLabel: { ...typography.body, color: colors.textSecondary },
  list: { paddingBottom: spacing.xl },
  emptyState: { flex: 1, alignItems: "center", justifyContent: "center", paddingHorizontal: spacing.xl, gap: spacing.xs },
  emptyTitle: { ...typography.title, color: colors.textPrimary },
  emptyBody: { ...typography.body, color: colors.textSecondary, textAlign: "center" },
  micButton: {
    flexDirection: "row",
    backgroundColor: colors.accent,
    paddingVertical: spacing.md,
    borderRadius: radius.lg,
    alignItems: "center",
    justifyContent: "center",
    marginTop: spacing.sm,
  },
  micButtonActive: { backgroundColor: colors.critical },
  micButtonText: { ...typography.title, color: "#fff" },
  statusRow: { flexDirection: "row", justifyContent: "center", marginTop: spacing.sm },
  statusItem: { flexDirection: "row", alignItems: "center", gap: spacing.xs },
  statusDot: { width: 8, height: 8, borderRadius: radius.pill, backgroundColor: colors.textMuted },
  statusDotActive: { backgroundColor: colors.normal },
  statusText: { ...typography.label, color: colors.textMuted },
});
