import React, { useState } from "react";
import { ActivityIndicator, Pressable, ScrollView, StyleSheet, Text, TextInput, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { speak, stopSpeaking } from "@/services/ttsService";
import { translateText, isPendingPlaceholder } from "@/services/translationService";
import { LANGUAGE_LABELS, SupportedLanguage } from "@/types";
import { colors, radius, shadow, spacing, typography } from "@/theme/colors";

const QUICK_PHRASES = [
  "Hi, I'm Deaf. I'm typing so the app can speak for me.",
  "Could you please repeat that?",
  "One moment, please.",
  "Thank you so much.",
  "Could you write it down instead?",
];

const LANGUAGES = Object.keys(LANGUAGE_LABELS) as SupportedLanguage[];

export default function TypeToSpeakScreen() {
  const [message, setMessage] = useState("");
  const [language, setLanguage] = useState<SupportedLanguage>("en");
  const [history, setHistory] = useState<string[]>([]);
  const [isTranslating, setIsTranslating] = useState(false);
  const [note, setNote] = useState<string | null>(null);

  const handleSpeak = async (text: string) => {
    if (!text.trim()) return;
    setNote(null);

    if (language === "en") {
      speak(text, "en");
      setHistory((prev) => [text, ...prev].slice(0, 20));
      return;
    }

    // Non-English selected: translate the message first, then speak the
    // translated text with that language's voice. Picking a language here
    // changes what gets *said*, not just the accent used to say it.
    setIsTranslating(true);
    try {
      const translated = await translateText(text, language);
      if (isPendingPlaceholder(translated)) {
        // No real translation available yet for this text — speak the
        // original rather than reading the placeholder bracket text aloud,
        // and tell the user why.
        speak(text, "en");
        setNote(
          `Translation to ${LANGUAGE_LABELS[language]} isn't connected yet for this phrase, so this was spoken in English. See the README for wiring up a real translation API.`
        );
        setHistory((prev) => [text, ...prev].slice(0, 20));
      } else {
        speak(translated, language);
        setHistory((prev) => [translated, ...prev].slice(0, 20));
      }
    } finally {
      setIsTranslating(false);
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Type to speak</Text>
      <Text style={styles.subtitle}>
        Type a message in English — your device will translate and speak it in the language you pick below.
      </Text>

      <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.langRow}>
        {LANGUAGES.map((lang) => (
          <Pressable
            key={lang}
            onPress={() => { setLanguage(lang); setNote(null); }}
            style={[styles.langChip, language === lang && styles.langChipActive]}
          >
            <Text style={[styles.langChipText, language === lang && styles.langChipTextActive]}>
              {LANGUAGE_LABELS[lang]}
            </Text>
          </Pressable>
        ))}
      </ScrollView>

      <View style={styles.quickRow}>
        {QUICK_PHRASES.map((phrase) => (
          <Pressable key={phrase} style={styles.quickChip} onPress={() => handleSpeak(phrase)}>
            <Text style={styles.quickChipText} numberOfLines={1}>{phrase}</Text>
          </Pressable>
        ))}
      </View>

      <TextInput
        style={styles.input}
        placeholder="Type your message…"
        placeholderTextColor={colors.textMuted}
        value={message}
        onChangeText={setMessage}
        multiline
      />

      <View style={styles.buttonRow}>
        <Pressable style={[styles.speakButton, shadow.md]} onPress={() => handleSpeak(message)} disabled={isTranslating}>
          {isTranslating ? (
            <ActivityIndicator color="#fff" />
          ) : (
            <>
              <Ionicons name="volume-high" size={18} color="#fff" style={{ marginRight: 6 }} />
              <Text style={styles.speakButtonText}>Speak</Text>
            </>
          )}
        </Pressable>
        <Pressable style={styles.stopButton} onPress={stopSpeaking}>
          <Text style={styles.stopButtonText}>Stop</Text>
        </Pressable>
      </View>

      {note && <Text style={styles.note}>{note}</Text>}

      {history.length > 0 && (
        <View style={styles.historyWrap}>
          <Text style={styles.historyTitle}>Recently spoken</Text>
          <ScrollView>
            {history.map((h, i) => (
              <Pressable key={`${h}-${i}`} onPress={() => speak(h, language)}>
                <Text style={styles.historyItem}>{h}</Text>
              </Pressable>
            ))}
          </ScrollView>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.background, padding: spacing.md },
  title: { ...typography.heading, color: colors.textPrimary },
  subtitle: { ...typography.body, color: colors.textSecondary, marginTop: spacing.xs, marginBottom: spacing.md },
  langRow: { marginBottom: spacing.sm, maxHeight: 44 },
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
  quickRow: { flexDirection: "row", flexWrap: "wrap", gap: spacing.xs, marginBottom: spacing.md },
  quickChip: {
    backgroundColor: colors.surface,
    borderRadius: radius.pill,
    borderWidth: 1,
    borderColor: colors.border,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    maxWidth: "100%",
  },
  quickChipText: { ...typography.label, color: colors.textSecondary },
  input: {
    backgroundColor: colors.surface,
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.md,
    minHeight: 100,
    ...typography.caption,
    color: colors.textPrimary,
    textAlignVertical: "top",
  },
  buttonRow: { flexDirection: "row", gap: spacing.sm, marginTop: spacing.md },
  speakButton: {
    flex: 1,
    flexDirection: "row",
    backgroundColor: colors.accent,
    borderRadius: radius.lg,
    paddingVertical: spacing.md,
    alignItems: "center",
    justifyContent: "center",
  },
  speakButtonText: { ...typography.title, color: "#fff" },
  stopButton: {
    backgroundColor: colors.surface,
    borderRadius: radius.lg,
    paddingVertical: spacing.md,
    paddingHorizontal: spacing.lg,
    alignItems: "center",
    borderWidth: 1,
    borderColor: colors.border,
  },
  stopButtonText: { ...typography.title, color: colors.textSecondary },
  note: {
    ...typography.label,
    color: colors.alertImportant,
    marginTop: spacing.sm,
    lineHeight: 18,
  },
  historyWrap: { marginTop: spacing.lg, flex: 1 },
  historyTitle: { ...typography.label, color: colors.textMuted, marginBottom: spacing.xs },
  historyItem: {
    ...typography.body,
    color: colors.textSecondary,
    paddingVertical: spacing.sm,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
});
