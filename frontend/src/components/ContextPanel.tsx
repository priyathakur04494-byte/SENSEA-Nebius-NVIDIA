import React, { useMemo, useState } from "react";
import { Pressable, ScrollView, StyleSheet, Text, TextInput, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { useAppState } from "@/context/AppContext";
import { colors, radius, spacing, typography } from "@/theme/colors";
import {
  SUGGESTED_QUESTIONS,
  answerFromTranscript,
  buildDecisions,
  buildImportantPoints,
  buildLiveSummary,
  buildMissed,
  toTranscriptLines,
} from "@/utils/contextInsights";

/**
 * The "AI Context" panel — what SENSEA understands about the live
 * conversation. Renders full-width as its own screen on phones, or as the
 * secondary column next to Live Captions on wider (tablet/laptop/desktop)
 * layouts. All content is derived from the existing caption transcript —
 * see src/utils/contextInsights.ts.
 */
export default function ContextPanel({ compact = false }: { compact?: boolean }) {
  const { captions, speakers } = useAppState();
  const [question, setQuestion] = useState("");
  const [qaHistory, setQaHistory] = useState<{ q: string; a: string }[]>([]);
  const [acknowledgedCount, setAcknowledgedCount] = useState(0);

  const lines = useMemo(() => toTranscriptLines(captions, speakers), [captions, speakers]);
  const summary = useMemo(() => buildLiveSummary(lines), [lines]);
  const importantPoints = useMemo(() => buildImportantPoints(lines), [lines]);
  const decisions = useMemo(() => buildDecisions(lines), [lines]);
  const missed = useMemo(() => buildMissed(lines, acknowledgedCount), [lines, acknowledgedCount]);

  const ask = (q: string) => {
    if (!q.trim()) return;
    const a = answerFromTranscript(q, lines);
    setQaHistory((prev) => [{ q, a }, ...prev]);
    setQuestion("");
  };

  const hasContent = lines.length > 0;

  return (
    <ScrollView
      style={styles.container}
      contentContainerStyle={[styles.content, compact && styles.contentCompact]}
      showsVerticalScrollIndicator={false}
    >
      {!compact && <Text style={styles.panelTitle}>AI Context</Text>}

      {!hasContent ? (
        <EmptyState />
      ) : (
        <>
          <Section icon="pulse-outline" title="Live summary">
            <Text style={styles.summaryText}>{summary}</Text>
          </Section>

          <Section icon="list-outline" title="Important points">
            {importantPoints.length === 0 ? (
              <MutedNote text="Nothing time-sensitive has come up yet." />
            ) : (
              importantPoints.slice(-5).map((p) => (
                <BulletRow key={p.id} text={p.text} />
              ))
            )}
          </Section>

          <Section icon="checkmark-done-outline" title="Decisions">
            {decisions.length === 0 ? (
              <MutedNote text="No decisions reached yet." />
            ) : (
              decisions.slice(-5).map((d) => <CheckRow key={d.id} text={d.text} />)
            )}
          </Section>

          <MissedSection missed={missed} onCaughtUp={() => setAcknowledgedCount(lines.length)} />
        </>
      )}

      <Section icon="chatbubbles-outline" title="Ask SENSEA">
        <View style={styles.askRow}>
          <TextInput
            style={styles.askInput}
            placeholder="Ask anything about this conversation…"
            placeholderTextColor={colors.textMuted}
            value={question}
            onChangeText={setQuestion}
            onSubmitEditing={() => ask(question)}
            accessibilityLabel="Ask SENSEA a question"
          />
          <Pressable
            style={styles.askButton}
            onPress={() => ask(question)}
            accessibilityRole="button"
            accessibilityLabel="Send question"
          >
            <Ionicons name="arrow-up" size={18} color="#fff" />
          </Pressable>
        </View>

        <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.suggestedRow}>
          {SUGGESTED_QUESTIONS.map((q) => (
            <Pressable key={q} style={styles.suggestedChip} onPress={() => ask(q)}>
              <Text style={styles.suggestedChipText}>{q}</Text>
            </Pressable>
          ))}
        </ScrollView>

        {qaHistory.map((item, i) => (
          <View key={`${item.q}-${i}`} style={styles.qaCard}>
            <Text style={styles.qaQuestion}>{item.q}</Text>
            <Text style={styles.qaAnswer}>{item.a}</Text>
          </View>
        ))}
      </Section>
    </ScrollView>
  );
}

function Section({
  icon,
  title,
  children,
}: {
  icon: keyof typeof Ionicons.glyphMap;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <View style={styles.section}>
      <View style={styles.sectionHeader}>
        <Ionicons name={icon} size={16} color={colors.accentStrong} />
        <Text style={styles.sectionTitle}>{title}</Text>
      </View>
      {children}
    </View>
  );
}

function BulletRow({ text }: { text: string }) {
  return (
    <View style={styles.row}>
      <View style={styles.dot} />
      <Text style={styles.rowText}>{text}</Text>
    </View>
  );
}

function CheckRow({ text }: { text: string }) {
  return (
    <View style={styles.row}>
      <Ionicons name="checkmark" size={16} color={colors.normal} style={{ marginTop: 2 }} />
      <Text style={styles.rowText}>{text}</Text>
    </View>
  );
}

function MutedNote({ text }: { text: string }) {
  return <Text style={styles.mutedNote}>{text}</Text>;
}

function MissedSection({ missed, onCaughtUp }: { missed: { id: string; text: string }[]; onCaughtUp: () => void }) {
  return (
    <View style={[styles.missedCard, missed.length > 0 && styles.missedCardActive]}>
      <View style={styles.sectionHeader}>
        <Ionicons
          name={missed.length > 0 ? "alert-circle" : "checkmark-circle-outline"}
          size={16}
          color={missed.length > 0 ? colors.warning : colors.normal}
        />
        <Text style={styles.sectionTitle}>You may have missed</Text>
        {missed.length > 0 && (
          <View style={styles.missedBadge}>
            <Text style={styles.missedBadgeText}>{missed.length}</Text>
          </View>
        )}
      </View>

      {missed.length === 0 ? (
        <MutedNote text="You're caught up — nothing pending." />
      ) : (
        <>
          {missed.map((m, i) => (
            <View key={m.id} style={styles.row}>
              <Text style={styles.missedIndex}>{i + 1}</Text>
              <Text style={styles.rowText}>{m.text}</Text>
            </View>
          ))}
          <Pressable
            style={styles.caughtUpButton}
            onPress={onCaughtUp}
            accessibilityRole="button"
            accessibilityLabel="Mark as caught up"
          >
            <Text style={styles.caughtUpButtonText}>Mark as caught up</Text>
          </Pressable>
        </>
      )}
    </View>
  );
}

function EmptyState() {
  return (
    <View style={styles.emptyState}>
      <Ionicons name="sparkles-outline" size={26} color={colors.textMuted} />
      <Text style={styles.emptyTitle}>No context yet</Text>
      <Text style={styles.emptyBody}>
        Start listening from the Live tab — SENSEA will summarize what's happening here as the
        conversation goes.
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.background },
  content: { padding: spacing.md, paddingBottom: spacing.xl },
  contentCompact: { padding: spacing.sm },
  panelTitle: { ...typography.title, color: colors.textPrimary, marginBottom: spacing.md },
  section: {
    backgroundColor: colors.surface,
    borderRadius: radius.lg,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.md,
    marginBottom: spacing.sm,
  },
  sectionHeader: { flexDirection: "row", alignItems: "center", gap: spacing.xs, marginBottom: spacing.sm },
  sectionTitle: { ...typography.label, color: colors.textPrimary, letterSpacing: 0.4, flex: 1 },
  summaryText: { ...typography.body, color: colors.textSecondary, lineHeight: 21 },
  row: { flexDirection: "row", gap: spacing.sm, marginBottom: spacing.xs, alignItems: "flex-start" },
  dot: { width: 6, height: 6, borderRadius: radius.pill, backgroundColor: colors.accent, marginTop: 8 },
  rowText: { ...typography.body, color: colors.textPrimary, flex: 1 },
  mutedNote: { ...typography.body, color: colors.textMuted, fontStyle: "italic" },
  missedCard: {
    backgroundColor: colors.surface,
    borderRadius: radius.lg,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.md,
    marginBottom: spacing.sm,
  },
  missedCardActive: { backgroundColor: colors.warningSoft, borderColor: colors.warning },
  missedBadge: {
    backgroundColor: colors.warning,
    borderRadius: radius.pill,
    minWidth: 20,
    height: 20,
    alignItems: "center",
    justifyContent: "center",
    paddingHorizontal: 6,
  },
  missedBadgeText: { color: "#fff", fontSize: 11, fontWeight: "700" },
  missedIndex: { ...typography.label, color: colors.warning, width: 16 },
  caughtUpButton: {
    alignSelf: "flex-start",
    marginTop: spacing.xs,
    minHeight: 40,
    justifyContent: "center",
    paddingVertical: spacing.xs,
    paddingHorizontal: spacing.md,
    borderRadius: radius.pill,
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.warning,
  },
  caughtUpButtonText: { ...typography.label, color: colors.warning },
  askRow: { flexDirection: "row", gap: spacing.xs, marginBottom: spacing.sm },
  askInput: {
    flex: 1,
    backgroundColor: colors.surfaceMuted,
    borderRadius: radius.md,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    color: colors.textPrimary,
    borderWidth: 1,
    borderColor: colors.border,
  },
  askButton: {
    backgroundColor: colors.accent,
    borderRadius: radius.md,
    width: 44,
    minHeight: 44,
    alignItems: "center",
    justifyContent: "center",
  },
  suggestedRow: { marginBottom: spacing.sm },
  suggestedChip: {
    backgroundColor: colors.accentSoft,
    borderRadius: radius.pill,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.xs,
    marginRight: spacing.xs,
  },
  suggestedChipText: { ...typography.label, color: colors.accentStrong },
  qaCard: {
    backgroundColor: colors.surfaceMuted,
    borderRadius: radius.md,
    padding: spacing.sm,
    marginBottom: spacing.xs,
  },
  qaQuestion: { ...typography.body, color: colors.textPrimary, fontWeight: "700" },
  qaAnswer: { ...typography.body, color: colors.textSecondary, marginTop: 2 },
  emptyState: { alignItems: "center", padding: spacing.xl, gap: spacing.xs },
  emptyTitle: { ...typography.title, fontSize: 16, color: colors.textPrimary, marginTop: spacing.xs },
  emptyBody: { ...typography.body, color: colors.textMuted, textAlign: "center" },
});
