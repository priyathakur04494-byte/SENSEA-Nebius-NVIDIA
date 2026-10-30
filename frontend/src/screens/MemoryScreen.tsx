import React, { useCallback, useEffect, useState } from "react";
import { FlatList, Pressable, StyleSheet, Switch, Text, TextInput, View } from "react-native";
import { useFocusEffect } from "@react-navigation/native";
import { RememberedDeadline, RememberedPreference, Speaker } from "@/types";
import {
  addPreference,
  getDeadlines,
  getPreferences,
  getSpeakers,
  removeDeadline,
  removePreference,
  upsertSpeaker,
} from "@/services/memoryService";
import { colors, radius, shadow, spacing, typography } from "@/theme/colors";

export default function MemoryScreen() {
  const [speakers, setSpeakers] = useState<Speaker[]>([]);
  const [deadlines, setDeadlines] = useState<RememberedDeadline[]>([]);
  const [preferences, setPreferences] = useState<RememberedPreference[]>([]);
  const [newPref, setNewPref] = useState("");

  const load = useCallback(async () => {
    setSpeakers(await getSpeakers());
    setDeadlines(await getDeadlines());
    setPreferences(await getPreferences());
  }, []);

  useFocusEffect(
    useCallback(() => {
      load();
    }, [load])
  );

  const renameSpeaker = async (speaker: Speaker, name: string) => {
    const updated = { ...speaker, name };
    await upsertSpeaker(updated);
    setSpeakers((prev) => prev.map((s) => (s.id === speaker.id ? updated : s)));
  };

  const toggleImportant = async (speaker: Speaker) => {
    const updated = { ...speaker, isImportantPerson: !speaker.isImportantPerson };
    await upsertSpeaker(updated);
    setSpeakers((prev) => prev.map((s) => (s.id === speaker.id ? updated : s)));
  };

  const submitPreference = async () => {
    if (!newPref.trim()) return;
    const pref: RememberedPreference = {
      id: `pref_${Date.now()}`,
      category: "general",
      detail: newPref.trim(),
      createdAt: Date.now(),
    };
    await addPreference(pref);
    setNewPref("");
    setPreferences(await getPreferences());
  };

  return (
    <FlatList
      style={styles.container}
      contentContainerStyle={{ paddingBottom: spacing.xl }}
      data={[{ key: "content" }]}
      renderItem={() => (
        <View>
          <Section title="People">
            {speakers.length === 0 && <EmptyNote text="No speakers recognized yet. Start captioning to build this list." />}
            {speakers.map((speaker) => (
              <View key={speaker.id} style={[styles.speakerRow, shadow.sm]}>
                <View style={[styles.dot, { backgroundColor: speaker.colorTag }]} />
                <TextInput
                  style={styles.speakerInput}
                  placeholder={speaker.label}
                  placeholderTextColor={colors.textMuted}
                  defaultValue={speaker.name}
                  onEndEditing={(e) => renameSpeaker(speaker, e.nativeEvent.text)}
                />
                <View style={styles.importantToggle}>
                  <Text style={styles.importantLabel}>Important</Text>
                  <Switch
                    value={speaker.isImportantPerson}
                    onValueChange={() => toggleImportant(speaker)}
                    trackColor={{ false: colors.border, true: colors.accentSoft }}
                    thumbColor={speaker.isImportantPerson ? colors.accent : colors.textMuted}
                  />
                </View>
              </View>
            ))}
          </Section>

          <Section title="Deadlines mentioned in conversation">
            {deadlines.length === 0 && <EmptyNote text="No deadlines captured yet." />}
            {deadlines.map((d) => (
              <View key={d.id} style={[styles.card, shadow.sm]}>
                <Text style={styles.cardTitle}>{d.description}</Text>
                <Text style={styles.cardSubtitle}>
                  Due {new Date(d.dueDate).toLocaleDateString()}
                </Text>
                <Pressable onPress={async () => { await removeDeadline(d.id); setDeadlines(await getDeadlines()); }}>
                  <Text style={styles.removeLink}>Remove</Text>
                </Pressable>
              </View>
            ))}
          </Section>

          <Section title="Preferences">
            <View style={styles.addRow}>
              <TextInput
                style={styles.addInput}
                placeholder="e.g. Prefers text over calls"
                placeholderTextColor={colors.textMuted}
                value={newPref}
                onChangeText={setNewPref}
                onSubmitEditing={submitPreference}
              />
              <Pressable style={styles.addButton} onPress={submitPreference}>
                <Text style={styles.addButtonText}>Add</Text>
              </Pressable>
            </View>
            {preferences.length === 0 && <EmptyNote text="No preferences saved yet." />}
            {preferences.map((p) => (
              <View key={p.id} style={[styles.card, shadow.sm]}>
                <Text style={styles.cardTitle}>{p.detail}</Text>
                <Pressable onPress={async () => { await removePreference(p.id); setPreferences(await getPreferences()); }}>
                  <Text style={styles.removeLink}>Remove</Text>
                </Pressable>
              </View>
            ))}
          </Section>
        </View>
      )}
    />
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <View style={styles.section}>
      <Text style={styles.sectionTitle}>{title}</Text>
      {children}
    </View>
  );
}

function EmptyNote({ text }: { text: string }) {
  return <Text style={styles.emptyNote}>{text}</Text>;
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.background, padding: spacing.md },
  section: { marginBottom: spacing.lg },
  sectionTitle: { ...typography.title, color: colors.textPrimary, marginBottom: spacing.sm },
  emptyNote: { ...typography.body, color: colors.textMuted, fontStyle: "italic" },
  speakerRow: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: colors.surface,
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.sm,
    marginBottom: spacing.xs,
    gap: spacing.sm,
  },
  dot: { width: 12, height: 12, borderRadius: radius.pill },
  speakerInput: { flex: 1, ...typography.body, color: colors.textPrimary },
  importantToggle: { flexDirection: "row", alignItems: "center", gap: spacing.xs },
  importantLabel: { ...typography.label, color: colors.textSecondary },
  card: {
    backgroundColor: colors.surface,
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.md,
    marginBottom: spacing.xs,
  },
  cardTitle: { ...typography.body, color: colors.textPrimary, fontWeight: "600" },
  cardSubtitle: { ...typography.label, color: colors.textSecondary, marginTop: 2 },
  removeLink: { ...typography.label, color: colors.alertEmergency, marginTop: spacing.xs },
  addRow: { flexDirection: "row", gap: spacing.xs, marginBottom: spacing.sm },
  addInput: {
    flex: 1,
    backgroundColor: colors.surface,
    borderRadius: radius.md,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    color: colors.textPrimary,
    borderWidth: 1,
    borderColor: colors.border,
  },
  addButton: {
    backgroundColor: colors.accent,
    borderRadius: radius.md,
    paddingHorizontal: spacing.md,
    justifyContent: "center",
  },
  addButtonText: { ...typography.body, color: "#fff", fontWeight: "700" },
});
