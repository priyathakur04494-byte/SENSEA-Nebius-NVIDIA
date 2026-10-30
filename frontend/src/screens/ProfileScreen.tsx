import React, { useState } from "react";
import { Alert, Platform, Pressable, ScrollView, StyleSheet, Switch, Text, TextInput, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { useAuth } from "@/context/AuthContext";
import { useAppState } from "@/context/AppContext";
import { LANGUAGE_LABELS, SupportedLanguage } from "@/types";
import { colors, radius, shadow, spacing, typography } from "@/theme/colors";

const LANGUAGES = Object.keys(LANGUAGE_LABELS) as SupportedLanguage[];

export default function ProfileScreen() {
  const { user, logOut, updateProfile } = useAuth();
  const { targetLanguage, setTargetLanguage, showTranslation, setShowTranslation, showSimplified, setShowSimplified } =
    useAppState();

  const [isEditingName, setIsEditingName] = useState(false);
  const [nameDraft, setNameDraft] = useState(user?.name ?? "");

  if (!user) return null;

  const memberSince = new Date(user.createdAt).toLocaleDateString(undefined, {
    year: "numeric",
    month: "long",
    day: "numeric",
  });

  const saveName = async () => {
    const trimmed = nameDraft.trim();
    if (trimmed) await updateProfile({ name: trimmed });
    setIsEditingName(false);
  };

  // Alert.alert's button/onPress callbacks don't fire on the web renderer
  // (react-native-web only shows a plain OK dialog and ignores the buttons
  // array), which is why "Log out" appeared to do nothing when run in a
  // browser. Route web through window.confirm instead so the callback
  // actually runs there too; native platforms keep the normal Alert.
  const confirmLogOut = () => {
    if (Platform.OS === "web") {
      const confirmed = typeof window !== "undefined" ? window.confirm("Are you sure you want to log out?") : true;
      if (confirmed) logOut();
      return;
    }

    Alert.alert("Log out", "Are you sure you want to log out?", [
      { text: "Cancel", style: "cancel" },
      { text: "Log out", style: "destructive", onPress: () => logOut() },
    ]);
  };

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <View style={[styles.headerCard, shadow.md]}>
        <View style={styles.avatar}>
          <Text style={styles.avatarText}>{user.name[0]?.toUpperCase() ?? "?"}</Text>
        </View>

        {isEditingName ? (
          <View style={styles.editNameRow}>
            <TextInput
              style={styles.nameInput}
              value={nameDraft}
              onChangeText={setNameDraft}
              autoFocus
              accessibilityLabel="Edit name"
            />
            <Pressable style={styles.smallButton} onPress={saveName} accessibilityRole="button">
              <Text style={styles.smallButtonText}>Save</Text>
            </Pressable>
          </View>
        ) : (
          <Pressable
            onPress={() => {
              setNameDraft(user.name);
              setIsEditingName(true);
            }}
            style={styles.nameRow}
            accessibilityRole="button"
            accessibilityLabel="Edit name"
          >
            <Text style={styles.name}>{user.name}</Text>
            <Ionicons name="pencil-outline" size={15} color={colors.textMuted} style={{ marginLeft: spacing.xs }} />
          </Pressable>
        )}

        <Text style={styles.email}>{user.email}</Text>
        <Text style={styles.memberSince}>Member since {memberSince}</Text>
      </View>

      <Text style={styles.sectionLabel}>Preferences</Text>

      <View style={[styles.sectionCard, shadow.sm]}>
        <Text style={styles.rowLabel}>Preferred language</Text>
        <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.langRow}>
          {LANGUAGES.map((lang) => (
            <Pressable
              key={lang}
              onPress={() => {
                setTargetLanguage(lang);
                updateProfile({ preferredLanguage: lang });
              }}
              style={[styles.langChip, targetLanguage === lang && styles.langChipActive]}
            >
              <Text style={[styles.langChipText, targetLanguage === lang && styles.langChipTextActive]}>
                {LANGUAGE_LABELS[lang]}
              </Text>
            </Pressable>
          ))}
        </ScrollView>

        <View style={styles.switchRow}>
          <View style={{ flex: 1 }}>
            <Text style={styles.rowLabel}>Show translations in captions</Text>
            <Text style={styles.rowHint}>Display the translated line alongside the original text.</Text>
          </View>
          <Switch
            value={showTranslation}
            onValueChange={setShowTranslation}
            trackColor={{ false: colors.border, true: colors.accentSoft }}
            thumbColor={showTranslation ? colors.accent : colors.textMuted}
          />
        </View>

        <View style={styles.switchRow}>
          <View style={{ flex: 1 }}>
            <Text style={styles.rowLabel}>Show simplified captions</Text>
            <Text style={styles.rowHint}>Display a plain-language version alongside the original text.</Text>
          </View>
          <Switch
            value={showSimplified}
            onValueChange={setShowSimplified}
            trackColor={{ false: colors.border, true: colors.accentSoft }}
            thumbColor={showSimplified ? colors.accent : colors.textMuted}
          />
        </View>
      </View>

      <Text style={styles.sectionLabel}>Account</Text>
      <View style={[styles.sectionCard, shadow.sm]}>
        <Text style={styles.rowHint}>
          Your account and preferences are stored on this device. SENSEA doesn't use a backend server, so
          data won't sync to other devices yet.
        </Text>
      </View>

      <Pressable style={styles.logoutButton} onPress={confirmLogOut} accessibilityRole="button">
        <Text style={styles.logoutButtonText}>Log out</Text>
      </Pressable>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.background },
  content: { padding: spacing.md, paddingBottom: spacing.xl },
  headerCard: {
    alignItems: "center",
    backgroundColor: colors.surface,
    borderRadius: radius.lg,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.lg,
    marginBottom: spacing.lg,
  },
  avatar: {
    width: 72,
    height: 72,
    borderRadius: radius.pill,
    backgroundColor: colors.accentSoft,
    alignItems: "center",
    justifyContent: "center",
    marginBottom: spacing.sm,
  },
  avatarText: { ...typography.heading, fontSize: 30, color: colors.accent },
  name: { ...typography.title, color: colors.textPrimary },
  nameRow: { flexDirection: "row", alignItems: "center" },
  editNameRow: { flexDirection: "row", alignItems: "center", gap: spacing.xs, marginTop: spacing.xs },
  nameInput: {
    ...typography.title,
    color: colors.textPrimary,
    borderBottomWidth: 1,
    borderBottomColor: colors.accent,
    paddingVertical: 2,
    minWidth: 140,
    textAlign: "center",
  },
  smallButton: { backgroundColor: colors.accent, borderRadius: radius.sm, paddingHorizontal: spacing.sm, paddingVertical: 4 },
  smallButtonText: { ...typography.label, color: "#fff" },
  email: { ...typography.body, color: colors.textSecondary, marginTop: spacing.xs },
  memberSince: { ...typography.label, color: colors.textMuted, marginTop: 4 },
  sectionLabel: { ...typography.label, color: colors.textMuted, marginBottom: spacing.sm, letterSpacing: 1 },
  sectionCard: {
    backgroundColor: colors.surface,
    borderRadius: radius.lg,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.md,
    marginBottom: spacing.lg,
    gap: spacing.md,
  },
  rowLabel: { ...typography.body, color: colors.textPrimary, fontWeight: "600" },
  rowHint: { ...typography.label, color: colors.textMuted, marginTop: 2, lineHeight: 17 },
  langRow: { maxHeight: 44 },
  langChip: {
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    borderRadius: radius.pill,
    backgroundColor: colors.background,
    borderWidth: 1,
    borderColor: colors.border,
    marginRight: spacing.xs,
  },
  langChipActive: { backgroundColor: colors.accentSoft, borderColor: colors.accent },
  langChipText: { ...typography.body, fontSize: 13, color: colors.textSecondary },
  langChipTextActive: { color: colors.accentStrong, fontWeight: "700" },
  switchRow: { flexDirection: "row", alignItems: "center", gap: spacing.sm },
  logoutButton: {
    backgroundColor: colors.surface,
    borderRadius: radius.lg,
    borderWidth: 1,
    borderColor: colors.alertEmergency,
    paddingVertical: spacing.md,
    alignItems: "center",
  },
  logoutButtonText: { ...typography.title, color: colors.alertEmergency },
});
