import React from "react";
import { Pressable, StyleSheet, Text, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { AppMode, MODE_LABELS } from "@/types";
import { MODES, MODE_DESCRIPTIONS, MODE_ICONS } from "@/config/modeConfig";
import { colors, radius, spacing, typography } from "@/theme/colors";

/**
 * The "choose your mode" control shared by Login, SignUp, and Profile.
 * Picking a chip is the only interaction — the caller decides what happens
 * with the selected mode (sign up with it, log in and sync it, or save it
 * to the profile immediately).
 */
export default function ModePicker({
  value,
  onChange,
  compact = false,
}: {
  value: AppMode;
  onChange: (mode: AppMode) => void;
  compact?: boolean;
}) {
  return (
    <View style={styles.wrap}>
      {MODES.map((mode) => {
        const active = mode === value;
        return (
          <Pressable
            key={mode}
            onPress={() => onChange(mode)}
            style={[styles.chip, active && styles.chipActive]}
            accessibilityRole="button"
            accessibilityLabel={`${MODE_LABELS[mode]} mode`}
            accessibilityState={{ selected: active }}
          >
            <Ionicons name={MODE_ICONS[mode]} size={18} color={active ? colors.accentStrong : colors.textMuted} />
            <Text style={[styles.chipLabel, active && styles.chipLabelActive]}>{MODE_LABELS[mode]}</Text>
            {!compact && (
              <Text style={[styles.chipDescription, active && styles.chipDescriptionActive]} numberOfLines={2}>
                {MODE_DESCRIPTIONS[mode]}
              </Text>
            )}
          </Pressable>
        );
      })}
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: { gap: spacing.sm },
  chip: {
    flexDirection: "row",
    flexWrap: "wrap",
    alignItems: "center",
    backgroundColor: colors.background,
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: colors.border,
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.md,
    gap: spacing.xs,
  },
  chipActive: { backgroundColor: colors.accentSoft, borderColor: colors.accent },
  chipLabel: { ...typography.body, fontWeight: "700", color: colors.textSecondary },
  chipLabelActive: { color: colors.accentStrong },
  chipDescription: {
    ...typography.label,
    color: colors.textMuted,
    fontWeight: "400",
    flexBasis: "100%",
    lineHeight: 16,
  },
  chipDescriptionActive: { color: colors.accentStrong },
});
