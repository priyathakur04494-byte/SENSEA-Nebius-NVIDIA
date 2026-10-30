import React from "react";
import { StyleSheet, Text, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { SoundDirection } from "@/types";
import { colors, radius, spacing, typography } from "@/theme/colors";

interface Props {
  direction: SoundDirection;
}

export default function DirectionIndicator({ direction }: Props) {
  return (
    <View style={styles.row} accessible accessibilityLabel={`Sound direction: ${direction}`}>
      <Segment active={direction === "left"} icon="chevron-back" label="Left" />
      <Segment active={direction === "center"} icon="ellipse" label="Center" />
      <Segment active={direction === "right"} icon="chevron-forward" label="Right" />
    </View>
  );
}

function Segment({
  active,
  icon,
  label,
}: {
  active: boolean;
  icon: keyof typeof Ionicons.glyphMap;
  label: string;
}) {
  return (
    <View style={[styles.segment, active && styles.segmentActive]}>
      <Ionicons name={icon} size={14} color={active ? colors.accent : colors.textMuted} style={{ marginRight: 4 }} />
      <Text style={[styles.segmentText, active && styles.segmentTextActive]}>{label}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  row: { flexDirection: "row", gap: spacing.xs },
  segment: {
    flex: 1,
    flexDirection: "row",
    paddingVertical: spacing.sm,
    borderRadius: radius.sm,
    backgroundColor: colors.surface,
    alignItems: "center",
    justifyContent: "center",
    borderWidth: 1,
    borderColor: colors.border,
  },
  segmentActive: { backgroundColor: colors.accentSoft, borderColor: colors.accent },
  segmentText: { ...typography.label, color: colors.textMuted },
  segmentTextActive: { color: colors.accentStrong },
});
