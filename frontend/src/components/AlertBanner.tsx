import React, { useEffect, useRef } from "react";
import { Animated, StyleSheet, Text, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { AlertLevel, SoundDirection } from "@/types";
import { colors, radius, shadow, spacing, typography } from "@/theme/colors";

interface Props {
  label: string;
  level: AlertLevel;
  direction?: SoundDirection;
  timestamp?: number;
}

// Shape + icon differ per level, not just color/background — some users may
// have color-vision differences, and this app must never rely on hue alone
// to convey urgency.
const LEVEL_META: Record<
  AlertLevel,
  { color: string; soft: string; icon: keyof typeof Ionicons.glyphMap; text: string }
> = {
  normal: { color: colors.normal, soft: colors.normalSoft, icon: "checkmark-circle", text: "Normal" },
  important: { color: colors.warning, soft: colors.warningSoft, icon: "warning", text: "Important" },
  emergency: { color: colors.critical, soft: colors.criticalSoft, icon: "alert-circle", text: "Emergency" },
};

const DIRECTION_LABEL: Record<SoundDirection, string> = {
  left: "← coming from your left",
  center: "in front of you",
  right: "coming from your right →",
};

export default function AlertBanner({ label, level, direction, timestamp }: Props) {
  const meta = LEVEL_META[level];
  const pulse = useRef(new Animated.Value(1)).current;

  useEffect(() => {
    if (level === "emergency") {
      const loop = Animated.loop(
        Animated.sequence([
          Animated.timing(pulse, { toValue: 1.015, duration: 550, useNativeDriver: true }),
          Animated.timing(pulse, { toValue: 1, duration: 550, useNativeDriver: true }),
        ])
      );
      loop.start();
      return () => loop.stop();
    }
  }, [level]);

  return (
    <Animated.View
      style={[
        styles.container,
        shadow.sm,
        { backgroundColor: meta.soft, borderColor: meta.color, transform: [{ scale: pulse }] },
      ]}
      accessible
      accessibilityRole="alert"
      accessibilityLabel={`${meta.text} alert: ${label}`}
    >
      <View style={[styles.iconWrap, { backgroundColor: meta.color }]}>
        <Ionicons name={meta.icon} size={20} color="#fff" />
      </View>
      <View style={styles.textWrap}>
        <View style={styles.levelRow}>
          <Text style={[styles.levelLabel, { color: meta.color }]}>{meta.text.toUpperCase()}</Text>
          {timestamp !== undefined && (
            <Text style={styles.timestamp}>{new Date(timestamp).toLocaleTimeString()}</Text>
          )}
        </View>
        <Text style={styles.label}>{label}</Text>
        {direction && <Text style={styles.direction}>{DIRECTION_LABEL[direction]}</Text>}
      </View>
    </Animated.View>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: "row",
    alignItems: "flex-start",
    borderWidth: 1.5,
    borderRadius: radius.lg,
    padding: spacing.md,
    gap: spacing.md,
  },
  iconWrap: {
    width: 40,
    height: 40,
    borderRadius: radius.pill,
    alignItems: "center",
    justifyContent: "center",
  },
  textWrap: { flex: 1 },
  levelRow: { flexDirection: "row", alignItems: "center", justifyContent: "space-between" },
  levelLabel: { ...typography.label, letterSpacing: 1 },
  timestamp: { ...typography.label, color: colors.textMuted, fontWeight: "500" },
  label: { ...typography.title, fontSize: 17, color: colors.textPrimary, marginTop: 2 },
  direction: { ...typography.body, color: colors.textSecondary, marginTop: 2 },
});
