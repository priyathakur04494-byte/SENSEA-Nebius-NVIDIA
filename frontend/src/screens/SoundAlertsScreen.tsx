import React from "react";
import { FlatList, StyleSheet, Text, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { useAppState } from "@/context/AppContext";
import AlertBanner from "@/components/AlertBanner";
import { colors, radius, spacing, typography } from "@/theme/colors";

export default function SoundAlertsScreen() {
  const { soundEvents } = useAppState();

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Sound alerts</Text>
      <Text style={styles.subtitle}>
        Your device listens for important sounds around you — sirens, alarms, horns, doorbells — and
        alerts you here with an urgency level and the direction it came from.
      </Text>

      <View style={styles.legend}>
        <LegendItem color={colors.alertNormal} icon="checkmark-circle" label="Normal — everyday sounds (doorbell, knock)" />
        <LegendItem color={colors.alertImportant} icon="warning" label="Important — pay attention (alarm, horn)" />
        <LegendItem color={colors.alertEmergency} icon="alert-circle" label="Emergency — act now (siren)" />
      </View>

      {soundEvents.length === 0 ? (
        <Text style={styles.emptyNote}>No sounds detected yet.</Text>
      ) : (
        <FlatList
          data={soundEvents}
          keyExtractor={(item) => item.id}
          contentContainerStyle={{ gap: spacing.sm, paddingBottom: spacing.xl }}
          renderItem={({ item }) => (
            <AlertBanner
              label={item.label}
              level={item.alertLevel}
              direction={item.direction}
              timestamp={item.timestamp}
            />
          )}
        />
      )}
    </View>
  );
}

function LegendItem({ color, icon, label }: { color: string; icon: keyof typeof Ionicons.glyphMap; label: string }) {
  return (
    <View style={styles.legendRow}>
      <View style={[styles.legendIconWrap, { backgroundColor: `${color}26` }]}>
        <Ionicons name={icon} size={14} color={color} />
      </View>
      <Text style={styles.legendText}>{label}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.background, padding: spacing.md },
  title: { ...typography.heading, color: colors.textPrimary },
  subtitle: { ...typography.body, color: colors.textSecondary, marginTop: spacing.xs, marginBottom: spacing.md },
  legend: { marginBottom: spacing.lg, gap: spacing.xs },
  legendRow: { flexDirection: "row", alignItems: "center", gap: spacing.sm },
  legendIconWrap: {
    width: 22,
    height: 22,
    borderRadius: radius.pill,
    alignItems: "center",
    justifyContent: "center",
  },
  legendText: { ...typography.body, color: colors.textSecondary },
  emptyNote: { ...typography.body, color: colors.textMuted, fontStyle: "italic" },
});
