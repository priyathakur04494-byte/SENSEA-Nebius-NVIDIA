import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  AccessibilityInfo,
  Animated,
  Easing,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { LinearGradient } from "expo-linear-gradient";
import { useNavigation } from "@react-navigation/native";
import type { BottomTabNavigationProp } from "@react-navigation/bottom-tabs";
import type { NativeStackNavigationProp } from "@react-navigation/native-stack";
import { useAuth } from "@/context/AuthContext";
import { useAppState } from "@/context/AppContext";
import { colors, radius, shadow, spacing, typography } from "@/theme/colors";
import type { MainTabParamList, RootStackParamList } from "@/navigation/AppNavigator";

type Nav = BottomTabNavigationProp<MainTabParamList> & NativeStackNavigationProp<RootStackParamList>;

interface Feature {
  icon: keyof typeof Ionicons.glyphMap;
  title: string;
  description: string;
  tab?: keyof MainTabParamList;
}

const FEATURES: Feature[] = [
  {
    icon: "chatbubble-ellipses-outline",
    title: "Live Captions",
    description: "Real-time speech-to-text with speaker labels, translation, and plain-language mode.",
    tab: "Captions",
  },
  {
    icon: "sparkles-outline",
    title: "AI Context",
    description: "A live summary, important points, decisions, and a running answer to \"what did I miss?\"",
    tab: "Summary",
  },
  {
    icon: "notifications-outline",
    title: "Sound Alerts",
    description: "Sirens, alarms, horns, and doorbells — flagged with urgency and direction.",
    tab: "Alerts",
  },
  {
    icon: "volume-high-outline",
    title: "Type to Speak",
    description: "Type a message and SENSEA translates and speaks it aloud for you.",
    tab: "Speak",
  },
  {
    icon: "bulb-outline",
    title: "Personal Memory",
    description: "Remembers speakers, important people, mentioned deadlines, and your preferences.",
    tab: "Memory",
  },
];

// ---------------------------------------------------------------------------
// Small helper: a scale-on-press wrapper so every tappable surface (avatar,
// CTA, feature cards) gets the same tactile "premium app" feedback without
// repeating Animated boilerplate five times.
// ---------------------------------------------------------------------------
function Pressy({
  children,
  onPress,
  style,
  scaleTo = 0.97,
  accessibilityLabel,
  accessibilityRole = "button",
}: {
  children: React.ReactNode;
  onPress?: () => void;
  style?: any;
  scaleTo?: number;
  accessibilityLabel?: string;
  accessibilityRole?: "button";
}) {
  const scale = useRef(new Animated.Value(1)).current;

  const animateTo = (value: number) => {
    Animated.spring(scale, {
      toValue: value,
      useNativeDriver: true,
      speed: 40,
      bounciness: 8,
    }).start();
  };

  return (
    <Pressable
      onPress={onPress}
      onPressIn={() => animateTo(scaleTo)}
      onPressOut={() => animateTo(1)}
      accessibilityRole={accessibilityRole}
      accessibilityLabel={accessibilityLabel}
    >
      <Animated.View style={[style, { transform: [{ scale }] }]}>{children}</Animated.View>
    </Pressable>
  );
}

// ---------------------------------------------------------------------------
// Pulsing "live" ring — a soft expanding halo behind the status dot while
// SENSEA is actively listening. Skipped entirely when the OS-level "reduce
// motion" accessibility setting is on.
// ---------------------------------------------------------------------------
function PulseDot({ active, reduceMotion }: { active: boolean; reduceMotion: boolean }) {
  const pulse = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    pulse.stopAnimation();
    pulse.setValue(0);
    if (!active || reduceMotion) return;
    const loop = Animated.loop(
      Animated.timing(pulse, {
        toValue: 1,
        duration: 1400,
        easing: Easing.out(Easing.ease),
        useNativeDriver: true,
      })
    );
    loop.start();
    return () => loop.stop();
  }, [active, reduceMotion, pulse]);

  const ringScale = pulse.interpolate({ inputRange: [0, 1], outputRange: [1, 2.4] });
  const ringOpacity = pulse.interpolate({ inputRange: [0, 1], outputRange: [0.5, 0] });

  return (
    <View style={styles.pulseWrap}>
      {active && (
        <Animated.View
          style={[
            styles.pulseRing,
            { opacity: ringOpacity, transform: [{ scale: ringScale }] },
          ]}
        />
      )}
      <View style={[styles.statusDot, active && styles.statusDotActive]} />
    </View>
  );
}

// ---------------------------------------------------------------------------
// A feature row with its own entrance animation (fade + rise), staggered by
// `index`, plus the shared press-scale feedback.
// ---------------------------------------------------------------------------
function FeatureCard({
  feature,
  index,
  onPress,
}: {
  feature: Feature;
  index: number;
  onPress: () => void;
}) {
  const progress = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.timing(progress, {
      toValue: 1,
      duration: 420,
      delay: 90 * index,
      easing: Easing.out(Easing.cubic),
      useNativeDriver: true,
    }).start();
  }, [progress, index]);

  const opacity = progress;
  const translateY = progress.interpolate({ inputRange: [0, 1], outputRange: [16, 0] });

  return (
    <Animated.View style={{ opacity, transform: [{ translateY }] }}>
      <Pressy
        onPress={onPress}
        accessibilityLabel={feature.title}
        style={[styles.card, shadow.md]}
        scaleTo={0.98}
      >
        <View style={styles.cardIconWrap}>
          <Ionicons name={feature.icon} size={20} color={colors.accentStrong} />
        </View>
        <View style={{ flex: 1 }}>
          <Text style={styles.cardTitle}>{feature.title}</Text>
          <Text style={styles.cardDescription}>{feature.description}</Text>
        </View>
        <View style={styles.cardChevronWrap}>
          <Ionicons name="chevron-forward" size={16} color={colors.textMuted} />
        </View>
      </Pressy>
    </Animated.View>
  );
}

export default function HomeScreen() {
  const navigation = useNavigation<Nav>();
  const { user } = useAuth();
  const { isListening, toggleListening, soundEvents, captions } = useAppState();
  const firstName = user?.name?.split(" ")[0] || "there";
  const [reduceMotion, setReduceMotion] = useState(false);

  useEffect(() => {
    AccessibilityInfo.isReduceMotionEnabled?.().then(setReduceMotion).catch(() => {});
    const sub = AccessibilityInfo.addEventListener?.("reduceMotionChanged", setReduceMotion);
    return () => sub?.remove?.();
  }, []);

  // Hero entrance: a single gentle fade/rise so the whole top section
  // arrives together rather than popping in.
  const heroProgress = useRef(new Animated.Value(0)).current;
  useEffect(() => {
    Animated.timing(heroProgress, {
      toValue: 1,
      duration: 460,
      easing: Easing.out(Easing.cubic),
      useNativeDriver: true,
    }).start();
  }, [heroProgress]);
  const heroOpacity = heroProgress;
  const heroTranslateY = heroProgress.interpolate({ inputRange: [0, 1], outputRange: [12, 0] });

  const stats = useMemo(
    () => [
      { icon: "notifications-outline" as const, label: "events today", value: String(soundEvents.length) },
      { icon: "chatbubble-outline" as const, label: "captions logged", value: String(captions.length) },
    ],
    [soundEvents.length, captions.length]
  );

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <Animated.View style={{ opacity: heroOpacity, transform: [{ translateY: heroTranslateY }] }}>
        <LinearGradient
          colors={[colors.primary, colors.accentStrong, colors.accent]}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 1 }}
          style={styles.hero}
        >
          <View style={styles.headerRow}>
            <View style={{ flex: 1 }}>
              <Text style={styles.greeting}>Hi, {firstName}</Text>
              <Text style={styles.brand}>SENSEA</Text>
            </View>
            <Pressy
              onPress={() => navigation.navigate("Profile")}
              accessibilityLabel="Open profile"
              style={styles.avatar}
            >
              <Text style={styles.avatarText}>{(user?.name?.[0] || "?").toUpperCase()}</Text>
            </Pressy>
          </View>

          <Text style={styles.tagline}>
            Sense every conversation — what's happening, what matters, and what you missed.
          </Text>

          <View style={styles.heroDivider} />

          <View style={styles.statusRow}>
            <View style={styles.statusLeft}>
              <PulseDot active={isListening} reduceMotion={reduceMotion} />
              <Text style={styles.statusLabel}>{isListening ? "Listening" : "Idle"}</Text>
            </View>

            <Pressy onPress={toggleListening} accessibilityLabel={isListening ? "Stop listening" : "Start listening"}>
              <View style={[styles.ctaButton, isListening && styles.ctaButtonActive]}>
                <Ionicons
                  name={isListening ? "stop-circle-outline" : "mic-outline"}
                  size={17}
                  color={isListening ? colors.textOnAccent : colors.accentStrong}
                />
                <Text style={[styles.ctaText, isListening && styles.ctaTextActive]}>
                  {isListening ? "Stop" : "Start listening"}
                </Text>
              </View>
            </Pressy>
          </View>

          <View style={styles.statsRow}>
            {stats.map((stat) => (
              <View key={stat.label} style={styles.statChip}>
                <Ionicons name={stat.icon} size={14} color={colors.textOnDark} />
                <Text style={styles.statValue}>{stat.value}</Text>
                <Text style={styles.statLabel}>{stat.label}</Text>
              </View>
            ))}
          </View>
        </LinearGradient>
      </Animated.View>

      <Text style={styles.sectionLabel}>WHAT SENSEA DOES</Text>

      {FEATURES.map((feature, index) => (
        <FeatureCard
          key={feature.title}
          feature={feature}
          index={index}
          onPress={() => feature.tab && navigation.navigate(feature.tab)}
        />
      ))}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.background },
  content: { paddingBottom: spacing.xl },

  hero: {
    paddingTop: spacing.lg,
    paddingHorizontal: spacing.md,
    paddingBottom: spacing.lg,
    borderBottomLeftRadius: 28,
    borderBottomRightRadius: 28,
    marginBottom: spacing.lg,
    ...shadow.lg,
  },
  headerRow: { flexDirection: "row", alignItems: "center", marginBottom: spacing.sm },
  greeting: { ...typography.body, color: "rgba(255,255,255,0.75)" },
  brand: { ...typography.heading, color: colors.textOnDark, letterSpacing: 2 },
  avatar: {
    width: 46,
    height: 46,
    borderRadius: radius.pill,
    backgroundColor: "rgba(255,255,255,0.14)",
    borderWidth: 1.5,
    borderColor: "rgba(255,255,255,0.4)",
    alignItems: "center",
    justifyContent: "center",
    marginLeft: spacing.sm,
  },
  avatarText: { ...typography.title, color: colors.textOnDark },
  tagline: {
    ...typography.body,
    color: "rgba(255,255,255,0.88)",
    lineHeight: 21,
    marginBottom: spacing.md,
  },
  heroDivider: {
    height: 1,
    backgroundColor: "rgba(255,255,255,0.18)",
    marginBottom: spacing.md,
  },

  statusRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    marginBottom: spacing.md,
  },
  statusLeft: { flexDirection: "row", alignItems: "center" },
  pulseWrap: {
    width: 20,
    height: 20,
    alignItems: "center",
    justifyContent: "center",
    marginRight: spacing.xs,
  },
  pulseRing: {
    position: "absolute",
    width: 10,
    height: 10,
    borderRadius: radius.pill,
    backgroundColor: colors.normal,
  },
  statusDot: {
    width: 10,
    height: 10,
    borderRadius: radius.pill,
    backgroundColor: "rgba(255,255,255,0.4)",
  },
  statusDotActive: { backgroundColor: colors.normal },
  statusLabel: { ...typography.label, color: colors.textOnDark },

  ctaButton: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: colors.textOnDark,
    borderRadius: radius.pill,
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.md,
    gap: 6,
  },
  ctaButtonActive: { backgroundColor: colors.critical },
  ctaText: { ...typography.label, color: colors.accentStrong },
  ctaTextActive: { color: colors.textOnAccent },

  statsRow: { flexDirection: "row", gap: spacing.sm },
  statChip: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "rgba(255,255,255,0.12)",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.2)",
    borderRadius: radius.md,
    paddingVertical: spacing.xs,
    paddingHorizontal: spacing.sm,
    gap: 6,
  },
  statValue: { ...typography.label, color: colors.textOnDark, fontSize: 14 },
  statLabel: { ...typography.body, color: "rgba(255,255,255,0.8)", fontSize: 12 },

  sectionLabel: {
    ...typography.label,
    color: colors.textMuted,
    marginBottom: spacing.sm,
    marginHorizontal: spacing.md,
    letterSpacing: 1,
  },
  card: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: colors.surface,
    borderRadius: radius.lg,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.md,
    marginBottom: spacing.sm,
    marginHorizontal: spacing.md,
  },
  cardIconWrap: {
    width: 42,
    height: 42,
    borderRadius: radius.md,
    backgroundColor: colors.accentSoft,
    alignItems: "center",
    justifyContent: "center",
    marginRight: spacing.md,
  },
  cardTitle: { ...typography.title, fontSize: 16, color: colors.textPrimary },
  cardDescription: { ...typography.body, fontSize: 13, color: colors.textSecondary, marginTop: 2, lineHeight: 18 },
  cardChevronWrap: {
    marginLeft: spacing.sm,
    width: 26,
    height: 26,
    borderRadius: radius.pill,
    backgroundColor: colors.surfaceMuted,
    alignItems: "center",
    justifyContent: "center",
  },
});
