import React from "react";
import { ActivityIndicator, Pressable, StyleSheet, Text, View } from "react-native";
import { NavigationContainer, DefaultTheme } from "@react-navigation/native";
import { createBottomTabNavigator } from "@react-navigation/bottom-tabs";
import { createNativeStackNavigator } from "@react-navigation/native-stack";
import { Ionicons } from "@expo/vector-icons";
import HomeScreen from "@/screens/HomeScreen";
import LiveCaptionScreen from "@/screens/LiveCaptionScreen";
import SoundAlertsScreen from "@/screens/SoundAlertsScreen";
import TypeToSpeakScreen from "@/screens/TypeToSpeakScreen";
import SummaryScreen from "@/screens/SummaryScreen";
import MemoryScreen from "@/screens/MemoryScreen";
import LoginScreen from "@/screens/LoginScreen";
import SignUpScreen from "@/screens/SignUpScreen";
import ProfileScreen from "@/screens/ProfileScreen";
import { useAuth } from "@/context/AuthContext";
import { colors, radius, shadow, spacing } from "@/theme/colors";

// ---------------------------------------------------------------------------
// Navigation shape:
//   Not logged in -> RootStack shows the Auth screens (Login, SignUp).
//   Logged in      -> RootStack shows MainTabs, with Profile pushed on top
//                      from the Home header.
//
// Tab order follows the product's information hierarchy: Live captions is
// the primary, always-first destination; AI Context (summary, important
// points, decisions, "what did I miss") is the secondary destination right
// next to it; Alerts, Speak, and Memory follow. Route names are kept as
// they were (e.g. "Summary") so existing navigation calls / types elsewhere
// don't need to change — only the visible tab label and icon changed.
// ---------------------------------------------------------------------------

export type AuthStackParamList = {
  Login: undefined;
  SignUp: undefined;
};

export type MainTabParamList = {
  Home: undefined;
  Captions: undefined;
  Summary: undefined;
  Alerts: undefined;
  Speak: undefined;
  Memory: undefined;
};

export type RootStackParamList = {
  Login: undefined;
  SignUp: undefined;
  MainTabs: undefined;
  Profile: undefined;
};

const Tab = createBottomTabNavigator<MainTabParamList>();
const RootStack = createNativeStackNavigator<RootStackParamList>();

const navTheme = {
  ...DefaultTheme,
  colors: {
    ...DefaultTheme.colors,
    background: colors.background,
    card: colors.surface,
    border: colors.border,
    primary: colors.accent,
    text: colors.textPrimary,
  },
};

const ICONS: Record<string, { active: keyof typeof Ionicons.glyphMap; inactive: keyof typeof Ionicons.glyphMap }> = {
  Captions: { active: "chatbubble-ellipses", inactive: "chatbubble-ellipses-outline" },
  Summary: { active: "sparkles", inactive: "sparkles-outline" },
  Alerts: { active: "notifications", inactive: "notifications-outline" },
  Speak: { active: "volume-high", inactive: "volume-high-outline" },
  Memory: { active: "bulb", inactive: "bulb-outline" },
  Home: { active: "home", inactive: "home-outline" },
};

// A single restrained accent for every active tab — differentiation comes
// from the icon and label, not a different hue per tab.
const TAB_LABELS: Record<string, string> = {
  Home: "Home",
  Captions: "Live",
  Summary: "Context",
  Alerts: "Alerts",
  Speak: "Speak",
  Memory: "Memory",
};

const SUBTITLES: Record<string, string> = {
  Home: "Home",
  Captions: "Live Captions",
  Summary: "AI Context",
  Alerts: "Sound Alerts",
  Speak: "Type to Speak",
  Memory: "Memory",
};

function BrandHeaderTitle({ subtitle }: { subtitle: string }) {
  return (
    <View style={styles.headerTitleWrap}>
      <Text style={styles.headerBrand}>SENSEA</Text>
      <Text style={styles.headerSubtitle}>{subtitle}</Text>
    </View>
  );
}

function MainTabs() {
  return (
    <Tab.Navigator
      screenOptions={({ route, navigation }) => ({
        headerStyle: { backgroundColor: colors.surface, elevation: 0, shadowOpacity: 0 },
        headerTitle: () => <BrandHeaderTitle subtitle={SUBTITLES[route.name] ?? route.name} />,
        headerRight: () => (
          <Pressable
            onPress={() => navigation.getParent()?.navigate("Profile")}
            style={styles.headerProfileButton}
            accessibilityRole="button"
            accessibilityLabel="Open profile"
          >
            <Ionicons name="person-outline" size={18} color={colors.accentStrong} />
          </Pressable>
        ),
        tabBarShowLabel: true,
        tabBarLabel: TAB_LABELS[route.name] ?? route.name,
        tabBarStyle: styles.tabBar,
        tabBarItemStyle: styles.tabItem,
        tabBarLabelStyle: styles.tabLabel,
        tabBarActiveTintColor: colors.accentStrong,
        tabBarInactiveTintColor: colors.textMuted,
        tabBarIcon: ({ focused }) => (
          <View style={[styles.iconWrap, focused && styles.iconWrapActive]}>
            <Ionicons
              name={focused ? ICONS[route.name]?.active : ICONS[route.name]?.inactive}
              size={19}
              color={focused ? colors.accentStrong : colors.textMuted}
            />
          </View>
        ),
      })}
    >
      <Tab.Screen name="Captions" component={LiveCaptionScreen} />
      <Tab.Screen name="Summary" component={SummaryScreen} />
      <Tab.Screen name="Alerts" component={SoundAlertsScreen} />
      <Tab.Screen name="Speak" component={TypeToSpeakScreen} />
      <Tab.Screen name="Memory" component={MemoryScreen} />
      <Tab.Screen name="Home" component={HomeScreen} />
    </Tab.Navigator>
  );
}

function SplashLoading() {
  return (
    <View style={styles.splash}>
      <ActivityIndicator color={colors.accent} size="large" />
    </View>
  );
}

export default function AppNavigator() {
  const { user, isLoading } = useAuth();

  if (isLoading) {
    return <SplashLoading />;
  }

  return (
    <NavigationContainer theme={navTheme}>
      <RootStack.Navigator screenOptions={{ headerShown: false }}>
        {user ? (
          <>
            <RootStack.Screen name="MainTabs" component={MainTabs} />
            <RootStack.Screen
              name="Profile"
              component={ProfileScreen}
              options={{
                headerShown: true,
                headerTitle: () => <BrandHeaderTitle subtitle="Profile" />,
                headerStyle: { backgroundColor: colors.surface },
                headerTintColor: colors.accentStrong,
              }}
            />
          </>
        ) : (
          <>
            <RootStack.Screen name="Login" component={LoginScreen} />
            <RootStack.Screen name="SignUp" component={SignUpScreen} />
          </>
        )}
      </RootStack.Navigator>
    </NavigationContainer>
  );
}

const styles = StyleSheet.create({
  splash: { flex: 1, backgroundColor: colors.background, alignItems: "center", justifyContent: "center" },
  headerTitleWrap: { alignItems: "flex-start" },
  headerBrand: {
    fontSize: 17,
    fontWeight: "800",
    letterSpacing: 1.2,
    color: colors.primary,
  },
  headerSubtitle: {
    fontSize: 12,
    fontWeight: "600",
    color: colors.textMuted,
    marginTop: 1,
  },
  tabBar: {
    backgroundColor: colors.surface,
    borderTopColor: colors.border,
    borderTopWidth: 1,
    height: 68,
    paddingTop: 6,
    paddingBottom: 10,
    ...shadow.sm,
  },
  tabItem: {
    paddingTop: 2,
  },
  tabLabel: {
    fontSize: 11,
    fontWeight: "700",
    marginTop: 2,
  },
  iconWrap: {
    width: 36,
    height: 30,
    borderRadius: radius.pill,
    alignItems: "center",
    justifyContent: "center",
  },
  iconWrapActive: { backgroundColor: colors.accentSoft },
  headerProfileButton: {
    marginRight: spacing.md,
    width: 34,
    height: 34,
    borderRadius: radius.pill,
    backgroundColor: colors.accentSoft,
    alignItems: "center",
    justifyContent: "center",
  },
});
