import React, { useState } from "react";
import {
  ActivityIndicator,
  KeyboardAvoidingView,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import type { NativeStackScreenProps } from "@react-navigation/native-stack";
import { useAuth } from "@/context/AuthContext";
import { colors, radius, shadow, spacing, typography } from "@/theme/colors";
import type { AuthStackParamList } from "@/navigation/AppNavigator";

type Props = NativeStackScreenProps<AuthStackParamList, "SignUp">;

export default function SignUpScreen({ navigation }: Props) {
  const { signUp } = useAuth();
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [isPasswordVisible, setIsPasswordVisible] = useState(false);
  const [isConfirmVisible, setIsConfirmVisible] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSignUp = async () => {
    setError(null);
    if (password !== confirmPassword) {
      setError("Passwords don't match.");
      return;
    }
    setIsSubmitting(true);
    const result = await signUp(name, email, password);
    setIsSubmitting(false);
    if (!result.success) setError(result.error);
    // On success, the root navigator swaps to the main app automatically.
  };

  return (
    <KeyboardAvoidingView
      style={styles.flex}
      behavior={Platform.OS === "ios" ? "padding" : undefined}
    >
      <ScrollView contentContainerStyle={styles.container} keyboardShouldPersistTaps="handled">
        <View style={styles.brandWrap}>
          <View style={[styles.logoCircle, shadow.glow]}>
            <Text style={styles.logoText}>S</Text>
          </View>
          <Text style={styles.brand}>SENSEA</Text>
          <Text style={styles.tagline}>Sense every conversation.</Text>
        </View>

        <View style={[styles.card, shadow.lg]}>
          <Text style={styles.title}>Create your account</Text>
          <Text style={styles.subtitle}>
            Your captions, memory, and preferences will be saved to this account on your device.
          </Text>

          <Text style={styles.label}>Name</Text>
          <View style={styles.inputRow}>
            <Ionicons name="person-outline" size={18} color={colors.textMuted} style={styles.inputIcon} />
            <TextInput
              style={styles.input}
              placeholder="Your name"
              placeholderTextColor={colors.textMuted}
              value={name}
              onChangeText={setName}
              autoCapitalize="words"
              textContentType="name"
              accessibilityLabel="Name"
            />
          </View>

          <Text style={styles.label}>Email</Text>
          <View style={styles.inputRow}>
            <Ionicons name="mail-outline" size={18} color={colors.textMuted} style={styles.inputIcon} />
            <TextInput
              style={styles.input}
              placeholder="you@example.com"
              placeholderTextColor={colors.textMuted}
              value={email}
              onChangeText={setEmail}
              autoCapitalize="none"
              autoCorrect={false}
              keyboardType="email-address"
              textContentType="emailAddress"
              accessibilityLabel="Email"
            />
          </View>

          <Text style={styles.label}>Password</Text>
          <View style={styles.inputRow}>
            <Ionicons name="lock-closed-outline" size={18} color={colors.textMuted} style={styles.inputIcon} />
            <TextInput
              style={[styles.input, styles.inputWithTrailingIcon]}
              placeholder="At least 6 characters"
              placeholderTextColor={colors.textMuted}
              value={password}
              onChangeText={setPassword}
              secureTextEntry={!isPasswordVisible}
              textContentType="newPassword"
              accessibilityLabel="Password"
            />
            <Pressable
              onPress={() => setIsPasswordVisible((prev) => !prev)}
              style={styles.eyeButton}
              hitSlop={10}
              accessibilityRole="button"
              accessibilityLabel={isPasswordVisible ? "Hide password" : "Show password"}
            >
              <Ionicons
                name={isPasswordVisible ? "eye-outline" : "eye-off-outline"}
                size={20}
                color={colors.textMuted}
              />
            </Pressable>
          </View>

          <Text style={styles.label}>Confirm password</Text>
          <View style={styles.inputRow}>
            <Ionicons name="lock-closed-outline" size={18} color={colors.textMuted} style={styles.inputIcon} />
            <TextInput
              style={[styles.input, styles.inputWithTrailingIcon]}
              placeholder="Re-enter your password"
              placeholderTextColor={colors.textMuted}
              value={confirmPassword}
              onChangeText={setConfirmPassword}
              secureTextEntry={!isConfirmVisible}
              textContentType="newPassword"
              accessibilityLabel="Confirm password"
            />
            <Pressable
              onPress={() => setIsConfirmVisible((prev) => !prev)}
              style={styles.eyeButton}
              hitSlop={10}
              accessibilityRole="button"
              accessibilityLabel={isConfirmVisible ? "Hide password" : "Show password"}
            >
              <Ionicons
                name={isConfirmVisible ? "eye-outline" : "eye-off-outline"}
                size={20}
                color={colors.textMuted}
              />
            </Pressable>
          </View>

          {error && (
            <View style={styles.errorBox}>
              <Ionicons name="alert-circle-outline" size={16} color={colors.alertEmergency} />
              <Text style={styles.errorText}>{error}</Text>
            </View>
          )}

          <Pressable
            style={({ pressed }) => [
              styles.primaryButton,
              shadow.glow,
              isSubmitting && styles.buttonDisabled,
              pressed && styles.buttonPressed,
            ]}
            onPress={handleSignUp}
            disabled={isSubmitting}
            accessibilityRole="button"
            accessibilityLabel="Create account"
          >
            {isSubmitting ? (
              <ActivityIndicator color="#fff" />
            ) : (
              <>
                <Text style={styles.primaryButtonText}>Create account</Text>
                <Ionicons name="arrow-forward" size={18} color="#fff" style={{ marginLeft: 6 }} />
              </>
            )}
          </Pressable>
        </View>

        <View style={styles.footerRow}>
          <Text style={styles.footerText}>Already have an account?</Text>
          <Pressable onPress={() => navigation.navigate("Login")} accessibilityRole="button">
            <Text style={styles.footerLink}> Log in</Text>
          </Pressable>
        </View>
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  flex: { flex: 1, backgroundColor: colors.background },
  container: { flexGrow: 1, backgroundColor: colors.background, padding: spacing.lg, justifyContent: "center" },
  brandWrap: { alignItems: "center", marginBottom: spacing.lg },
  logoCircle: {
    width: 72,
    height: 72,
    borderRadius: radius.pill,
    backgroundColor: colors.accentSoft,
    borderWidth: 1,
    borderColor: `${colors.accent}55`,
    alignItems: "center",
    justifyContent: "center",
    marginBottom: spacing.sm,
  },
  logoText: { ...typography.heading, color: colors.accent, fontSize: 32 },
  brand: { ...typography.heading, color: colors.textPrimary, letterSpacing: 3 },
  tagline: { ...typography.body, color: colors.textSecondary, marginTop: 2 },

  card: {
    backgroundColor: colors.surface,
    borderRadius: radius.lg,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.lg,
  },

  title: { ...typography.title, color: colors.textPrimary, marginBottom: spacing.xs },
  subtitle: { ...typography.body, color: colors.textSecondary, marginBottom: spacing.sm },
  label: { ...typography.label, color: colors.textMuted, marginBottom: spacing.xs, marginTop: spacing.md },

  inputRow: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: colors.surfaceElevated,
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: colors.border,
    paddingHorizontal: spacing.md,
  },
  inputIcon: { marginRight: spacing.sm },
  input: {
    flex: 1,
    paddingVertical: spacing.md,
    ...typography.body,
    color: colors.textPrimary,
  },
  inputWithTrailingIcon: { paddingRight: spacing.sm },
  eyeButton: { padding: spacing.xs, marginLeft: spacing.xs },

  errorBox: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: `${colors.alertEmergency}1A`,
    borderRadius: radius.sm,
    borderWidth: 1,
    borderColor: `${colors.alertEmergency}55`,
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.sm,
    marginTop: spacing.md,
  },
  errorText: {
    ...typography.label,
    color: colors.alertEmergency,
    marginLeft: spacing.xs,
    lineHeight: 18,
    flexShrink: 1,
  },

  primaryButton: {
    flexDirection: "row",
    backgroundColor: colors.accent,
    borderRadius: radius.lg,
    paddingVertical: spacing.md,
    alignItems: "center",
    justifyContent: "center",
    marginTop: spacing.lg,
  },
  buttonPressed: { opacity: 0.9 },
  buttonDisabled: { opacity: 0.7 },
  primaryButtonText: { ...typography.title, color: "#fff" },

  footerRow: { flexDirection: "row", justifyContent: "center", marginTop: spacing.lg, flexWrap: "wrap" },
  footerText: { ...typography.body, color: colors.textSecondary },
  footerLink: { ...typography.body, color: colors.accent, fontWeight: "700" },
});
