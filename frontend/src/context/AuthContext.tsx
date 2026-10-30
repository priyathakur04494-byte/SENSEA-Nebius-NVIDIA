import React, { createContext, useCallback, useContext, useEffect, useState } from "react";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { SupportedLanguage } from "@/types";

// ---------------------------------------------------------------------------
// Local, on-device auth. There's no backend in this project, so accounts are
// created and checked entirely on the device via AsyncStorage — this is
// enough to demo a real login/sign-up flow and a persistent profile, but it
// is NOT secure storage (passwords aren't hashed) and accounts don't sync
// across devices. Swap this file for calls to a real auth backend (Firebase
// Auth, Supabase, your own API, etc.) before shipping to real users — no
// other screen needs to change, they all go through useAuth().
// ---------------------------------------------------------------------------

export interface UserProfile {
  id: string;
  name: string;
  email: string;
  createdAt: number;
  preferredLanguage: SupportedLanguage;
}

interface StoredUser extends UserProfile {
  password: string;
}

type AuthResult = { success: true } | { success: false; error: string };

interface AuthState {
  user: UserProfile | null;
  isLoading: boolean;
  signUp: (name: string, email: string, password: string) => Promise<AuthResult>;
  logIn: (email: string, password: string) => Promise<AuthResult>;
  logOut: () => Promise<void>;
  updateProfile: (updates: Partial<Pick<UserProfile, "name" | "preferredLanguage">>) => Promise<void>;
}

const USERS_KEY = "sensea_users_v1";
const SESSION_KEY = "sensea_session_v1";

const AuthContext = createContext<AuthState | undefined>(undefined);

async function readUsers(): Promise<Record<string, StoredUser>> {
  const raw = await AsyncStorage.getItem(USERS_KEY);
  return raw ? JSON.parse(raw) : {};
}

async function writeUsers(users: Record<string, StoredUser>) {
  await AsyncStorage.setItem(USERS_KEY, JSON.stringify(users));
}

function toProfile(stored: StoredUser): UserProfile {
  const { password: _password, ...profile } = stored;
  return profile;
}

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<UserProfile | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    (async () => {
      try {
        const sessionEmail = await AsyncStorage.getItem(SESSION_KEY);
        if (sessionEmail) {
          const users = await readUsers();
          const found = users[sessionEmail];
          if (found) setUser(toProfile(found));
        }
      } finally {
        setIsLoading(false);
      }
    })();
  }, []);

  const signUp = useCallback(async (name: string, email: string, password: string): Promise<AuthResult> => {
    const trimmedName = name.trim();
    const normalizedEmail = email.trim().toLowerCase();

    if (!trimmedName) return { success: false, error: "Please enter your name." };
    if (!/^\S+@\S+\.\S+$/.test(normalizedEmail)) {
      return { success: false, error: "Please enter a valid email address." };
    }
    if (password.length < 6) {
      return { success: false, error: "Password must be at least 6 characters." };
    }

    const users = await readUsers();
    if (users[normalizedEmail]) {
      return { success: false, error: "An account with this email already exists. Try logging in instead." };
    }

    const newUser: StoredUser = {
      id: `user_${Date.now()}`,
      name: trimmedName,
      email: normalizedEmail,
      password,
      createdAt: Date.now(),
      preferredLanguage: "en",
    };

    users[normalizedEmail] = newUser;
    await writeUsers(users);
    await AsyncStorage.setItem(SESSION_KEY, normalizedEmail);
    setUser(toProfile(newUser));
    return { success: true };
  }, []);

  const logIn = useCallback(async (email: string, password: string): Promise<AuthResult> => {
    const normalizedEmail = email.trim().toLowerCase();
    if (!normalizedEmail || !password) {
      return { success: false, error: "Please enter your email and password." };
    }

    const users = await readUsers();
    const found = users[normalizedEmail];
    if (!found || found.password !== password) {
      return { success: false, error: "Incorrect email or password." };
    }

    await AsyncStorage.setItem(SESSION_KEY, normalizedEmail);
    setUser(toProfile(found));
    return { success: true };
  }, []);

  const logOut = useCallback(async () => {
    await AsyncStorage.removeItem(SESSION_KEY);
    setUser(null);
  }, []);

  const updateProfile = useCallback(
    async (updates: Partial<Pick<UserProfile, "name" | "preferredLanguage">>) => {
      setUser((prev) => (prev ? { ...prev, ...updates } : prev));

      const users = await readUsers();
      const current = user;
      if (!current) return;
      const existing = users[current.email];
      if (existing) {
        users[current.email] = { ...existing, ...updates };
        await writeUsers(users);
      }
    },
    [user]
  );

  return (
    <AuthContext.Provider value={{ user, isLoading, signUp, logIn, logOut, updateProfile }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthState {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
