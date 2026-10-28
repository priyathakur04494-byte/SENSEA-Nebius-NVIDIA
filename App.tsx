import React from "react";
import { StatusBar } from "expo-status-bar";
import { SafeAreaProvider } from "react-native-safe-area-context";
import { AppProvider } from "@/context/AppContext";
import { AuthProvider } from "@/context/AuthContext";
import AppNavigator from "@/navigation/AppNavigator";

export default function App() {
  return (
    <SafeAreaProvider>
      <AuthProvider>
        <AppProvider>
          <StatusBar style="dark" />
          <AppNavigator />
        </AppProvider>
      </AuthProvider>
    </SafeAreaProvider>
  );
}
