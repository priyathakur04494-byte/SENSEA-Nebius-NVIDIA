/**
 * Signs the user has taught the app ("Teach a sign"), saved on the device.
 * Works for ANY sign language — the app stores how the user's own hand moved.
 */
import AsyncStorage from "@react-native-async-storage/async-storage";

export interface CustomSign {
  id: string;
  label: string;
  /** Hand-movement fingerprints recorded by the camera page (opaque numbers). */
  templates: number[][];
  threshold: number;
}

const KEY = "signs:custom";

export async function loadCustomSigns(): Promise<CustomSign[]> {
  try {
    const raw = await AsyncStorage.getItem(KEY);
    return raw ? (JSON.parse(raw) as CustomSign[]) : [];
  } catch {
    return [];
  }
}

export async function saveCustomSigns(list: CustomSign[]): Promise<void> {
  try {
    await AsyncStorage.setItem(KEY, JSON.stringify(list));
  } catch {
    /* storage full / unavailable — the signs still work this session */
  }
}
