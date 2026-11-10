/**
 * BACKWARD-COMPATIBLE SHIM
 * -------------------------
 * Re-exports from the canonical mock implementation so that any existing
 * imports of `@/services/mocksoundDetectionService` keep compiling unchanged.
 *
 * Prefer importing directly from:
 *   @/services/mock/mockSoundDetectionService   (mock)
 *   @/services/real/realSoundDetectionService   (real native)
 */
export { default } from "@/services/mock/mockSoundDetectionService";
export { alertLevelWeight } from "@/services/mock/mockSoundDetectionService";

// Legacy named-export aliases used by older screens
import mockSoundDetectionService from "@/services/mock/mockSoundDetectionService";
export function startMockDetection(onEvent: Parameters<typeof mockSoundDetectionService.startDetection>[0]) {
  mockSoundDetectionService.startDetection(onEvent);
}
export function stopMockDetection() {
  mockSoundDetectionService.stopDetection();
}
