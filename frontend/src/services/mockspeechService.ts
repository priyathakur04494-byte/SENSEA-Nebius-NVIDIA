/**
 * BACKWARD-COMPATIBLE SHIM
 * -------------------------
 * Re-exports from the canonical mock implementation so that any existing
 * imports of `@/services/mockspeechService` keep compiling unchanged.
 *
 * Prefer importing directly from:
 *   @/services/mock/mockSpeechService   (mock)
 *   @/services/real/realSpeechService   (real backend)
 */
export { default } from "@/services/mock/mockSpeechService";
export { simplifyText } from "@/services/mock/mockSpeechService";
