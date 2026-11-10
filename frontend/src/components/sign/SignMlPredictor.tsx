import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  ActivityIndicator,
  Platform,
  Pressable,
  StyleSheet,
  Text,
  View,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { predictSignFromVideo, SignPredictResult } from "@/services/backendApi";
import { ISL_ML_SIGNS } from "@/config/recognizedSigns";
import { speak } from "@/services/ttsService";
import { colors, radius, shadow, spacing, typography } from "@/theme/colors";

export default function SignMlPredictor({
  onSignRecognized,
}: {
  onSignRecognized?: (sign: string) => void;
}) {
  const [isRecording, setIsRecording] = useState(false);
  const [countdown, setCountdown] = useState<number | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<SignPredictResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [streamActive, setStreamActive] = useState(false);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // Clean sign string: e.g. "thank_you" -> "Thank you"
  const formatLabel = (s: string) =>
    s
      .split("_")
      .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
      .join(" ");

  // Start webcam preview for ML
  const startPreview = useCallback(async () => {
    if (typeof navigator === "undefined" || !navigator.mediaDevices?.getUserMedia) return;
    try {
      setError(null);
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 480, height: 360, facingMode: "user" },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }
      setStreamActive(true);
    } catch (err: any) {
      console.warn("Could not start camera preview:", err);
      setError("Camera permission denied or camera not available.");
    }
  }, []);

  const stopPreview = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    setStreamActive(false);
  }, []);

  useEffect(() => {
    if (Platform.OS === "web") {
      startPreview();
    }
    return () => stopPreview();
  }, [startPreview, stopPreview]);

  // Start 2.5s recording
  const handleRecordGesture = () => {
    if (isRecording || isAnalyzing) return;
    setError(null);

    // 3-second countdown before recording starts
    setCountdown(3);
    const countTimer = setInterval(() => {
      setCountdown((prev) => {
        if (prev === null || prev <= 1) {
          clearInterval(countTimer);
          beginCapture();
          return null;
        }
        return prev - 1;
      });
    }, 800);
  };

  const beginCapture = async () => {
    if (!streamRef.current) {
      await startPreview();
    }
    if (!streamRef.current) {
      setError("Cannot capture video: camera stream not active.");
      return;
    }

    try {
      const mimeType = MediaRecorder.isTypeSupported("video/webm")
        ? "video/webm"
        : "video/mp4";
      const recorder = new MediaRecorder(streamRef.current, { mimeType });
      mediaRecorderRef.current = recorder;

      const chunks: BlobPart[] = [];
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunks.push(e.data);
      };

      recorder.onstop = async () => {
        setIsRecording(false);
        const blob = new Blob(chunks, { type: mimeType });
        await processVideoBlob(blob);
      };

      recorder.start();
      setIsRecording(true);

      // Record for 2.6 seconds (generates ~30 frames)
      setTimeout(() => {
        if (recorder.state === "recording") {
          recorder.stop();
        }
      }, 2600);
    } catch (err: any) {
      console.error("Recording error:", err);
      setError(`Recording failed: ${err.message || err}`);
      setIsRecording(false);
    }
  };

  const processVideoBlob = async (blob: Blob) => {
    setIsAnalyzing(true);
    setError(null);
    try {
      const res = await predictSignFromVideo(blob);
      setResult(res);
      const label = formatLabel(res.prediction.label);
      speak(label, "en");
      if (onSignRecognized) {
        onSignRecognized(label);
      }
    } catch (err: any) {
      console.error("ML Prediction error:", err);
      const msg = err?.message || "";
      if (msg.includes("Failed to fetch") || msg.includes("NetworkError")) {
        setError(
          "Cannot connect to FastAPI backend at http://127.0.0.1:8000. Ensure the server is running."
        );
      } else if (msg.includes("No frames could be extracted")) {
        setError(
          "Could not read video frames. Please keep your hands clearly visible in front of the camera and record again."
        );
      } else {
        setError(msg || "Backend ML evaluation failed.");
      }
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleFileUpload = (e: any) => {
    const file = e.target?.files?.[0];
    if (file) {
      processVideoBlob(file);
    }
  };

  const confidencePct = result
    ? Math.round((result.prediction.confidence ?? 0) * 100)
    : 0;

  // Top predictions from all_scores
  const topScores = result?.prediction.all_scores
    ? Object.entries(result.prediction.all_scores)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 3)
    : [];

  return (
    <View style={styles.container}>
      {/* Header Info */}
      <View style={styles.cardHeader}>
        <View style={styles.modelTag}>
          <Ionicons name="hardware-chip-outline" size={15} color={colors.accentStrong} />
          <Text style={styles.modelTagText}>PyTorch Conv1D + BiGRU + Attention</Text>
        </View>
        <Text style={styles.cardTitle}>ISL Deep Learning Model</Text>
        <Text style={styles.cardSubtitle}>
          Trained on Indian Sign Language (ISL). Record a short 2-second gesture to have the neural network classify your sign.
        </Text>
      </View>

      {/* Video Viewfinder */}
      <View style={[styles.viewfinderCard, shadow.md]}>
        {Platform.OS === "web" ? (
          <div style={{ position: "relative", width: "100%", height: "260px", background: "#0F1722", borderRadius: "12px", overflow: "hidden", display: "flex", alignItems: "center", justifyContent: "center" }}>
            <video
              ref={videoRef as any}
              autoPlay
              muted
              playsInline
              style={{ width: "100%", height: "100%", objectFit: "cover", transform: "scaleX(-1)" }}
            />
            {countdown !== null && (
              <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", background: "rgba(0,0,0,0.6)", color: "#fff", fontSize: "72px", fontWeight: "900" }}>
                {countdown}
              </div>
            )}
            {isRecording && (
              <div style={{ position: "absolute", top: "12px", left: "12px", background: "#EF4444", color: "#fff", padding: "4px 12px", borderRadius: "999px", fontWeight: "bold", fontSize: "13px", display: "flex", alignItems: "center", gap: "6px" }}>
                <span style={{ width: "8px", height: "8px", borderRadius: "50%", background: "#fff", display: "inline-block" }}></span>
                RECORDING GESTURE…
              </div>
            )}
          </div>
        ) : (
          <View style={styles.nativePlaceholder}>
            <Ionicons name="videocam" size={32} color={colors.accent} />
            <Text style={styles.nativeText}>Camera preview active</Text>
          </View>
        )}

        {/* Action Controls */}
        <View style={styles.controlBar}>
          <Pressable
            onPress={handleRecordGesture}
            disabled={isRecording || isAnalyzing}
            style={[
              styles.recordBtn,
              (isRecording || isAnalyzing) && styles.btnDisabled,
            ]}
            accessibilityRole="button"
          >
            <Ionicons
              name={isRecording ? "radio-button-on" : "videocam"}
              size={18}
              color={colors.textOnAccent}
            />
            <Text style={styles.recordBtnText}>
              {isRecording
                ? "Recording (hold sign)…"
                : countdown !== null
                ? `Ready in ${countdown}…`
                : "Record Gesture (2.5s)"}
            </Text>
          </Pressable>

          {Platform.OS === "web" && (
            <>
              <input
                ref={fileInputRef as any}
                type="file"
                accept="video/*"
                onChange={handleFileUpload}
                style={{ display: "none" }}
              />
              <Pressable
                onPress={() => fileInputRef.current?.click()}
                disabled={isRecording || isAnalyzing}
                style={[styles.uploadBtn, isAnalyzing && styles.btnDisabled]}
                accessibilityRole="button"
              >
                <Ionicons name="cloud-upload-outline" size={16} color={colors.accentStrong} />
                <Text style={styles.uploadBtnText}>Upload Video</Text>
              </Pressable>
            </>
          )}
        </View>
      </View>

      {/* Analyzing Spinner */}
      {isAnalyzing && (
        <View style={[styles.analyzingCard, shadow.sm]}>
          <ActivityIndicator size="small" color={colors.accent} />
          <Text style={styles.analyzingText}>
            Extracting 30 MediaPipe landmark frames & running PyTorch model inference…
          </Text>
        </View>
      )}

      {/* Error Message */}
      {!!error && (
        <View style={styles.errorBox}>
          <Ionicons name="alert-circle-outline" size={18} color={colors.critical} />
          <Text style={styles.errorText}>{error}</Text>
        </View>
      )}

      {/* Prediction Result Display */}
      {result && (
        <View style={[styles.resultCard, shadow.md]}>
          <View style={styles.resultHeader}>
            <Text style={styles.resultLabel}>AI PREDICTION</Text>
            <View style={styles.confidenceBadge}>
              <Text style={styles.confidenceBadgeText}>{confidencePct}% match</Text>
            </View>
          </View>

          <Text style={styles.predictedWord}>
            {formatLabel(result.prediction.label)}
          </Text>

          {/* Orientation & Frame details */}
          <View style={styles.metaRow}>
            <View style={styles.metaPill}>
              <Ionicons name="film-outline" size={13} color={colors.textMuted} />
              <Text style={styles.metaPillText}>
                {result.frames_used} frames evaluated
              </Text>
            </View>
            <View style={styles.metaPill}>
              <Ionicons name="swap-horizontal-outline" size={13} color={colors.textMuted} />
              <Text style={styles.metaPillText}>
                {result.orientation === "flipped" ? "Orientation: Flipped" : "Orientation: Original"}
              </Text>
            </View>
          </View>

          {/* Top 3 Breakdown */}
          {topScores.length > 0 && (
            <View style={styles.scoresWrap}>
              <Text style={styles.scoresHeader}>TOP PREDICTIONS</Text>
              {topScores.map(([s, score]) => {
                const pct = Math.round(score * 100);
                return (
                  <View key={s} style={styles.scoreRow}>
                    <Text style={styles.scoreName}>{formatLabel(s)}</Text>
                    <View style={styles.scoreTrack}>
                      <View style={[styles.scoreFill, { width: `${pct}%` }]} />
                    </View>
                    <Text style={styles.scorePct}>{pct}%</Text>
                  </View>
                );
              })}
            </View>
          )}

          <Pressable
            onPress={() => speak(formatLabel(result.prediction.label), "en")}
            style={styles.speakButton}
            accessibilityRole="button"
          >
            <Ionicons name="volume-high" size={16} color={colors.textOnAccent} />
            <Text style={styles.speakButtonText}>Pronounce Word</Text>
          </Pressable>
        </View>
      )}

      {/* Model Vocabulary List */}
      <View style={[styles.vocabCard, shadow.sm]}>
        <Text style={styles.vocabTitle}>
          Trained Neural Model Vocabulary ({ISL_ML_SIGNS.length} Signs)
        </Text>
        <Text style={styles.vocabSubtitle}>
          These 20 signs are recognized natively by your trained PyTorch ISL model:
        </Text>
        <View style={styles.chipsWrap}>
          {ISL_ML_SIGNS.map((s) => (
            <View key={s} style={styles.wordChip}>
              <Text style={styles.wordChipText}>{formatLabel(s)}</Text>
            </View>
          ))}
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    gap: spacing.md,
  },
  cardHeader: {
    gap: 4,
  },
  modelTag: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    alignSelf: "flex-start",
    backgroundColor: colors.accentSoft,
    paddingVertical: 4,
    paddingHorizontal: 10,
    borderRadius: radius.pill,
    marginBottom: 4,
  },
  modelTagText: {
    fontSize: 12,
    fontWeight: "700",
    color: colors.accentStrong,
  },
  cardTitle: {
    ...typography.heading,
    color: colors.textPrimary,
  },
  cardSubtitle: {
    ...typography.body,
    fontSize: 14,
    color: colors.textSecondary,
    lineHeight: 20,
  },
  viewfinderCard: {
    backgroundColor: colors.surface,
    borderRadius: radius.lg,
    padding: spacing.sm,
    borderWidth: 1,
    borderColor: colors.border,
    gap: spacing.sm,
  },
  nativePlaceholder: {
    height: 220,
    backgroundColor: colors.background,
    borderRadius: radius.md,
    alignItems: "center",
    justifyContent: "center",
    gap: spacing.sm,
  },
  nativeText: {
    ...typography.label,
    color: colors.textSecondary,
  },
  controlBar: {
    flexDirection: "row",
    gap: spacing.sm,
    paddingTop: 4,
  },
  recordBtn: {
    flex: 2,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 8,
    backgroundColor: colors.accent,
    paddingVertical: 12,
    borderRadius: radius.pill,
  },
  recordBtnText: {
    fontSize: 14,
    fontWeight: "800",
    color: colors.textOnAccent,
  },
  uploadBtn: {
    flex: 1,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 6,
    backgroundColor: colors.accentSoft,
    paddingVertical: 12,
    borderRadius: radius.pill,
  },
  uploadBtnText: {
    fontSize: 13,
    fontWeight: "700",
    color: colors.accentStrong,
  },
  btnDisabled: {
    opacity: 0.6,
  },
  analyzingCard: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.sm,
    backgroundColor: colors.surface,
    padding: spacing.md,
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: colors.border,
  },
  analyzingText: {
    ...typography.body,
    fontSize: 13,
    color: colors.textSecondary,
    flex: 1,
  },
  errorBox: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    backgroundColor: "#FEF2F2",
    padding: spacing.md,
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: "#FCA5A5",
  },
  errorText: {
    fontSize: 13,
    color: colors.critical,
    flex: 1,
  },
  resultCard: {
    backgroundColor: colors.surface,
    borderRadius: radius.lg,
    padding: spacing.lg,
    borderWidth: 2,
    borderColor: colors.accent,
    gap: spacing.sm,
  },
  resultHeader: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },
  resultLabel: {
    fontSize: 11,
    fontWeight: "800",
    letterSpacing: 1.2,
    color: colors.accentStrong,
  },
  confidenceBadge: {
    backgroundColor: colors.accentSoft,
    paddingVertical: 3,
    paddingHorizontal: 8,
    borderRadius: radius.pill,
  },
  confidenceBadgeText: {
    fontSize: 12,
    fontWeight: "800",
    color: colors.accentStrong,
  },
  predictedWord: {
    fontSize: 32,
    fontWeight: "900",
    color: colors.textPrimary,
  },
  metaRow: {
    flexDirection: "row",
    gap: spacing.sm,
    marginTop: 4,
  },
  metaPill: {
    flexDirection: "row",
    alignItems: "center",
    gap: 4,
    backgroundColor: colors.background,
    paddingVertical: 3,
    paddingHorizontal: 8,
    borderRadius: radius.sm,
  },
  metaPillText: {
    fontSize: 12,
    color: colors.textSecondary,
  },
  scoresWrap: {
    marginTop: spacing.sm,
    gap: 6,
  },
  scoresHeader: {
    fontSize: 11,
    fontWeight: "800",
    letterSpacing: 1,
    color: colors.textMuted,
    marginBottom: 2,
  },
  scoreRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.sm,
  },
  scoreName: {
    width: 90,
    fontSize: 13,
    fontWeight: "600",
    color: colors.textPrimary,
  },
  scoreTrack: {
    flex: 1,
    height: 8,
    backgroundColor: colors.border,
    borderRadius: radius.pill,
    overflow: "hidden",
  },
  scoreFill: {
    height: "100%",
    backgroundColor: colors.accent,
    borderRadius: radius.pill,
  },
  scorePct: {
    width: 38,
    fontSize: 12,
    fontWeight: "700",
    color: colors.textMuted,
    textAlign: "right",
  },
  speakButton: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 6,
    backgroundColor: colors.accent,
    paddingVertical: 10,
    borderRadius: radius.pill,
    marginTop: spacing.xs,
  },
  speakButtonText: {
    fontSize: 13,
    fontWeight: "800",
    color: colors.textOnAccent,
  },
  vocabCard: {
    backgroundColor: colors.surface,
    padding: spacing.md,
    borderRadius: radius.lg,
    borderWidth: 1,
    borderColor: colors.border,
    gap: spacing.xs,
  },
  vocabTitle: {
    ...typography.title,
    color: colors.textPrimary,
  },
  vocabSubtitle: {
    fontSize: 13,
    color: colors.textMuted,
    marginBottom: spacing.xs,
  },
  chipsWrap: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 6,
  },
  wordChip: {
    backgroundColor: colors.accentSoft,
    paddingVertical: 5,
    paddingHorizontal: 10,
    borderRadius: radius.pill,
  },
  wordChipText: {
    fontSize: 12,
    fontWeight: "700",
    color: colors.accentStrong,
  },
});
