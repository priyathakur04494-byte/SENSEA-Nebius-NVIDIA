import React from "react";
import { StyleSheet, Text, View } from "react-native";
import { CaptionEntry, Speaker } from "@/types";
import { colors, radius, spacing, typography } from "@/theme/colors";

interface Props {
  caption: CaptionEntry;
  speaker?: Speaker;
  showSimplified: boolean;
  showTranslation: boolean;
}

// Words worth calling out in a caption line — kept intentionally small so
// highlighting stays meaningful. This is presentation-only styling; it does
// not change the underlying transcript text.
const HIGHLIGHT_PATTERN =
  /\b(?:\d{1,2}(?::\d{2})?\s?(?:am|pm|AM|PM)?|deadline|friday|monday|tuesday|wednesday|thursday|saturday|sunday|today|tomorrow|urgent|asap|meeting|report|submit)\b/g;

function renderHighlighted(text: string) {
  const nodes: React.ReactNode[] = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null;
  HIGHLIGHT_PATTERN.lastIndex = 0;

  while ((match = HIGHLIGHT_PATTERN.exec(text)) !== null) {
    if (match.index > lastIndex) {
      nodes.push(text.slice(lastIndex, match.index));
    }
    nodes.push(
      <Text key={`${match.index}-${match[0]}`} style={styles.highlight}>
        {match[0]}
      </Text>
    );
    lastIndex = match.index + match[0].length;
  }
  if (lastIndex < text.length) {
    nodes.push(text.slice(lastIndex));
  }
  return nodes;
}

export default function CaptionBubble({ caption, speaker, showSimplified, showTranslation }: Props) {
  const displayText = showTranslation && caption.translatedText
    ? caption.translatedText
    : showSimplified && caption.simplifiedText
    ? caption.simplifiedText
    : caption.originalText;

  const speakerName = speaker?.name ?? speaker?.label ?? "Unknown speaker";

  return (
    <View
      style={[styles.row, { opacity: caption.isFinal ? 1 : 0.6 }]}
      accessible
      accessibilityLabel={`${speakerName} said: ${displayText}`}
    >
      <View style={[styles.speakerMark, { backgroundColor: speaker?.colorTag ?? colors.accent }]} />
      <View style={styles.body}>
        <View style={styles.header}>
          <Text style={styles.speakerName}>{speakerName.toUpperCase()}</Text>
          {!caption.isFinal && <Text style={styles.liveTag}>listening…</Text>}
        </View>
        <Text style={styles.text}>
          {showTranslation && caption.translatedText ? displayText : renderHighlighted(displayText)}
        </Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  row: { flexDirection: "row", marginBottom: spacing.lg },
  speakerMark: { width: 4, borderRadius: radius.pill, marginRight: spacing.md, alignSelf: "stretch" },
  body: { flex: 1 },
  header: { flexDirection: "row", alignItems: "center", marginBottom: spacing.xs, gap: spacing.sm },
  speakerName: { ...typography.label, color: colors.textSecondary, letterSpacing: 0.6 },
  liveTag: { ...typography.label, color: colors.accent, fontStyle: "italic" },
  text: { ...typography.caption, color: colors.textPrimary },
  highlight: {
    color: colors.accentStrong,
    fontWeight: "700",
    backgroundColor: colors.accentSoft,
  },
});
