// Central design tokens.
//
// SENSEA is an accessibility tool used in meetings, classrooms, hospitals,
// interviews and emergencies — it has to read as calm, trustworthy,
// professional software, not an "AI startup" demo. Color and shape (never
// color alone) carry every state change, since users may have color-vision
// differences and can't rely on sound cues at all.
//
// Palette direction: warm-white base, deep navy for primary text/surfacing,
// a single restrained teal accent for interactive/brand moments, and
// standard (not neon) amber/red for warning/critical. No gradients, no
// glassmorphism, no glow.

export const colors = {
  // ---- Surfaces ----
  background: "#F7F5F0", // warm off-white, easier on the eyes than pure white
  surface: "#FFFFFF",
  surfaceElevated: "#FFFFFF",
  surfaceMuted: "#EFECE4", // for subtle section backgrounds / input fills
  border: "#DDD8CC",
  borderStrong: "#C7C0B0",

  // ---- Text ----
  textPrimary: "#1B2430", // deep navy-charcoal, not pure black
  textSecondary: "#4B5563",
  textMuted: "#7B8390",
  textOnAccent: "#FFFFFF",
  textOnDark: "#F5F6F7",

  // ---- Primary / brand ----
  // Deep navy is the primary brand color (headers, active states on dark
  // surfaces); muted teal is the interactive accent (buttons, links,
  // highlighted words, focus states).
  primary: "#1E293B",
  primarySoft: "#E3E7EC",
  accent: "#2F7A72", // muted teal — accessible against white and dark navy
  accentSoft: "#E1EEEC",
  accentStrong: "#1F5A54",

  // A secondary supporting hue (soft blue) for a small amount of visual
  // variety — used sparingly (e.g. translation panel, secondary chips) so
  // the palette still reads as considered rather than a color-per-feature
  // grab bag.
  supportBlue: "#4C6FA5",
  supportBlueSoft: "#E7ECF4",
  supportGray: "#8A93A0",

  // ---- Status (icon + text always accompany these — never color alone) ----
  normal: "#2F7D52",
  normalSoft: "#E3F1E7",
  warning: "#9A6B12", // muted amber, darkened for AA contrast on light bg
  warningSoft: "#F6ECD8",
  critical: "#B3261E", // accessible red
  criticalSoft: "#F8E3E1",

  // Kept for any lingering references; map old alert* names onto the new
  // status colors so nothing silently goes untyped.
  alertNormal: "#2F7D52",
  alertImportant: "#9A6B12",
  alertEmergency: "#B3261E",

  // Restrained per-speaker distinction — muted, not a rainbow. Distinguish
  // primarily through the speaker's position/label; color is a light aid.
  speakerPalette: ["#2F7A72", "#4C6FA5", "#8A5A44", "#6B6F8A", "#7A6A9A", "#4A7A6A", "#9A6B12"],

  // Every "feature" now shares the same restrained accent family instead of
  // a distinct vivid hue per card — differentiation comes from icon + copy,
  // not color-coding the whole app like a dashboard.
  featureColors: {
    captions: "#2F7A72",
    alerts: "#9A6B12",
    speak: "#4C6FA5",
    summary: "#2F7A72",
    memory: "#4C6FA5",
  },
};

export const spacing = {
  xs: 4,
  sm: 8,
  md: 16,
  lg: 24,
  xl: 32,
};

export const radius = {
  sm: 6,
  md: 10,
  lg: 14,
  pill: 999,
};

// Minimal, functional elevation — a hint of depth on cards/sheets, nothing
// decorative. No colored glow anywhere in the redesigned UI.
export const shadow = {
  none: { shadowOpacity: 0, elevation: 0 },
  sm: {
    shadowColor: "#1B2430",
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.06,
    shadowRadius: 3,
    elevation: 1,
  },
  md: {
    shadowColor: "#1B2430",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.08,
    shadowRadius: 8,
    elevation: 2,
  },
  lg: {
    shadowColor: "#1B2430",
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.1,
    shadowRadius: 16,
    elevation: 4,
  },
  // Kept as an alias to `sm` (not a real glow) so any leftover `shadow.glow`
  // reference degrades to a normal, restrained shadow instead of a bright
  // colored glow.
  glow: {
    shadowColor: "#1B2430",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.08,
    shadowRadius: 8,
    elevation: 2,
  },
};

export const typography = {
  caption: { fontSize: 22, lineHeight: 31, fontWeight: "500" as const },
  captionLarge: { fontSize: 28, lineHeight: 38, fontWeight: "600" as const },
  body: { fontSize: 15, lineHeight: 22, fontWeight: "400" as const },
  label: { fontSize: 13, lineHeight: 18, fontWeight: "600" as const },
  title: { fontSize: 20, lineHeight: 27, fontWeight: "700" as const },
  heading: { fontSize: 26, lineHeight: 33, fontWeight: "800" as const },
};

// Breakpoints used to switch between the mobile-first single-column layout
// and the wider two-column "Live Captions | AI Context" workspace layout on
// tablet/laptop/desktop.
export const breakpoints = {
  tablet: 768,
  desktop: 1024,
};
