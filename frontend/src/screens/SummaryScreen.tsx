import React from "react";
import ContextPanel from "@/components/ContextPanel";

/**
 * Route kept as "Summary" for navigation-type compatibility, but the screen
 * is now the full "AI Context" experience (Live Summary, Important Points,
 * Decisions, You May Have Missed, Ask SENSEA) — see ContextPanel. On wide
 * layouts this same panel also renders inline next to Live Captions
 * (LiveCaptionScreen), so the logic lives in one shared component.
 */
export default function SummaryScreen() {
  return <ContextPanel />;
}
