import React, { useEffect, useMemo, useRef } from "react";
import { buildHandTrackerHtml } from "@/services/handTrackerPage";
import { HandTrackerViewProps, TrackerMessage } from "./handTrackerTypes";

/** Web: the camera page runs in an iframe (allowed to use the camera). */
export default function HandTrackerView({ onMessage, onHandle, height }: HandTrackerViewProps) {
  const ref = useRef<HTMLIFrameElement | null>(null);
  const html = useMemo(() => buildHandTrackerHtml("web", window.location.origin), []);
  const cb = useRef(onMessage);
  cb.current = onMessage;

  useEffect(() => {
    const onWin = (e: MessageEvent) => {
      if (e.source !== ref.current?.contentWindow) return;
      if (e.data && typeof e.data === "object" && typeof e.data.type === "string") cb.current(e.data as TrackerMessage);
    };
    window.addEventListener("message", onWin);
    onHandle({ send: (m) => ref.current?.contentWindow?.postMessage(m, "*") });
    return () => window.removeEventListener("message", onWin);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return React.createElement("iframe", {
    ref,
    srcDoc: html,
    title: "Camera hand tracking",
    allow: "camera",
    style: { border: 0, width: "100%", height, borderRadius: 14, display: "block", background: "#0F1722" },
  });
}
