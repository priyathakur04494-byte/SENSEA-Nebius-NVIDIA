import React, { useEffect, useMemo, useRef } from "react";
import { PermissionsAndroid, Platform } from "react-native";
import { WebView as RNWebView, WebViewMessageEvent } from "react-native-webview";
import { buildHandTrackerHtml } from "@/services/handTrackerPage";
import { HandTrackerViewProps, TrackerMessage } from "./handTrackerTypes";

// react-native-webview's class typings clash with React 19's JSX types; the runtime is fine.
const WebView = RNWebView as unknown as React.ComponentType<any>;

/**
 * Phones: the camera page runs in a WebView. The page does the camera and hand
 * tracking; this component only relays messages.
 */
export default function HandTrackerView({ onMessage, onHandle, height }: HandTrackerViewProps) {
  const ref = useRef<{ injectJavaScript: (js: string) => void } | null>(null);
  const html = useMemo(() => buildHandTrackerHtml("native"), []);
  const [granted, setGranted] = React.useState(Platform.OS !== "android");

  useEffect(() => {
    onHandle({
      send: (m) => ref.current?.injectJavaScript(`window.__host(${JSON.stringify(m)}); true;`),
    });
    if (Platform.OS === "android") {
      PermissionsAndroid.request(PermissionsAndroid.PERMISSIONS.CAMERA).then((r) => {
        if (r === PermissionsAndroid.RESULTS.GRANTED) setGranted(true);
        else onMessage({ type: "status", state: "denied", message: "Camera permission was denied. Enable it in Settings." });
      });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (!granted) return null;
  return (
    <WebView
      ref={ref}
      source={{ html, baseUrl: "https://sensea.app" }}
      originWhitelist={["*"]}
      style={{ height, borderRadius: 14, backgroundColor: "#0F1722" }}
      javaScriptEnabled
      allowsInlineMediaPlayback
      mediaPlaybackRequiresUserAction={false}
      mediaCapturePermissionGrantType="grant"
      onMessage={(e: WebViewMessageEvent) => {
        try {
          onMessage(JSON.parse(e.nativeEvent.data) as TrackerMessage);
        } catch {
          /* ignore malformed */
        }
      }}
      onError={() => onMessage({ type: "status", state: "error", message: "The camera view could not load." })}
    />
  );
}
