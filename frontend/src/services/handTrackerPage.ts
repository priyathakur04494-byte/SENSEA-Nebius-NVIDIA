/**
 * Builds the HTML page that runs the camera + hand-tracking engine.
 *
 *  - Web: loads the MediaPipe runtime and the hand model from /hand-tracker/
 *    (files in `public/hand-tracker`, bundled with the app, so it works offline).
 *  - Phones (WebView): those files are not served by Expo Go, so the same
 *    runtime and model are fetched from public CDNs on first use.
 */
import { HAND_TRACKER_TEMPLATE } from "@/generated/handTrackerHtml";

const MEDIAPIPE_VERSION = "1.2.0"; // keep in sync with public/hand-tracker/vision_bundle.mjs
const CDN = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}`;
const MODEL_CDN =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

export function buildHandTrackerHtml(target: "web" | "native", origin = ""): string {
  const cfg =
    target === "web"
      ? {
          bundleUrl: `${origin}/hand-tracker/vision_bundle.mjs`,
          wasmUrl: `${origin}/hand-tracker/wasm`,
          modelUrl: `${origin}/hand-tracker/hand_landmarker.task`,
        }
      : { bundleUrl: `${CDN}/vision_bundle.mjs`, wasmUrl: `${CDN}/wasm`, modelUrl: MODEL_CDN };
  return HAND_TRACKER_TEMPLATE.replace("/*__CONFIG__*/null", () => JSON.stringify(cfg));
}
