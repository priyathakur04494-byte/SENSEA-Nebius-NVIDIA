import React from "react";
import Svg, { G, Line, Path, Rect } from "react-native-svg";
import { FingerState, HAND_SHAPES, ShapeName, ThumbState } from "@/config/signLanguageData";

/**
 * A hand drawn from simple shapes, driven by a small description (which
 * fingers are up / curled, where the thumb is). One component draws every
 * hand shape in the sign library, so adding a new shape is just data.
 *
 * Drawn palm-facing-viewer, fingers pointing up, in a 100 x 160 box with the
 * palm centre at (50, 85). A forearm and sleeve run off the bottom so the
 * hand reads as attached to a person once it's rotated on the stage.
 */

export const HAND_VIEWBOX_W = 100;
export const HAND_VIEWBOX_H = 160;

const SKIN = "#F0C9A8";
const SKIN_EDGE = "#7A4E3A";
const SLEEVE = "#475569";
const SLEEVE_EDGE = "#1E293B";

const FINGER_X = [33, 44.3, 55.6, 66.9];
const FINGER_LEN = [40, 44, 40, 31];
const FINGER_W = 10;
const BASE_Y = 72;

interface Part {
  key: string;
  kind: "rect";
  x: number;
  y: number;
  w: number;
  h: number;
  r: number;
  rotate?: { deg: number; cx: number; cy: number };
}

function fingerPart(i: number, state: FingerState, spread: number): Part {
  const cx = FINGER_X[i];
  const fan = spread * (i - 1.5);
  if (state === "up") {
    const len = FINGER_LEN[i];
    return {
      key: `f${i}`,
      kind: "rect",
      x: cx - FINGER_W / 2,
      y: BASE_Y - len,
      w: FINGER_W,
      h: len + 10,
      r: FINGER_W / 2,
      rotate: { deg: fan, cx, cy: BASE_Y },
    };
  }
  if (state === "bent") {
    const len = FINGER_LEN[i] * 0.62;
    // Bent fingers lean toward the thumb side so they can meet it.
    const lean = (1.5 - i) * 7;
    return {
      key: `f${i}`,
      kind: "rect",
      x: cx - FINGER_W / 2,
      y: BASE_Y - len,
      w: FINGER_W,
      h: len + 10,
      r: FINGER_W / 2,
      rotate: { deg: lean, cx, cy: BASE_Y },
    };
  }
  // "down": a knuckle bump showing the finger folded onto the palm.
  return { key: `f${i}`, kind: "rect", x: cx - FINGER_W / 2, y: 60, w: FINGER_W, h: 20, r: FINGER_W / 2 };
}

function thumbPart(state: ThumbState): Part {
  switch (state) {
    case "out":
      return {
        key: "t",
        kind: "rect",
        x: 24,
        y: 92 - 36,
        w: 11,
        h: 44,
        r: 5.5,
        rotate: { deg: -52, cx: 30, cy: 92 },
      };
    case "side":
      return { key: "t", kind: "rect", x: 19, y: 62, w: 11, h: 40, r: 5.5 };
    case "touch":
      return {
        key: "t",
        kind: "rect",
        x: 26.5,
        y: 90 - 38,
        w: 11,
        h: 46,
        r: 5.5,
        rotate: { deg: 24, cx: 32, cy: 90 },
      };
    case "in":
    default:
      return { key: "t", kind: "rect", x: 30, y: 80, w: 38, h: 11, r: 5.5, rotate: { deg: -8, cx: 49, cy: 85 } };
  }
}

function rectEl(p: Part, extra: object) {
  return (
    <Rect
      key={p.key}
      x={p.x}
      y={p.y}
      width={p.w}
      height={p.h}
      rx={p.r}
      ry={p.r}
      transform={p.rotate ? `rotate(${p.rotate.deg} ${p.rotate.cx} ${p.rotate.cy})` : undefined}
      {...extra}
    />
  );
}

export default function HandSvg({
  shape,
  mirrored = false,
  width,
  height,
  forearm = true,
}: {
  shape: ShapeName;
  mirrored?: boolean;
  width: number;
  height: number;
  forearm?: boolean;
}) {
  const def = HAND_SHAPES[shape];
  const spread = def.spread ?? 0;
  const fingers = def.fingers.map((s, i) => fingerPart(i, s, spread));
  const thumb = thumbPart(def.thumb);
  const thumbOnTop = def.thumb === "in";

  // Everything that makes up the hand silhouette (palm + fingers + thumb that
  // sits beside the hand). Drawn twice: first as a thick dark outline, then
  // as skin fill — so overlapping shapes merge into one clean outline.
  const silhouette: Part[] = [
    { key: "palm", kind: "rect", x: 28, y: 66, w: 44, h: 38, r: 12 },
    ...fingers,
    ...(thumbOnTop ? [] : [thumb]),
  ];

  return (
    <Svg
      width={width}
      height={height}
      viewBox={`0 5 ${HAND_VIEWBOX_W} ${HAND_VIEWBOX_H}`}
      style={mirrored ? { transform: [{ scaleX: -1 }] } : undefined}
    >
      {forearm && (
        <G>
          <Rect x={37} y={98} width={26} height={42} fill={SKIN} stroke={SKIN_EDGE} strokeWidth={1.5} />
          <Rect x={30} y={128} width={40} height={36} rx={4} fill={SLEEVE} stroke={SLEEVE_EDGE} strokeWidth={1.5} />
        </G>
      )}

      {silhouette.map((p) => rectEl(p, { fill: SKIN_EDGE, stroke: SKIN_EDGE, strokeWidth: 3.2 }))}
      {silhouette.map((p) => rectEl(p, { fill: SKIN }))}

      {/* Curled-finger creases so a fist reads as folded fingers. */}
      {def.fingers.map((s, i) =>
        s === "down" ? (
          <Line
            key={`c${i}`}
            x1={FINGER_X[i] - 3.5}
            y1={76}
            x2={FINGER_X[i] + 3.5}
            y2={76}
            stroke={SKIN_EDGE}
            strokeWidth={1.3}
            strokeLinecap="round"
          />
        ) : null
      )}

      {/* A thumb lying across the palm needs its own outline to stay visible. */}
      {thumbOnTop && rectEl(thumb, { fill: SKIN, stroke: SKIN_EDGE, strokeWidth: 1.6 })}

      {/* Thumb nail hint on the extended thumb. */}
      {def.thumb === "out" && (
        <Path d="M16 63 q3 -3 6 -1" stroke={SKIN_EDGE} strokeWidth={1.2} fill="none" strokeLinecap="round" />
      )}
    </Svg>
  );
}
