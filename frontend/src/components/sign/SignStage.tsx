import React, { useEffect, useRef } from "react";
import { Animated, Easing, StyleSheet, View } from "react-native";
import Svg, { Circle, Ellipse, Path, Polygon, Rect } from "react-native-svg";
import HandSvg, { HAND_VIEWBOX_H, HAND_VIEWBOX_W } from "@/components/sign/HandSvg";
import { HandPose, Point, SignFrame, STAGE_H, STAGE_W } from "@/config/signLanguageData";
import { colors, radius } from "@/theme/colors";

/**
 * The signer: a simple front-facing figure with a hand (or two) that glides
 * between the frames of a sign. The figure and the dashed motion guide are
 * SVG; each hand is its own animated view so it can move and rotate smoothly
 * (using the native driver) — and snaps instantly when the user has "reduce
 * motion" turned on.
 */

const HAND_BASE_W = 80; // hand width, in stage units, at scale 1
const HAND_BASE_H = (HAND_BASE_W * HAND_VIEWBOX_H) / HAND_VIEWBOX_W;

const SKIN = "#F0C9A8";
const SKIN_EDGE = "#7A4E3A";
const HAIR = "#2B2118";
const GUIDE = colors.warning;

function smoothPath(pts: Point[]): string {
  if (pts.length < 2) return "";
  if (pts.length === 2) return `M${pts[0][0]} ${pts[0][1]} L${pts[1][0]} ${pts[1][1]}`;
  let d = `M${pts[0][0]} ${pts[0][1]}`;
  for (let i = 1; i < pts.length - 1; i++) {
    const mx = (pts[i][0] + pts[i + 1][0]) / 2;
    const my = (pts[i][1] + pts[i + 1][1]) / 2;
    d += ` Q${pts[i][0]} ${pts[i][1]} ${mx} ${my}`;
  }
  const last = pts[pts.length - 1];
  d += ` L${last[0]} ${last[1]}`;
  return d;
}

function arrowHead(pts: Point[]): string {
  const [x2, y2] = pts[pts.length - 1];
  const [x1, y1] = pts[pts.length - 2];
  const ang = Math.atan2(y2 - y1, x2 - x1);
  const size = 11;
  const p = (a: number, len: number): Point => [x2 - len * Math.cos(a), y2 - len * Math.sin(a)];
  const left = p(ang - 0.45, size);
  const right = p(ang + 0.45, size);
  return `${x2},${y2} ${left[0]},${left[1]} ${right[0]},${right[1]}`;
}

function Figure() {
  return (
    <>
      {/* torso */}
      <Path d="M20 340 C20 268 66 214 150 206 C234 214 280 268 280 340 Z" fill={colors.primary} />
      {/* neck + collar */}
      <Rect x={132} y={140} width={36} height={70} fill={SKIN} />
      <Path d="M126 204 L150 240 L174 204 Z" fill={SKIN} />
      <Path d="M126 204 L150 240 L174 204" fill="none" stroke={colors.primarySoft} strokeWidth={4} strokeLinejoin="round" />
      {/* ears */}
      <Ellipse cx={105} cy={102} rx={7} ry={12} fill={SKIN} stroke={SKIN_EDGE} strokeWidth={1.5} />
      <Ellipse cx={195} cy={102} rx={7} ry={12} fill={SKIN} stroke={SKIN_EDGE} strokeWidth={1.5} />
      {/* head */}
      <Ellipse cx={150} cy={100} rx={44} ry={54} fill={SKIN} stroke={SKIN_EDGE} strokeWidth={1.5} />
      {/* hair */}
      <Path
        d="M105 94 C100 36 200 36 195 94 C188 70 172 60 150 60 C128 60 112 70 105 94 Z"
        fill={HAIR}
      />
      {/* face */}
      <Circle cx={133} cy={102} r={3.2} fill={HAIR} />
      <Circle cx={167} cy={102} r={3.2} fill={HAIR} />
      <Path d="M135 128 Q150 138 165 128" fill="none" stroke={SKIN_EDGE} strokeWidth={2.4} strokeLinecap="round" />
    </>
  );
}

function AnimatedHand({
  pose,
  k,
  duration,
  reduceMotion,
}: {
  pose: HandPose;
  k: number;
  duration: number;
  reduceMotion: boolean;
}) {
  const w = HAND_BASE_W * k;
  const h = HAND_BASE_H * k;
  const x = useRef(new Animated.Value(pose.x)).current;
  const y = useRef(new Animated.Value(pose.y)).current;
  const rot = useRef(new Animated.Value(pose.rot)).current;
  const scale = useRef(new Animated.Value(pose.scale ?? 1)).current;

  useEffect(() => {
    const targets: [Animated.Value, number][] = [
      [x, pose.x],
      [y, pose.y],
      [rot, pose.rot],
      [scale, pose.scale ?? 1],
    ];
    if (reduceMotion || duration <= 0) {
      targets.forEach(([v, t]) => v.setValue(t));
      return;
    }
    const anim = Animated.parallel(
      targets.map(([v, t]) =>
        Animated.timing(v, {
          toValue: t,
          duration,
          easing: Easing.inOut(Easing.cubic),
          useNativeDriver: true,
        })
      )
    );
    anim.start();
    return () => anim.stop();
  }, [pose.x, pose.y, pose.rot, pose.scale, duration, reduceMotion, x, y, rot, scale]);

  // Position in screen px (stage units * k), offset so (x, y) is the palm centre.
  const translateX = Animated.subtract(Animated.multiply(x, k), w / 2);
  const translateY = Animated.subtract(Animated.multiply(y, k), h / 2);
  const rotate = rot.interpolate({ inputRange: [-360, 360], outputRange: ["-360deg", "360deg"] });

  return (
    <Animated.View
      pointerEvents="none"
      style={[
        styles.hand,
        {
          width: w,
          height: h,
          transform: [{ translateX }, { translateY }, { rotate }, { scale }],
        },
      ]}
    >
      <HandSvg shape={pose.shape} mirrored={!!pose.left} width={w} height={h} />
    </Animated.View>
  );
}

export default function SignStage({
  frame,
  signKey,
  width,
  durationMs,
  reduceMotion,
  label,
}: {
  frame: SignFrame;
  /** Changes when a different sign is shown, so hands are re-created fresh. */
  signKey: string;
  width: number;
  durationMs: number;
  reduceMotion: boolean;
  label: string;
}) {
  const k = width / STAGE_W;
  const height = STAGE_H * k;
  const fade = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    if (reduceMotion) {
      fade.setValue(1);
      return;
    }
    fade.setValue(0);
    Animated.timing(fade, { toValue: 1, duration: 260, useNativeDriver: true }).start();
  }, [signKey, reduceMotion, fade]);

  const arrow = frame.arrow && frame.arrow.length >= 2 ? frame.arrow : null;

  return (
    <View
      style={[styles.stage, { width, height }]}
      accessible
      accessibilityRole="image"
      accessibilityLabel={label}
    >
      <Svg width={width} height={height} viewBox={`0 0 ${STAGE_W} ${STAGE_H}`} style={StyleSheet.absoluteFill}>
        <Figure />
      </Svg>

      <Animated.View style={[StyleSheet.absoluteFill, { opacity: fade }]} pointerEvents="none">
        {frame.hands.map((pose, i) => (
          <AnimatedHand
            key={`${signKey}-${i}`}
            pose={pose}
            k={k}
            duration={durationMs}
            reduceMotion={reduceMotion}
          />
        ))}
      </Animated.View>

      {arrow && (
        <Svg
          width={width}
          height={height}
          viewBox={`0 0 ${STAGE_W} ${STAGE_H}`}
          style={StyleSheet.absoluteFill}
          pointerEvents="none"
        >
          {/* white halo keeps the guide readable on both the skin and the navy shirt */}
          <Path d={smoothPath(arrow)} stroke="#FFFFFF" strokeWidth={7} strokeLinecap="round" fill="none" opacity={0.9} />
          <Path
            d={smoothPath(arrow)}
            stroke={GUIDE}
            strokeWidth={3.5}
            strokeDasharray="2 7"
            strokeLinecap="round"
            fill="none"
          />
          <Polygon points={arrowHead(arrow)} fill={GUIDE} stroke="#FFFFFF" strokeWidth={2} strokeLinejoin="round" />
        </Svg>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  stage: {
    backgroundColor: colors.surfaceMuted,
    borderRadius: radius.lg,
    overflow: "hidden",
    alignSelf: "center",
  },
  hand: { position: "absolute", left: 0, top: 0 },
});
