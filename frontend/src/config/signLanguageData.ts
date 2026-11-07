/**
 * SIGN LANGUAGE LIBRARY
 * ----------------------
 * Data for the "Signs" tab. Every sign is a short sequence of *frames*; each
 * frame says which hand shape(s) to show, where they sit on the signer, and
 * what the learner should do at that step. The Signs screen plays those frames
 * turn by turn and draws them with `SignStage` (no video or image files).
 *
 * Language: American Sign Language (ASL). Sign languages are NOT universal —
 * India uses Indian Sign Language (ISL), the UK uses BSL, and so on. To add
 * another language later, add a new `SignLanguage` entry plus its own sign
 * list and the screen can offer a language switch with no other changes.
 *
 * Stage coordinates: a 300 x 340 canvas, signer facing the viewer. The
 * signer's dominant (right) hand therefore appears on the viewer's LEFT.
 */

export interface SignLanguage {
  id: string;
  name: string;
  shortName: string;
  note: string;
}

export const SIGN_LANGUAGES: SignLanguage[] = [
  {
    id: "asl",
    name: "American Sign Language",
    shortName: "ASL",
    note:
      "Sign languages differ by country. These signs are ASL. In India, Indian Sign Language (ISL) is widely used — some signs match, many don't. Confirm with a Deaf signer or teacher.",
  },
];

// ---------------------------------------------------------------------------
// Hand shapes
// ---------------------------------------------------------------------------

export type FingerState = "up" | "down" | "bent";
export type ThumbState = "out" | "in" | "side" | "touch";

export interface HandShape {
  /** index, middle, ring, pinky */
  fingers: [FingerState, FingerState, FingerState, FingerState];
  thumb: ThumbState;
  /** Degrees of fan-out between extended fingers (e.g. the "W" hand). */
  spread?: number;
}

export type ShapeName =
  | "flat"
  | "fistA"
  | "fistS"
  | "point"
  | "iLoveYou"
  | "wHand"
  | "flatO"
  | "noOpen"
  | "noClosed";

export const HAND_SHAPES: Record<ShapeName, HandShape> = {
  flat: { fingers: ["up", "up", "up", "up"], thumb: "in" },
  fistA: { fingers: ["down", "down", "down", "down"], thumb: "side" },
  fistS: { fingers: ["down", "down", "down", "down"], thumb: "in" },
  point: { fingers: ["up", "down", "down", "down"], thumb: "in" },
  iLoveYou: { fingers: ["up", "down", "down", "up"], thumb: "out" },
  wHand: { fingers: ["up", "up", "up", "down"], thumb: "in", spread: 9 },
  flatO: { fingers: ["bent", "bent", "bent", "bent"], thumb: "touch" },
  noOpen: { fingers: ["up", "up", "down", "down"], thumb: "out" },
  noClosed: { fingers: ["bent", "bent", "down", "down"], thumb: "touch" },
};

// ---------------------------------------------------------------------------
// Where things are on the signer (stage coordinates)
// ---------------------------------------------------------------------------

export const STAGE_W = 300;
export const STAGE_H = 340;

export interface HandPose {
  shape: ShapeName;
  /** Centre of the palm on the stage. */
  x: number;
  y: number;
  /** Degrees clockwise; 0 = fingers pointing up. */
  rot: number;
  /** 1 = default size. Slightly larger reads as "closer to the viewer". */
  scale?: number;
  /** Mirror the hand so it reads as the signer's left (non-dominant) hand. */
  left?: boolean;
}

/** A dashed guide showing the path the hand travels on this step. */
export type Point = [number, number];

export interface SignFrame {
  /** One pose per hand. Hand count must stay the same across a sign's frames. */
  hands: HandPose[];
  /** What the learner does on this step. */
  instruction: string;
  /** Optional motion guide drawn on this step. */
  arrow?: Point[];
}

export type SignCategory = "greetings" | "courtesy" | "replies" | "needs" | "emergency";

export const SIGN_CATEGORIES: { id: SignCategory; label: string }[] = [
  { id: "greetings", label: "Greetings" },
  { id: "courtesy", label: "Courtesy" },
  { id: "replies", label: "Yes / No" },
  { id: "needs", label: "Needs" },
  { id: "emergency", label: "Emergency" },
];

export interface SignEntry {
  id: string;
  word: string;
  /** What the sign means, in plain language. */
  meaning: string;
  /** When someone would use it. */
  useWhen: string;
  categories: SignCategory[];
  frames: SignFrame[];
}

// Small helper to draw a (mostly) circular guide, ending with an arrowhead.
function circle(cx: number, cy: number, r: number): Point[] {
  const pts: Point[] = [];
  for (let a = -90; a <= 200; a += 30) {
    const rad = (a * Math.PI) / 180;
    pts.push([cx + r * Math.cos(rad), cy + r * Math.sin(rad)]);
  }
  return pts;
}

// Anchors — kept in one place so signs stay consistent with the figure.
const CHEST = { x: 150, y: 252 };

const PLEASE_RING = circle(CHEST.x, CHEST.y, 18);
const ringPose = (shape: ShapeName, angle: number): HandPose => {
  const rad = (angle * Math.PI) / 180;
  return {
    shape,
    x: CHEST.x + 14 * Math.cos(rad),
    y: CHEST.y + 14 * Math.sin(rad),
    rot: 78,
  };
};

export const ASL_SIGNS: SignEntry[] = [
  {
    id: "hello",
    word: "Hello",
    meaning: "A friendly greeting — hello or hi.",
    useWhen: "When meeting someone or getting their attention in a friendly way.",
    categories: ["greetings"],
    frames: [
      {
        hands: [{ shape: "flat", x: 80, y: 112, rot: 14 }],
        instruction: "Hold a flat hand, fingers together, at your temple — like the start of a salute.",
        arrow: [[88, 66], [62, 58], [34, 52]],
      },
      {
        hands: [{ shape: "flat", x: 40, y: 98, rot: -28 }],
        instruction: "Move the hand outward and away from your head, with a small flick.",
      },
    ],
  },
  {
    id: "thankyou",
    word: "Thank you",
    meaning: "Thank you — showing appreciation.",
    useWhen: "After someone helps you, serves you, or does something kind.",
    categories: ["courtesy"],
    frames: [
      {
        hands: [{ shape: "flat", x: 126, y: 194, rot: 30 }],
        instruction: "Touch your fingertips to your chin, palm facing you.",
        arrow: [[146, 176], [136, 210], [120, 244]],
      },
      {
        hands: [{ shape: "flat", x: 108, y: 266, rot: 8, scale: 1.15 }],
        instruction: "Move the hand forward and down toward the person, as if giving thanks to them.",
      },
    ],
  },
  {
    id: "please",
    word: "Please",
    meaning: "Please — a polite request.",
    useWhen: "When asking for something, to be polite.",
    categories: ["courtesy"],
    frames: [
      {
        hands: [ringPose("flat", -90)],
        instruction: "Place a flat hand on your chest.",
        arrow: PLEASE_RING,
      },
      {
        hands: [ringPose("flat", 30)],
        instruction: "Rub it in a slow circle, clockwise.",
        arrow: PLEASE_RING,
      },
      {
        hands: [ringPose("flat", 150)],
        instruction: "Keep circling — one or two full rounds is enough.",
        arrow: PLEASE_RING,
      },
    ],
  },
  {
    id: "sorry",
    word: "Sorry",
    meaning: "Sorry — an apology.",
    useWhen: "When you make a mistake, bump into someone, or want to apologise.",
    categories: ["courtesy"],
    frames: [
      {
        hands: [ringPose("fistA", -90)],
        instruction: "Make a fist (thumb alongside) and place it on your chest.",
        arrow: PLEASE_RING,
      },
      {
        hands: [ringPose("fistA", 30)],
        instruction: "Rub the fist in a slow circle on your chest.",
        arrow: PLEASE_RING,
      },
      {
        hands: [ringPose("fistA", 150)],
        instruction: "Keep circling with a regretful face.",
        arrow: PLEASE_RING,
      },
    ],
  },
  {
    id: "yes",
    word: "Yes",
    meaning: "Yes — agreement or confirmation.",
    useWhen: "To agree, confirm, or say \"that's right\".",
    categories: ["replies"],
    frames: [
      {
        hands: [{ shape: "fistS", x: 112, y: 236, rot: 0 }],
        instruction: "Make a fist and hold it in front of you.",
        arrow: [[146, 214], [160, 236], [146, 260]],
      },
      {
        hands: [{ shape: "fistS", x: 112, y: 244, rot: 36 }],
        instruction: "Bend your wrist forward and down, like a head nodding.",
      },
      {
        hands: [{ shape: "fistS", x: 112, y: 236, rot: 0 }],
        instruction: "Bring it back up. Repeat once or twice.",
      },
    ],
  },
  {
    id: "no",
    word: "No",
    meaning: "No — refusal or disagreement.",
    useWhen: "To decline, disagree, or say something isn't so.",
    categories: ["replies"],
    frames: [
      {
        hands: [{ shape: "noOpen", x: 114, y: 234, rot: 4 }],
        instruction: "Hold your index and middle fingers out together, with your thumb apart from them.",
      },
      {
        hands: [{ shape: "noClosed", x: 114, y: 234, rot: 4 }],
        instruction: "Snap the two fingers down to meet your thumb — a quick, small pinch.",
      },
      {
        hands: [{ shape: "noOpen", x: 114, y: 234, rot: 4 }],
        instruction: "Open and snap closed once more, shaking your head slightly.",
      },
    ],
  },
  {
    id: "understand",
    word: "Understand",
    meaning: "I understand / I get it.",
    useWhen: "To show you've understood, or to ask someone if they have.",
    categories: ["replies"],
    frames: [
      {
        hands: [{ shape: "fistS", x: 82, y: 92, rot: 0 }],
        instruction: "Hold a loose fist near your temple.",
        arrow: [[100, 76], [100, 50], [100, 24]],
      },
      {
        hands: [{ shape: "point", x: 82, y: 90, rot: 0 }],
        instruction: "Flick your index finger up — like a light bulb coming on in your mind.",
      },
    ],
  },
  {
    id: "water",
    word: "Water",
    meaning: "Water — I need a drink.",
    useWhen: "When you're thirsty or asking for a drink of water.",
    categories: ["needs"],
    frames: [
      {
        hands: [{ shape: "wHand", x: 132, y: 200, rot: 18 }],
        instruction: "Make a \"W\" with three fingers spread (index, middle, ring) and touch your chin.",
        arrow: [[158, 178], [164, 198], [158, 178]],
      },
      {
        hands: [{ shape: "wHand", x: 122, y: 222, rot: 18 }],
        instruction: "Lift the hand slightly away from your chin.",
      },
      {
        hands: [{ shape: "wHand", x: 132, y: 200, rot: 18 }],
        instruction: "Tap your chin again. Two taps total.",
      },
    ],
  },
  {
    id: "eat",
    word: "Eat / Food",
    meaning: "Eat or food — I'm hungry or it's time to eat.",
    useWhen: "To say you're hungry, ask if someone wants to eat, or talk about food.",
    categories: ["needs"],
    frames: [
      {
        hands: [{ shape: "flatO", x: 140, y: 160, rot: 20 }],
        instruction: "Bunch your fingertips together with your thumb, as if holding a bite of food, at your mouth.",
        arrow: [[168, 140], [172, 164], [168, 140]],
      },
      {
        hands: [{ shape: "flatO", x: 126, y: 194, rot: 20 }],
        instruction: "Move the hand away from your mouth.",
      },
      {
        hands: [{ shape: "flatO", x: 140, y: 160, rot: 20 }],
        instruction: "Tap your fingertips to your lips again. Repeat once or twice.",
      },
    ],
  },
  {
    id: "help",
    word: "Help",
    meaning: "Help — I need help, or can you help me?",
    useWhen: "When you need assistance, or to offer help to someone.",
    categories: ["needs", "emergency"],
    frames: [
      {
        hands: [
          { shape: "fistA", x: 150, y: 232, rot: 0 },
          { shape: "flat", x: 150, y: 266, rot: -84, left: true },
        ],
        instruction: "Rest your fist (thumb up) on your other flat hand, palm facing up.",
        arrow: [[238, 276], [238, 246], [238, 216]],
      },
      {
        hands: [
          { shape: "fistA", x: 150, y: 196, rot: 0 },
          { shape: "flat", x: 150, y: 230, rot: -84, left: true },
        ],
        instruction: "Lift both hands together, as if raising someone up.",
      },
    ],
  },
  {
    id: "pain",
    word: "Pain / Hurt",
    meaning: "Pain — something hurts.",
    useWhen: "To tell a doctor, nurse, or friend that you're hurting. Point toward the sore area.",
    categories: ["emergency"],
    frames: [
      {
        hands: [
          { shape: "point", x: 106, y: 240, rot: 64 },
          { shape: "point", x: 194, y: 240, rot: -64, left: true },
        ],
        instruction: "Point both index fingers toward each other in front of you.",
        arrow: [[96, 296], [150, 308], [204, 296]],
      },
      {
        hands: [
          { shape: "point", x: 126, y: 246, rot: 100 },
          { shape: "point", x: 174, y: 246, rot: -100, left: true },
        ],
        instruction: "Twist your hands toward each other, jabbing slightly. Do it near where it hurts.",
      },
    ],
  },
  {
    id: "stop",
    word: "Stop",
    meaning: "Stop — halt what you're doing.",
    useWhen: "To ask someone to stop, wait, or not continue.",
    categories: ["emergency"],
    frames: [
      {
        hands: [
          { shape: "flat", x: 150, y: 184, rot: 0 },
          { shape: "flat", x: 150, y: 272, rot: -84, left: true },
        ],
        instruction: "Hold your other hand flat, palm up. Raise your dominant hand above it, fingers straight.",
        arrow: [[200, 150], [200, 184], [200, 226]],
      },
      {
        hands: [
          { shape: "flat", x: 150, y: 234, rot: 0 },
          { shape: "flat", x: 150, y: 272, rot: -84, left: true },
        ],
        instruction: "Chop the edge of your dominant hand down firmly onto the palm.",
      },
    ],
  },
  {
    id: "iloveyou",
    word: "I love you",
    meaning: "I love you.",
    useWhen: "To show love or deep appreciation to family and close friends.",
    categories: ["greetings"],
    frames: [
      {
        hands: [{ shape: "iLoveYou", x: 112, y: 236, rot: 0 }],
        instruction:
          "Raise your thumb, index finger and little finger. Fold your middle and ring fingers down.",
      },
      {
        hands: [{ shape: "iLoveYou", x: 112, y: 214, rot: 0, scale: 1.1 }],
        instruction: "Lift the hand with the palm facing the person. It blends the letters I, L and Y.",
      },
    ],
  },
];

export function signsForCategory(category: SignCategory | "all", signs: SignEntry[] = ASL_SIGNS): SignEntry[] {
  if (category === "all") return signs;
  return signs.filter((s) => s.categories.includes(category));
}
