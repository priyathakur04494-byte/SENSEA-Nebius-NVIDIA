/**
 * The built-in vocabulary of the camera recogniser (handtracker/classifier.js).
 * This list is only for showing the user what to try; the rules live in the
 * classifier. Keep the two in step when you add a sign.
 */
export interface RecognizableSign {
  label: string;
  how: string;
  moving?: boolean;
}

export const RECOGNIZABLE_SIGNS: RecognizableSign[] = [
  { label: "Hello", how: "Wave an open hand side to side", moving: true },
  { label: "Yes", how: "Fist nodding up and down", moving: true },
  { label: "No", how: "Index + middle finger tap the thumb", moving: true },
  { label: "I love you", how: "Thumb, index and little finger out" },
  { label: "Good / Yes", how: "Thumbs up" },
  { label: "Bad / No", how: "Thumbs down" },
  { label: "OK", how: "Thumb and index circle, other fingers up" },
  { label: "Stop / Open hand", how: "Open palm toward the camera" },
  { label: "Water", how: "Three fingers up (W), thumb on little finger" },
  { label: "Eat / Food", how: "Fingertips gathered together on the thumb" },
  { label: "Call me", how: "Thumb and little finger out (Y)" },
  { label: "Peace / Two", how: "Index and middle fingers apart (V)" },
  { label: "One / Point", how: "Index finger up" },
  { label: "L", how: "Index up, thumb out" },
  { label: "Three", how: "Three fingers up" },
  { label: "Four", how: "Four fingers up, thumb tucked" },
  { label: "C", how: "Curved hand like holding a cup" },
  { label: "Fist (S / A)", how: "Closed fist" },
];


/**
 * 20 Indian Sign Language (ISL) gestures recognized by the SENSEA backend
 * PyTorch Deep Learning Model (Conv1D + BiGRU + Attention).
 */
export const ISL_ML_SIGNS: string[] = [
  "hello",
  "goodbye",
  "thank_you",
  "sorry",
  "please",
  "yes",
  "no",
  "help",
  "stop",
  "okay",
  "me",
  "you",
  "mother",
  "father",
  "friend",
  "home",
  "hospital",
  "eat",
  "drink",
  "water",
];
