/*
 * Unit tests for handtracker/classifier.js using SYNTHETIC hands (forward
 * kinematics). They prove the rules are self-consistent and rotation-invariant;
 * they do NOT prove accuracy on real people — see README "Accuracy".
 *
 *   node scripts/test-classifier.js
 */
const path = require("path");
const S = require(path.join(__dirname, "..", "handtracker", "classifier.js"));

const { hand } = require("./synthetic-hand.js");

const U = "up", B = "bent", D = "down";
const CASES = [
  ["iloveyou", { fingers: [U, D, D, U], thumb: "out" }],
  ["callme", { fingers: [D, D, D, U], thumb: "out" }],
  ["thumbsup", { fingers: [D, D, D, D], thumb: "up" }],
  ["thumbsdown", { fingers: [D, D, D, D], thumb: "down" }],
  ["ok", { fingers: [B, U, U, U], thumb: { to: "idx" } }],
  ["water", { fingers: [U, U, U, D], thumb: { to: "pinky" }, spread: 8 }],
  ["peace", { fingers: [U, U, D, D], thumb: "in", spread: 22 }],
  ["letterU", { fingers: [U, U, D, D], thumb: "in", spread: 0 }],
  ["letterL", { fingers: [U, D, D, D], thumb: "out" }],
  ["one", { fingers: [U, D, D, D], thumb: "in" }],
  ["three", { fingers: [U, U, U, D], thumb: "out", spread: 8 }],
  ["four", { fingers: [U, U, U, U], thumb: "in" }],
  ["open", { fingers: [U, U, U, U], thumb: "out", spread: 12 }],
  ["fist", { fingers: [D, D, D, D], thumb: "in" }],
];

let fail = 0;
function check(name, ok, extra) {
  console.log((ok ? "PASS " : "FAIL ") + name + (extra ? "  " + extra : ""));
  if (!ok) fail++;
}

// 1. static poses, at several hand rotations (rotation invariance)
for (const [id, spec] of CASES) {
  for (const rot of [0, 25, -30]) {
    // thumbs up/down are defined relative to the screen, so only test upright
    if ((id === "thumbsup" || id === "thumbsdown") && rot !== 0) continue;
    const h = hand({ ...spec, rot });
    const r = S.classifyPose(h.world, h.image);
    const top = Object.entries(r.scores).sort((a, b) => b[1] - a[1]).slice(0, 3)
      .map(([k, v]) => k + ":" + v.toFixed(2)).join(" ");
    check(`${id} @${rot}°`, r.id === id, `got=${r.id} conf=${r.confidence.toFixed(2)}  [${top}]`);
  }
}

// 2. two obviously different shapes must not be confused
{
  const a = hand({ fingers: [U, U, U, U], thumb: "out" });
  const r = S.classifyPose(a.world, a.image);
  check("open hand is not 'fist'", r.id !== "fist");
}

// 3. motion: wave, nod, no-snap
function runMotion(spec, wristAt, ms = 1600, fps = 30) {
  const m = new S.MotionTracker();
  let hit = null;
  for (let k = 0; k * (1000 / fps) < ms; k++) {
    const t = k * (1000 / fps);
    const h = hand(spec(t));
    const [dx, dy] = wristAt(t);
    const image = h.image.map(([x, y, z]) => [x + dx, y + dy, z]);
    const pose = S.classifyPose(h.world, image);
    const out = m.push(t, h.world, image, pose);
    if (out && !hit) hit = out;
  }
  return hit;
}
{
  const wave = runMotion(() => ({ fingers: [U, U, U, U], thumb: "out", spread: 12 }),
    (t) => [0.12 * Math.sin((t / 1000) * 2 * Math.PI * 2.5), 0]);
  check("wave -> hello", wave && wave.id === "hello", JSON.stringify(wave));
  const still = runMotion(() => ({ fingers: [U, U, U, U], thumb: "out", spread: 12 }), () => [0, 0]);
  check("still open hand -> no motion sign", !still, JSON.stringify(still));
  const jitter = runMotion(() => ({ fingers: [U, U, U, U], thumb: "out", spread: 12 }),
    (t) => [0.004 * Math.sin(t / 20), 0.004 * Math.cos(t / 17)]);
  check("hand tremor does not trigger", !jitter, JSON.stringify(jitter));
  const nod = runMotion(() => ({ fingers: [D, D, D, D], thumb: "in" }),
    (t) => [0, 0.06 * Math.sin((t / 1000) * 2 * Math.PI * 2)]);
  check("fist nod -> yes", nod && nod.id === "yes", JSON.stringify(nod));
  const no = runMotion((t) => {
    const open = Math.sin((t / 1000) * 2 * Math.PI * 2) > 0;
    return { fingers: [open ? U : B, open ? U : B, D, D], thumb: open ? "out" : { to: "idx" } };
  }, () => [0, 0]);
  // closed state must also bring the middle tip to the thumb: handled by thumb target idx (tMid checked loosely)
  check("index+middle snap -> no (best effort)", no && no.id === "no", JSON.stringify(no));
}

// 4. custom taught signs
{
  const mk = (spec, dx = (t) => [0, 0], ms = 1300, fps = 30) => {
    const frames = [];
    for (let k = 0; k * (1000 / fps) < ms; k++) {
      const t = k * (1000 / fps);
      const h = hand(spec(t));
      const [ox, oy] = dx(t);
      frames.push({ t, hands: [{ world: h.world, image: h.image.map(([x, y, z]) => [x + ox, y + oy, z]) }] });
    }
    return frames;
  };
  const toTemplate = (frames) => S.makeTemplate(frames.map((f) => ({ t: f.t, v: S.frameVector(f.hands) })));
  // sign A: open hand sweeping right.  sign B: static L shape.
  const aSpec = () => ({ fingers: [U, U, U, U], thumb: "out", spread: 12 });
  const aMove = (t) => [(t / 1300) * 0.25, 0];
  const bSpec = () => ({ fingers: [U, D, D, D], thumb: "out" });
  const m = new S.CustomMatcher();
  m.setSigns([
    { id: "sweep", label: "Sweep", templates: [toTemplate(mk(aSpec, aMove))] },
    { id: "ell", label: "Ell", templates: [toTemplate(mk(bSpec))] },
  ]);
  const feed = (frames) => {
    m.reset(); let r = null;
    frames.forEach((f) => { const o = m.push(f.t, f.hands); if (o) r = o; });
    return r;
  };
  const rA = feed(mk(aSpec, (t) => aMove(t * 0.93))); // performed slightly faster, still same sign
  check("taught moving sign recognised", rA && rA.id === "sweep", JSON.stringify(rA));
  const rB = feed(mk(bSpec, () => [0.1, 0.05])); // different place on screen
  check("taught static sign recognised anywhere on screen", rB && rB.id === "ell", JSON.stringify(rB));
  const rC = feed(mk(() => ({ fingers: [D, D, D, D], thumb: "in" })));
  check("unrelated fist is rejected", !rC, JSON.stringify(rC));
}

// 5. stabiliser
{
  const st = new S.Stabilizer({ holdMs: 600, repeatMs: 2500 });
  let confirmed = [];
  for (let t = 0; t < 4500; t += 33) {
    const det = t < 4000 ? { id: "ok", label: "OK", confidence: 0.9 } : null;
    const r = st.update(t, det);
    if (r.confirmed) confirmed.push(Math.round(t));
  }
  check("hold 600ms then confirm; repeat after 2.5s", confirmed.length === 2 && confirmed[0] >= 600 && confirmed[0] < 700, confirmed.join(","));
}

console.log(fail ? `\n${fail} FAILED` : "\nall passed");
process.exit(fail ? 1 : 0);
