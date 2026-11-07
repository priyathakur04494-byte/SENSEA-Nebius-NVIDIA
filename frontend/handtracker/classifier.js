/*
 * SENSEA sign classifier — plain JavaScript, no dependencies.
 *
 * This file runs inside the camera page (see page.js) and is also loaded by the
 * unit tests in scripts/test-classifier.js, so there is one copy of the logic.
 *
 * Input is the 21-point hand skeleton MediaPipe produces:
 *   0 wrist | 1-4 thumb | 5-8 index | 9-12 middle | 13-16 ring | 17-20 pinky
 * (each index is [MCP, PIP, DIP, TIP]; the thumb is [CMC, MCP, IP, TIP]).
 *
 * Three layers:
 *   1. classifyPose()      — one frame -> best built-in handshape + confidence
 *   2. MotionTracker       — wave / nod / snap signs that need movement
 *   3. CustomMatcher       — the user's own taught signs (any sign language)
 *
 * IMPORTANT: the built-in rules are geometric heuristics for a small, common
 * vocabulary. They are NOT a trained model of a whole sign language. Anything
 * outside the vocabulary is covered by "Teach a sign" (CustomMatcher).
 */
(function (root) {
  "use strict";

  // ------------------------------------------------------------------ helpers
  function dist(a, b) {
    var dx = a[0] - b[0], dy = a[1] - b[1], dz = (a[2] || 0) - (b[2] || 0);
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
  }
  function clamp01(x) { return x < 0 ? 0 : x > 1 ? 1 : x; }
  function mean(a) { var s = 0; for (var i = 0; i < a.length; i++) s += a[i]; return a.length ? s / a.length : 0; }
  function minOf(a) { var m = 1; for (var i = 0; i < a.length; i++) if (a[i] < m) m = a[i]; return m; }

  // Fingers: [MCP, PIP, DIP, TIP] landmark indices.
  var FINGERS = [[5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]];

  /** 1.0 = perfectly straight finger, ~0.45 = tightly curled. Rotation invariant. */
  function straightness(p, ids) {
    var a = p[ids[0]], b = p[ids[1]], c = p[ids[2]], d = p[ids[3]];
    var chain = dist(a, b) + dist(b, c) + dist(c, d);
    return chain > 1e-9 ? dist(a, d) / chain : 0;
  }

  // Soft membership scores (0..1) so every rule yields a confidence, not a boolean.
  function up(r) { return clamp01((r - 0.80) / 0.10); }       // straight
  function down(r) { return clamp01((0.68 - r) / 0.12); }     // curled into palm
  function bent(r) { return clamp01(Math.min((r - 0.55) / 0.10, (0.90 - r) / 0.08)); } // hooked / curved

  /**
   * Pull out every measurement the rules need, once per frame.
   * `w` = 3D landmarks (metres, hand-centred; best for angles/distances).
   * `img` = normalised image landmarks (x right, y DOWN) — only used for
   * "which way is up" signs such as thumbs-up.
   */
  function features(w, img) {
    var palm = dist(w[0], w[9]) || 1e-6;
    var r = FINGERS.map(function (f) { return straightness(w, f); });
    var rt = straightness(w, [1, 2, 3, 4]);
    var f = {
      palm: palm,
      r: r,
      up: r.map(up), down: r.map(down), bent: r.map(bent),
      rt: rt,
      // Thumb is "out" when it is straight AND sits away from the palm — either far
      // from the index knuckle (L / Y shapes) or far from the wrist (thumbs up).
      thumbOut: clamp01((rt - 0.78) / 0.10) * Math.max(
        clamp01((dist(w[4], w[5]) / palm - 0.62) / 0.25),
        clamp01((dist(w[4], w[0]) / palm - 1.0) / 0.25)),
      tIdx: dist(w[4], w[8]) / palm,
      tMid: dist(w[4], w[12]) / palm,
      tRing: dist(w[4], w[16]) / palm,
      tPinky: dist(w[4], w[20]) / palm,
      idxMid: dist(w[8], w[12]) / palm,
      midRing: dist(w[12], w[16]) / palm,
    };
    // Thumb direction in the image (for thumbs up / down). Needs the image coords.
    f.thumbDy = 0;
    if (img) {
      var dx = img[4][0] - img[2][0], dy = img[4][1] - img[2][1];
      var len = Math.sqrt(dx * dx + dy * dy) || 1e-6;
      f.thumbDy = dy / len; // -1 = thumb pointing up on screen, +1 = down
    }
    return f;
  }

  // ------------------------------------------------------- built-in vocabulary
  // Each rule returns a 0..1 score. `meaning` is what we tell the user.
  // `needsMotion` marks signs produced by MotionTracker instead of this table.
  var SIGNS = {
    iloveyou: { label: "I love you", note: "Thumb, index and little finger out" },
    callme: { label: "Call me", note: "Thumb and little finger out (letter Y)" },
    thumbsup: { label: "Good / Yes", note: "Thumbs up" },
    thumbsdown: { label: "Bad / No", note: "Thumbs down" },
    ok: { label: "OK", note: "Thumb and index circle, others up (letter F)" },
    eat: { label: "Eat / Food", note: "Fingertips gathered on the thumb" },
    water: { label: "Water", note: "Three fingers up, thumb on little finger (letter W)" },
    peace: { label: "Peace / Two", note: "Index and middle apart (letter V)" },
    letterU: { label: "Two / U", note: "Index and middle together" },
    letterL: { label: "L", note: "Index up, thumb out" },
    one: { label: "One / Point", note: "Index finger up" },
    three: { label: "Three", note: "Three fingers up" },
    four: { label: "Four", note: "Four fingers up, thumb tucked" },
    open: { label: "Stop / Open hand", note: "Open palm (also five or letter B)" },
    letterC: { label: "C", note: "Curved hand" },
    fist: { label: "Fist", note: "Closed fist (letters S / A)" },
    // motion signs
    hello: { label: "Hello", note: "Open hand waving side to side", needsMotion: true },
    yes: { label: "Yes", note: "Fist nodding up and down", needsMotion: true },
    no: { label: "No", note: "Index and middle fingers snapping to the thumb", needsMotion: true },
  };

  function scoreAll(f) {
    var u = f.up, d = f.down, b = f.bent;
    var s = {};
    var othersDown = function (i) { return mean(i.map(function (k) { return d[k]; })); };

    s.iloveyou = minOf([u[0], d[1], d[2], u[3], f.thumbOut]);
    s.callme = minOf([d[0], d[1], d[2], u[3], f.thumbOut]);

    // Thumbs up / down: fist-like fingers, thumb straight and vertical on screen.
    var fistish = minOf([d[0], d[1], d[2], d[3]]);
    s.thumbsup = minOf([fistish, f.thumbOut, clamp01((-f.thumbDy - 0.45) / 0.3)]);
    s.thumbsdown = minOf([fistish, f.thumbOut, clamp01((f.thumbDy - 0.45) / 0.3)]);

    // OK / F: thumb tip meets index tip, other three straight.
    s.ok = minOf([clamp01((0.42 - f.tIdx) / 0.15), u[1], u[2], u[3]]);

    // Eat / flat-O: every fingertip gathers near the thumb tip.
    s.eat = minOf([
      clamp01((0.55 - f.tIdx) / 0.2), clamp01((0.65 - f.tMid) / 0.2), clamp01((0.85 - f.tRing) / 0.25),
      1 - u[0], 1 - u[1],
    ]);

    // Water / W: index+middle+ring up, little finger held down by the thumb.
    s.water = minOf([u[0], u[1], u[2], d[3], clamp01((0.75 - f.tPinky) / 0.25)]);

    // V vs U
    var twoUp = minOf([u[0], u[1], d[2], d[3]]);
    s.peace = Math.min(twoUp, clamp01((f.idxMid - 0.38) / 0.15));
    s.letterU = Math.min(twoUp, clamp01((0.40 - f.idxMid) / 0.12), 1 - f.thumbOut * 0.5);

    // L / one
    s.letterL = minOf([u[0], d[1], d[2], d[3], f.thumbOut]);
    s.one = minOf([u[0], d[1], d[2], d[3], 1 - f.thumbOut]);

    s.three = minOf([u[0], u[1], u[2], d[3], clamp01((f.tPinky - 0.75) / 0.25)]);

    var fourUp = minOf([u[0], u[1], u[2], u[3]]);
    // Four = thumb clearly tucked across the palm; otherwise it is an open hand.
    s.four = Math.min(fourUp, 1 - f.thumbOut);
    s.open = fourUp * (f.thumbOut > 0.5 ? 1 : 0.85);

    // C: curved fingers with a gap between thumb and index.
    s.letterC = minOf([b[0], b[1], b[2], b[3],
      clamp01((f.tIdx - 0.55) / 0.2), clamp01((1.5 - f.tIdx) / 0.3)]);

    s.fist = Math.min(fistish, 1 - f.thumbOut);
    return s;
  }

  /** Classify one hand. Returns { id, label, note, confidence, scores } or null. */
  function classifyPose(world, image) {
    if (!world || world.length < 21) return null;
    var f = features(world, image);
    var scores = scoreAll(f);
    var bestId = null, best = 0;
    // Evaluation order = tie-break priority (specific shapes before generic ones).
    var order = ["iloveyou", "callme", "thumbsup", "thumbsdown", "ok", "water", "eat",
      "peace", "letterU", "letterL", "one", "three", "four", "letterC", "open", "fist"];
    for (var i = 0; i < order.length; i++) {
      var sc = scores[order[i]];
      if (sc > best + 0.02) { best = sc; bestId = order[i]; }
    }
    if (!bestId || best < 0.45) return { id: null, label: null, note: "", confidence: 0, scores: scores, features: f };
    return {
      id: bestId, label: SIGNS[bestId].label, note: SIGNS[bestId].note,
      confidence: Math.min(1, 0.5 + 0.5 * best),
      scores: scores, features: f,
    };
  }

  // ------------------------------------------------------------ motion signs
  /**
   * Tracks one hand over time. Feed it every frame with `push()`; it returns a
   * motion sign id ("hello" | "yes" | "no") when it sees one, else null.
   */
  function MotionTracker() {
    this.frames = []; // {t, x, y, palmImg, pose, noOpen, noClosed}
  }
  MotionTracker.prototype.reset = function () { this.frames.length = 0; };

  function reversals(values, minAmp) {
    // Count direction changes whose swing exceeds minAmp (hysteresis, so jitter is ignored).
    var count = 0, dir = 0, i;
    var anchor = values[0];
    for (i = 1; i < values.length; i++) {
      var d = values[i] - anchor;
      if (dir <= 0 && d > minAmp) { if (dir === -1) count++; dir = 1; anchor = values[i]; }
      else if (dir >= 0 && d < -minAmp) { if (dir === 1) count++; dir = -1; anchor = values[i]; }
      else if (dir === 1 && values[i] > anchor) anchor = values[i];
      else if (dir === -1 && values[i] < anchor) anchor = values[i];
    }
    return count;
  }

  MotionTracker.prototype.push = function (tMs, world, image, pose) {
    var f = pose && pose.features ? pose.features : features(world, image);
    var palmImg = dist(image[0], image[9]) || 1e-6; // palm size in image units
    var openHand = mean([f.up[0], f.up[1], f.up[2], f.up[3]]);
    var fist = mean([f.down[0], f.down[1], f.down[2], f.down[3]]);
    // "No" shape: index + middle extended-ish toward the thumb, ring/pinky curled.
    var noShape = Math.min(1 - f.down[0], 1 - f.down[1], f.down[2], f.down[3]);
    var frame = {
      t: tMs, x: image[0][0], y: image[0][1], palmImg: palmImg,
      open: openHand, fist: fist,
      noOpen: noShape > 0.6 && f.tIdx > 0.85 && f.tMid > 0.85,
      noClosed: noShape > 0.5 && f.tIdx < 0.5 && f.tMid < 0.6,
    };
    this.frames.push(frame);
    while (this.frames.length && tMs - this.frames[0].t > 1800) this.frames.shift();
    if (this.frames.length < 8) return null;

    var win = this.frames.filter(function (q) { return tMs - q.t <= 1400; });
    if (win.length < 8) return null;
    var palm = mean(win.map(function (q) { return q.palmImg; }));
    var avgOpen = mean(win.map(function (q) { return q.open; }));
    var avgFist = mean(win.map(function (q) { return q.fist; }));

    // HELLO: open hand, side-to-side swing of >= ~0.5 palm width, >= 3 reversals.
    if (avgOpen > 0.6) {
      var xs = win.map(function (q) { return q.x; });
      var rev = reversals(xs, palm * 0.35);
      if (rev >= 3) return { id: "hello", confidence: Math.min(1, 0.65 + rev * 0.08) };
    }
    // YES: fist, up/down nodding of >= ~0.3 palm, >= 2 reversals.
    if (avgFist > 0.6) {
      var ys = win.map(function (q) { return q.y; });
      var xsF = win.map(function (q) { return q.x; });
      var revY = reversals(ys, palm * 0.28);
      var revX = reversals(xsF, palm * 0.28);
      if (revY >= 2 && revY > revX) return { id: "yes", confidence: Math.min(1, 0.6 + revY * 0.08) };
    }
    // NO: shape alternates open <-> closed against the thumb at least twice.
    var flips = 0, last = null;
    win.forEach(function (q) {
      var st = q.noClosed ? "c" : q.noOpen ? "o" : null;
      if (st && last && st !== last) flips++;
      if (st) last = st;
    });
    if (flips >= 2) return { id: "no", confidence: Math.min(1, 0.6 + flips * 0.1) };
    return null;
  };

  // ------------------------------------------------------ taught (custom) signs
  var RESAMPLE_N = 12;
  var WINDOWS_MS = [900, 1300, 1700];

  /** One frame -> flat vector, wrist-relative and palm-normalised. */
  function frameVector(hands) {
    // hands: [{world, image}] ; up to two, ordered left-to-right on screen
    var out = [];
    for (var h = 0; h < 2; h++) {
      var hand = hands[h];
      if (!hand) { for (var z = 0; z < 63 + 2; z++) out.push(0); continue; }
      var w = hand.world, palm = dist(w[0], w[9]) || 1e-6;
      for (var i = 1; i < 21; i++) {
        out.push((w[i][0] - w[0][0]) / palm, (w[i][1] - w[0][1]) / palm, (w[i][2] - w[0][2]) / palm);
      }
      out.push(0, 0, 0); // pad to 63 (21 landmarks incl. wrist-origin)
      out.push(hand.image[0][0], hand.image[0][1]); // wrist position, made relative later
    }
    return out;
  }

  function resample(frames, n) {
    // frames: [{t, v}] sorted by t. linear interpolate to n evenly spaced vectors.
    var t0 = frames[0].t, t1 = frames[frames.length - 1].t, span = Math.max(t1 - t0, 1);
    var out = [], j = 0;
    for (var k = 0; k < n; k++) {
      var t = t0 + span * k / (n - 1);
      while (j < frames.length - 2 && frames[j + 1].t < t) j++;
      var a = frames[j], b = frames[Math.min(j + 1, frames.length - 1)];
      var u = b.t === a.t ? 0 : clamp01((t - a.t) / (b.t - a.t));
      var v = new Array(a.v.length);
      for (var i = 0; i < v.length; i++) v[i] = a.v[i] + (b.v[i] - a.v[i]) * u;
      out.push(v);
    }
    return out;
  }

  /** Make a window of frames into a fixed-size, position-independent descriptor. */
  function describe(frames) {
    if (frames.length < 4) return null;
    var rs = resample(frames, RESAMPLE_N);
    var len = rs[0].length, half = len / 2;
    // Wrist position (the last 2 of each hand block) -> relative to first frame,
    // in units of a typical palm width so movement size matters, not screen spot.
    var scale = 0.12; // ~ palm width in normalised image coords at arm's length
    var flat = [];
    for (var k = 0; k < rs.length; k++) {
      var v = rs[k].slice();
      for (var h = 0; h < 2; h++) {
        var o = h * half + 63;
        v[o] = (v[o] - rs[0][o]) / scale;
        v[o + 1] = (v[o + 1] - rs[0][o + 1]) / scale;
      }
      for (var i = 0; i < v.length; i++) flat.push(v[i]);
    }
    return flat;
  }

  function descDist(a, b) {
    // RMS over entries where either side has data, so the empty "second hand"
    // padding of one-handed signs does not dilute the difference.
    var s = 0, c = 0, n = Math.min(a.length, b.length);
    for (var i = 0; i < n; i++) {
      if (a[i] === 0 && b[i] === 0) continue;
      var d = a[i] - b[i]; s += d * d; c++;
    }
    return c ? Math.sqrt(s / c) : Infinity;
  }

  function CustomMatcher(opts) {
    opts = opts || {};
    this.maxDist = opts.maxDist || 0.30;  // ceiling for a match (RMS); each sign may be stricter
    this.margin = opts.margin || 0.06;    // and clearly nearer than the runner-up
    this.signs = [];                      // [{id, label, templates: [descriptor]}]
    this.frames = [];                     // rolling {t, v}
  }
  CustomMatcher.prototype.setSigns = function (list) { this.signs = list || []; };
  CustomMatcher.prototype.reset = function () { this.frames.length = 0; };
  CustomMatcher.prototype.push = function (tMs, hands) {
    if (!hands || !hands.length) { this.frames.length = 0; return null; }
    this.frames.push({ t: tMs, v: frameVector(hands) });
    while (this.frames.length && tMs - this.frames[0].t > 2000) this.frames.shift();
    return this.match(tMs);
  };
  CustomMatcher.prototype.match = function (tMs) {
    if (!this.signs.length || this.frames.length < 6) return null;
    var perSign = {};
    for (var wi = 0; wi < WINDOWS_MS.length; wi++) {
      var ms = WINDOWS_MS[wi];
      var win = this.frames.filter(function (q) { return tMs - q.t <= ms; });
      if (win.length < 6 || win[win.length - 1].t - win[0].t < ms * 0.6) continue;
      var desc = describe(win);
      if (!desc) continue;
      for (var si = 0; si < this.signs.length; si++) {
        var sg = this.signs[si];
        for (var ti = 0; ti < sg.templates.length; ti++) {
          var d = descDist(desc, sg.templates[ti]);
          if (perSign[sg.id] === undefined || d < perSign[sg.id].d) perSign[sg.id] = { d: d, sign: sg };
        }
      }
    }
    var ids = Object.keys(perSign).sort(function (a, b) { return perSign[a].d - perSign[b].d; });
    if (!ids.length) return null;
    var best = perSign[ids[0]], second = ids[1] ? perSign[ids[1]].d : Infinity;
    var limit = Math.min(this.maxDist, best.sign.threshold || this.maxDist);
    if (best.d > limit || second - best.d < this.margin) return null;
    return {
      id: best.sign.id, label: best.sign.label, custom: true,
      confidence: Math.max(0.5, Math.min(1, 1 - best.d / (limit * 1.6))),
      distance: best.d,
    };
  };

  /**
   * How forgiving a taught sign should be: the more the user's own repeats
   * differ from each other, the wider the tolerance (clamped 0.18 .. 0.30).
   */
  function makeThreshold(templates) {
    if (templates.length < 2) return 0.24;
    var ds = [];
    for (var i = 0; i < templates.length; i++)
      for (var j = i + 1; j < templates.length; j++) ds.push(descDist(templates[i], templates[j]));
    return Math.max(0.18, Math.min(0.30, mean(ds) * 2.5 + 0.12));
  }

  /** Turn a recorded stretch of frames into a template descriptor. */
  function makeTemplate(frames) {
    return describe(frames.map(function (q) { return { t: q.t, v: q.v }; }));
  }

  // ------------------------------------------------- stabilising the stream
  /**
   * Turns flickery per-frame guesses into "confirmed" signs. A sign is
   * confirmed once it has been the answer for `holdMs`; it will not repeat
   * until the hand leaves, the sign changes, or `repeatMs` passes.
   */
  function Stabilizer(opts) {
    opts = opts || {};
    this.holdMs = opts.holdMs || 650;
    this.repeatMs = opts.repeatMs || 6000; // holding one sign does not repeat it every moment
    this.graceMs = opts.graceMs || 250;
    this.cand = null; this.since = 0; this.lastSeen = 0;
    this.confirmedId = null; this.confirmedAt = 0;
  }
  Stabilizer.prototype.update = function (tMs, det) {
    // det: {id,label,confidence,...} | null. Returns {current, confirmed}
    if (det && det.id) {
      if (this.cand !== det.id) { this.cand = det.id; this.since = tMs; }
      this.lastSeen = tMs;
    } else if (tMs - this.lastSeen > this.graceMs) {
      this.cand = null;
      this.confirmedId = null;
    }
    var confirmed = null;
    if (det && det.id && this.cand === det.id) {
      var instant = det.motion || det.custom; // motion/custom matches already span a time window
      var held = tMs - this.since >= (instant ? 0 : this.holdMs);
      var fresh = this.confirmedId !== det.id || tMs - this.confirmedAt > this.repeatMs;
      if (held && fresh) { this.confirmedId = det.id; this.confirmedAt = tMs; confirmed = det; }
    }
    return { current: det && det.id ? det : null, confirmed: confirmed };
  };

  var api = {
    SIGNS: SIGNS, classifyPose: classifyPose, features: features, MotionTracker: MotionTracker,
    CustomMatcher: CustomMatcher, Stabilizer: Stabilizer, frameVector: frameVector,
    makeTemplate: makeTemplate, makeThreshold: makeThreshold, WINDOWS_MS: WINDOWS_MS,
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  root.SenseaSigns = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
