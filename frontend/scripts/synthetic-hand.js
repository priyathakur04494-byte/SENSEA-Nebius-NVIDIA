/* Forward-kinematics synthetic hands for tests (shared by test-classifier.js and the browser e2e). */
const PALM = 0.09;
const MCP = [[-0.026, 0.085], [0, 0.09], [0.022, 0.085], [0.043, 0.075]];
const SEG = [[0.04, 0.025, 0.022], [0.045, 0.028, 0.024], [0.042, 0.026, 0.023], [0.034, 0.02, 0.019]];
const CURL = { up: [0, 0, 0], bent: [35, 45, 30], down: [85, 90, 60] };

function finger(i, state, spreadDeg) {
  const [x0, y0] = MCP[i];
  const c = CURL[state];
  let x = x0, y = y0, z = 0, cum = 0;
  const pts = [[x, y, z]];
  const sp = ((spreadDeg || 0) * Math.PI) / 180;
  for (let j = 0; j < 3; j++) {
    cum += (c[j] * Math.PI) / 180;
    x += SEG[i][j] * Math.sin(sp) * Math.cos(cum);
    y += SEG[i][j] * Math.cos(sp) * Math.cos(cum);
    z += SEG[i][j] * Math.sin(cum);
    pts.push([x, y, z]);
  }
  return pts; // MCP, PIP, DIP, TIP
}

/**
 * spec: {fingers:[state x4], spread, thumb: 'out'|'up'|'in'|'side'|{to:'idx'|'mid'|'ring'|'pinky'}}
 * Returns {world, image} with the whole hand rotated by rotDeg in the image plane.
 */
function hand(spec) {
  const spread = spec.spread || 0;
  const lm = new Array(21);
  lm[0] = [0, 0, 0];
  const names = ["idx", "mid", "ring", "pinky"];
  const tips = {};
  spec.fingers.forEach((st, i) => {
    const sd = spread * (i - 1.5) * 0.5;
    const p = finger(i, st, sd);
    for (let j = 0; j < 4; j++) lm[5 + i * 4 + j] = p[j];
    tips[names[i]] = p[3];
  });
  const cmc = [-0.02, 0.015, 0];
  lm[1] = cmc;
  const unit = (v) => { const n = Math.hypot(...v); return v.map((a) => a / n); };
  const chain = (dirs) => {
    let p = cmc; const L = [0.04, 0.032, 0.027]; const out = [];
    dirs.forEach((d, k) => { const u = unit(d); p = p.map((a, q) => a + u[q] * L[k]); out.push(p); });
    return out;
  };
  let t;
  if (typeof spec.thumb === "object" && spec.thumb.at) {
    const target = spec.thumb.at;
    t = [0.4, 0.72, 1].map((f) => cmc.map((a, q) => a + (target[q] - cmc[q]) * f));
  } else if (typeof spec.thumb === "object") {
    const target = tips[spec.thumb.to];
    const v = target.map((a, q) => a - cmc[q]); const n = Math.hypot(...v);
    t = [0.4, 0.72, 1].map((f) => cmc.map((a, q) => a + v[q] * f));
    void n;
  } else if (spec.thumb === "out") t = chain([[-0.8, 0.6, 0], [-0.8, 0.6, 0], [-0.8, 0.6, 0]]);
  else if (spec.thumb === "up") t = chain([[-0.25, 1, 0.1], [-0.1, 1, 0], [0, 1, 0]]);
  else if (spec.thumb === "down") t = chain([[-0.25, -1, 0.1], [-0.1, -1, 0], [0, -1, 0]]);
  else if (spec.thumb === "side") t = hand({ ...spec, thumb: { at: [-0.045, 0.075, 0.03] } }).world.slice(2, 5);
  else /* in: tucked across the palm */ t = hand({ ...spec, thumb: { at: [0.005, 0.07, 0.035] }, rot: 0 }).world.slice(2, 5);
  lm[2] = t[0]; lm[3] = t[1]; lm[4] = t[2];

  const rot = ((spec.rot || 0) * Math.PI) / 180;
  const c = Math.cos(rot), s = Math.sin(rot);
  const world = lm.map(([x, y, z]) => [x * c - y * s, x * s + y * c, z]);
  const image = world.map(([x, y, z]) => [0.5 + x * 3, 0.5 - y * 3, z]);
  return { world, image };
}


module.exports = { hand };
