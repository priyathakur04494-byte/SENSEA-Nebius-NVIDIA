/*
 * SENSEA camera page. Runs inside an <iframe srcdoc> on web and a WebView on
 * phones. It owns the camera, the MediaPipe hand model, the skeleton overlay
 * and the recogniser; it talks to the React Native app only via messages.
 *
 *  page -> app   {type:"status", state, message}
 *                {type:"frame", hands:[{label,confidence}], current:{...}|null}
 *                {type:"confirmed", sign:{id,label,note,confidence,custom,motion}}
 *                {type:"teach-progress", phase, take, takes, count?}
 *                {type:"taught", sign:{id,label,templates,threshold}} | {type:"teach-failed", reason}
 *  app -> page   {type:"init", customSigns, facing}
 *                {type:"teach", id, label}
 *                {type:"cancel-teach"}
 *                {type:"set-custom", customSigns}
 *                {type:"camera", facing}
 */
(function () {
  "use strict";
  var cfg = window.__SENSEA_CONFIG__;
  var S = window.SenseaSigns;

  var els = {
    stage: document.getElementById("stage"),
    cam: document.getElementById("cam"),
    video: document.getElementById("video"),
    canvas: document.getElementById("overlay"),
    chip: document.getElementById("chip"),
    msg: document.getElementById("msg"),
    big: document.getElementById("big"),
  };
  var ctx = els.canvas.getContext("2d");

  function post(m) {
    try {
      if (window.ReactNativeWebView) window.ReactNativeWebView.postMessage(JSON.stringify(m));
      else if (window.parent && window.parent !== window) window.parent.postMessage(m, "*");
    } catch (e) { /* host gone */ }
  }
  function status(state, message) {
    els.msg.textContent = message || "";
    els.msg.style.display = message ? "block" : "none";
    post({ type: "status", state: state, message: message || "" });
  }

  var landmarker = null;
  var facing = "user";
  var stream = null;
  var running = false;
  var lastVideoTime = -1;
  var motion = [new S.MotionTracker(), new S.MotionTracker()];
  var custom = new S.CustomMatcher();
  var stab = new S.Stabilizer();
  var teach = null; // active teaching session
  var lastPost = 0, lastFrameSig = "";

  // ---------------------------------------------------------------- startup
  async function loadModel() {
    status("loading", "Loading hand model…");
    var mod = await import(cfg.bundleUrl);
    var fileset = await mod.FilesetResolver.forVisionTasks(cfg.wasmUrl);
    var opts = function (delegate) {
      return {
        baseOptions: { modelAssetPath: cfg.modelUrl, delegate: delegate },
        runningMode: "VIDEO", numHands: 2,
        minHandDetectionConfidence: 0.6, minHandPresenceConfidence: 0.6, minTrackingConfidence: 0.5,
      };
    };
    try { landmarker = await mod.HandLandmarker.createFromOptions(fileset, opts("GPU")); }
    catch (e) { landmarker = await mod.HandLandmarker.createFromOptions(fileset, opts("CPU")); }
  }

  async function startCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      status("error", "Camera not available here. Browsers only allow the camera on https:// or localhost.");
      return false;
    }
    stopCamera();
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { facingMode: facing, width: { ideal: 640 }, height: { ideal: 480 } },
      });
    } catch (e) {
      var denied = e && (e.name === "NotAllowedError" || e.name === "SecurityError");
      status(denied ? "denied" : "error",
        denied ? "Camera permission was blocked. Allow camera access and try again."
          : "Could not open the camera (" + (e && e.name) + ").");
      return false;
    }
    els.video.srcObject = stream;
    els.cam.classList.toggle("mirror", facing === "user");
    await els.video.play();
    await new Promise(function (r) {
      if (els.video.videoWidth) return r();
      els.video.onloadedmetadata = function () { r(); };
    });
    els.canvas.width = els.video.videoWidth; els.canvas.height = els.video.videoHeight;
    fit();
    return true;
  }
  function stopCamera() {
    if (stream) stream.getTracks().forEach(function (t) { t.stop(); });
    stream = null;
  }

  function fit() {
    var vw = els.video.videoWidth || 640, vh = els.video.videoHeight || 480;
    var cw = els.stage.clientWidth, ch = els.stage.clientHeight;
    var k = Math.min(cw / vw, ch / vh);
    els.cam.style.width = Math.floor(vw * k) + "px";
    els.cam.style.height = Math.floor(vh * k) + "px";
  }
  window.addEventListener("resize", fit);

  // ----------------------------------------------------------------- drawing
  var BONES = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12],
    [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17]];
  var HAND_COLORS = ["#7FE0D2", "#FFD479"];

  function draw(hands) {
    var w = els.canvas.width, h = els.canvas.height;
    ctx.clearRect(0, 0, w, h);
    hands.forEach(function (hd, i) {
      var p = hd.image, col = HAND_COLORS[i % 2];
      ctx.lineWidth = Math.max(2, w / 220); ctx.strokeStyle = col; ctx.lineCap = "round";
      BONES.forEach(function (b) {
        ctx.beginPath(); ctx.moveTo(p[b[0]][0] * w, p[b[0]][1] * h); ctx.lineTo(p[b[1]][0] * w, p[b[1]][1] * h); ctx.stroke();
      });
      p.forEach(function (pt, k) {
        ctx.beginPath(); ctx.fillStyle = k % 4 === 0 && k > 0 ? "#fff" : col;
        ctx.arc(pt[0] * w, pt[1] * h, Math.max(2.5, w / (k % 4 === 0 && k > 0 ? 120 : 190)), 0, 6.283); ctx.fill();
      });
    });
  }

  function placeChip(text, hands) {
    if (!text || !hands.length) { els.chip.style.display = "none"; return; }
    var best = hands[0].image[0];
    var xs = hands[0].image.map(function (q) { return q[0]; }), ys = hands[0].image.map(function (q) { return q[1]; });
    var top = Math.min.apply(null, ys), cx = (Math.min.apply(null, xs) + Math.max.apply(null, xs)) / 2;
    void best;
    var x = facing === "user" ? 1 - cx : cx;
    els.chip.textContent = text;
    els.chip.style.display = "block";
    els.chip.style.left = Math.min(0.85, Math.max(0.15, x)) * 100 + "%";
    els.chip.style.top = Math.max(0.02, top - 0.09) * 100 + "%";
  }

  // --------------------------------------------------------------- main loop
  function sortedHands(res) {
    var out = [];
    for (var i = 0; i < (res.landmarks || []).length; i++) {
      out.push({
        image: res.landmarks[i].map(function (q) { return [q.x, q.y, q.z]; }),
        world: res.worldLandmarks[i].map(function (q) { return [q.x, q.y, q.z]; }),
        side: res.handedness && res.handedness[i] && res.handedness[i][0] ? res.handedness[i][0].categoryName : "",
      });
    }
    out.sort(function (a, b) { return a.image[0][0] - b.image[0][0]; });
    return out;
  }

  function tick() {
    if (!running) return;
    requestAnimationFrame(tick);
    if (document.hidden || !landmarker || els.video.readyState < 2) return;
    if (els.video.currentTime === lastVideoTime) return;
    lastVideoTime = els.video.currentTime;
    var now = performance.now();
    var res;
    try { res = landmarker.detectForVideo(els.video, now); } catch (e) { return; }
    processHands(now, sortedHands(res));
  }

  /** Everything after landmark detection. Also callable from tests via window.__senseaTest. */
  function processHands(now, hands) {
    draw(hands);

    if (teach) { teachFrame(now, hands); return; }

    // Per hand: pose + motion
    var perHand = hands.map(function (hd, i) {
      var pose = S.classifyPose(hd.world, hd.image);
      var mo = motion[i].push(now, hd.world, hd.image, pose);
      if (mo) return { id: mo.id, label: S.SIGNS[mo.id].label, note: S.SIGNS[mo.id].note, confidence: mo.confidence, motion: true };
      return pose && pose.id ? { id: pose.id, label: pose.label, note: pose.note, confidence: pose.confidence } : null;
    });
    for (var j = hands.length; j < 2; j++) motion[j].reset();

    var cm = custom.push(now, hands);
    var best = null;
    perHand.forEach(function (d) { if (d && (!best || d.confidence > best.confidence)) best = d; });
    if (cm) best = { id: "custom:" + cm.id, label: cm.label, note: "Your own taught sign", confidence: cm.confidence, custom: true };

    var out = stab.update(now, best);
    var cur = out.current;
    placeChip(cur ? cur.label : "", hands);

    // Throttle: send when the picture changes or at most ~8x per second.
    var sig = hands.length + "|" + (cur ? cur.id : "") + "|" + perHand.map(function (d) { return d ? d.label : "-"; }).join(",");
    if (sig !== lastFrameSig || now - lastPost > 125) {
      lastFrameSig = sig; lastPost = now;
      post({
        type: "frame",
        hands: perHand.map(function (d, i) { return { label: d ? d.label : null, confidence: d ? d.confidence : 0, side: hands[i].side }; }),
        current: cur,
      });
    }
    if (out.confirmed) post({ type: "confirmed", sign: out.confirmed });
  }

  // ---------------------------------------------------------------- teaching
  var TAKES = 3, RECORD_MS = 1400;

  function teachFrame(now, hands) {
    if (teach.phase !== "record") { placeChip("", []); return; }
    if (hands.length) teach.frames.push({ t: now, v: S.frameVector(hands), n: hands.length });
    teach.total++;
    if (now - teach.t0 >= RECORD_MS) finishTake();
  }

  function overlayBig(text) {
    els.big.textContent = text || "";
    els.big.style.display = text ? "flex" : "none";
  }

  async function runTeach(id, label) {
    teach = { id: id, label: label, phase: "wait", takes: [], frames: [], total: 0, t0: 0, cancelled: false };
    custom.reset();
    for (var take = 1; take <= TAKES; take++) {
      for (var c = 3; c >= 1; c--) {
        if (teach.cancelled) return endTeach();
        teach.phase = "countdown";
        overlayBig(String(c));
        post({ type: "teach-progress", phase: "countdown", take: take, takes: TAKES, count: c });
        await sleep(900);
      }
      if (teach.cancelled) return endTeach();
      overlayBig("Sign now");
      post({ type: "teach-progress", phase: "record", take: take, takes: TAKES });
      teach.frames = []; teach.total = 0; teach.t0 = performance.now(); teach.phase = "record";
      var ok = await new Promise(function (resolve) { teach.done = resolve; });
      if (teach.cancelled) return endTeach();
      overlayBig("");
      if (!ok) {
        post({ type: "teach-failed", reason: "We could not see your hand for the whole sign. Keep it in view and try again." });
        return endTeach();
      }
    }
    var templates = teach.takes.map(function (r) { return round(r); });
    var sign = { id: id, label: label, templates: templates, threshold: S.makeThreshold(teach.takes) };
    sign.templates = templates;
    endTeach();
    var list = customSigns.filter(function (s) { return s.id !== id; }); list.push(sign);
    setCustom(list);
    post({ type: "taught", sign: sign });
  }
  function finishTake() {
    var f = teach.frames;
    var seen = f.length / Math.max(1, teach.total);
    var ok = f.length >= 8 && seen > 0.75;
    teach.phase = "wait";
    if (ok) {
      var tpl = S.makeTemplate(f);
      if (tpl) teach.takes.push(tpl); else ok = false;
    }
    teach.done(ok);
  }
  function endTeach() { teach = null; overlayBig(""); stab = new S.Stabilizer(); custom.reset(); }
  function sleep(ms) { return new Promise(function (r) { setTimeout(r, ms); }); }
  function round(arr) { return arr.map(function (x) { return Math.round(x * 1000) / 1000; }); }

  var customSigns = [];
  function setCustom(list) { customSigns = list || []; custom.setSigns(customSigns); }

  // ----------------------------------------------------------- host messages
  async function onHost(m) {
    if (!m || typeof m !== "object") return;
    if (m.type === "init") {
      setCustom(m.customSigns || []);
      if (m.facing) facing = m.facing;
      begin();
    } else if (m.type === "set-custom") setCustom(m.customSigns || []);
    else if (m.type === "teach") { if (!teach && landmarker) runTeach(m.id, m.label); }
    else if (m.type === "cancel-teach") { if (teach) { teach.cancelled = true; if (teach.done) teach.done(false); } }
    else if (m.type === "camera") {
      facing = m.facing === "environment" ? "environment" : "user";
      if (landmarker) { lastVideoTime = -1; await startCamera(); }
    }
  }
  window.__host = onHost;
  window.addEventListener("message", function (e) { onHost(e.data); });
  document.addEventListener("message", function (e) { // Android WebView
    try { onHost(typeof e.data === "string" ? JSON.parse(e.data) : e.data); } catch (x) { /* ignore */ }
  });

  var started = false;
  async function begin() {
    if (started) return; started = true;
    try {
      await loadModel();
      status("loading", "Starting camera…");
      if (!(await startCamera())) return;
      running = true;
      status("running", "");
      requestAnimationFrame(tick);
    } catch (e) {
      status("error", "Could not start hand tracking: " + (e && e.message ? e.message : e));
    }
  }

  window.__senseaTest = { processHands: processHands, setRunning: function (v) { running = v; } };

  post({ type: "ready" });
  // If the host never sends init (standalone testing), start on our own.
  setTimeout(function () { if (!started && !window.__waitForHost) begin(); }, 1500);
})();
