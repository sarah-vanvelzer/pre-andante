import { PoseLandmarker, FilesetResolver, DrawingUtils } from "@mediapipe/tasks-vision";
import * as Tone from "tone";

const video = document.getElementById("video") as HTMLVideoElement;
const canvas = document.getElementById("canvas") as HTMLCanvasElement;
const status = document.getElementById("status")!;
const ctx = canvas.getContext("2d")!;

// ---------- Graph ----------
const graph = document.getElementById("graph") as HTMLCanvasElement;
const g = graph.getContext("2d")!;
type Frame = { v: number; hi: number; lo: number; step: boolean };
const history: Frame[] = [];

function drawGraph() {
  const W = graph.width, H = graph.height, MAX = 0.6;
  const y = (v: number) => H - (Math.min(v, MAX) / MAX) * H;
  const x = (i: number) => (i / 300) * W;
  g.clearRect(0, 0, W, H);
  const line = (key: "v" | "hi" | "lo", color: string) => {
    g.strokeStyle = color; g.beginPath();
    history.forEach((p, i) => (i ? g.lineTo(x(i), y(p[key])) : g.moveTo(x(i), y(p[key]))));
    g.stroke();
  };
  g.fillStyle = "#fbbf24";
  history.forEach((p, i) => { if (p.step) g.fillRect(x(i) - 1, 0, 2, H); });
  line("hi", "#f87171"); // red = "foot up" threshold
  line("lo", "#60a5fa"); // blue = "foot down" threshold
  line("v", "#4ade80");  // green = smoothed foot-height gap
}

// ---------- Sound (click per step, for debugging) ----------
const soundBtn = document.getElementById("sound") as HTMLButtonElement;
let synth: Tone.MembraneSynth | null = null;
soundBtn.onclick = async () => {
  await Tone.start();
  synth = new Tone.MembraneSynth().toDestination();
  soundBtn.textContent = "Sound on";
};

async function main() {
  status.textContent = "Starting camera…";
  video.srcObject = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
  await video.play();

  status.textContent = "Loading pose model…";
  const vision = await FilesetResolver.forVisionTasks("/node_modules/@mediapipe/tasks-vision/wasm");
  const pose = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numPoses: 1,
    minPoseDetectionConfidence: 0.3,
    minTrackingConfidence: 0.3,
  });

  const draw = new DrawingUtils(ctx);
  let lastVideoTime = -1;

  // ---------- Step detection settings ----------
  const FOOT = [27, 28, 29, 30, 31, 32]; // ankles, heels, toes (both feet)
  const ALPHA = 0.35;        // smoothing: lower = smoother but slower
  const WINDOW_MS = 2000;    // thresholds adapt to the last 2 s
  const MIN_RANGE = 0.05;    // less movement than this = standing still, don't count
  const MIN_STEP_MS = 250;   // max 240 steps/min

  let smooth = 0;
  let footUp = false;
  let steps = 0;
  let lastStepTime = 0;
  const recent: { t: number; v: number }[] = [];
  const stepTimes: number[] = [];
  let cadence = 0;

  // ---------- Beat (auto-locks to your cadence) ----------
  const beatBtn = document.getElementById("beat") as HTMLButtonElement;
  let beatOn = false;
  let armed = false; // waiting to auto-lock?

  function startBeat() {
    const transport = Tone.getTransport(); // Tone's own steady clock
    const beatSynth = new Tone.MembraneSynth().toDestination();
    transport.bpm.value = Math.round(cadence); // 1 beat per step
    transport.scheduleRepeat((time) => {
      beatSynth.triggerAttackRelease("C3", "16n", time);
    }, "4n"); // once per beat
    transport.start();
    beatOn = true;
    armed = false;
    beatBtn.textContent = `Beat: ${Math.round(transport.bpm.value)} BPM (click to stop)`;
  }

  function stopBeat() {
    const transport = Tone.getTransport();
    transport.stop();
    transport.cancel();
    beatOn = false;
    beatBtn.textContent = "Lock beat";
  }

  beatBtn.onclick = async () => {
    await Tone.start(); // the one click that unlocks audio
    if (beatOn) { stopBeat(); return; }
    armed = true;
    beatBtn.textContent = "Armed: start marching…";
  };

  function loop() {
    if (video.currentTime !== lastVideoTime) {
      lastVideoTime = video.currentTime;
      const now = performance.now();
      const result = pose.detectForVideo(video, now);
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const lm = result.landmarks[0];
      if (!lm) {
        status.textContent = "No person detected";
      } else {
        draw.drawConnectors(lm, PoseLandmarker.POSE_CONNECTIONS, { color: "#4ade80", lineWidth: 3 });
        draw.drawLandmarks(lm, { color: "#f97316", radius: 3 });

        // Need at least one ankle clearly visible
        const ankleVis = Math.max(lm[27].visibility ?? 0, lm[28].visibility ?? 0);
        if (ankleVis < 0.5) {
          status.textContent = `Steps: ${steps} | Cadence: ${cadence.toFixed(0)} spm (feet not visible)`;
        } else {
          // 1. Raw signal: height gap between highest and lowest foot point,
          //    divided by leg length so it works at any distance from the camera
          const ys = FOOT.map((i) => lm[i].y);
          const hipY = (lm[23].y + lm[24].y) / 2;
          const ankleY = (lm[27].y + lm[28].y) / 2;
          const legLen = Math.max(Math.abs(ankleY - hipY), 0.05);
          const raw = (Math.max(...ys) - Math.min(...ys)) / legLen;

          // 2. Smooth it (exponential moving average)
          smooth += ALPHA * (raw - smooth);

          // 3. Adaptive thresholds from the last 2 s
          recent.push({ t: now, v: smooth });
          while (recent.length && now - recent[0].t > WINDOW_MS) recent.shift();
          const vals = recent.map((r) => r.v);
          const peak = Math.max(...vals), trough = Math.min(...vals);
          const range = peak - trough;
          const hi = trough + 0.6 * range;
          const lo = trough + 0.35 * range;

          // 4. Count: up through hi = step, back down through lo = ready for next
          let stepped = false;
          if (range > MIN_RANGE) {
            if (!footUp && smooth > hi && now - lastStepTime > MIN_STEP_MS) {
              steps++;
              stepped = true;
              footUp = true;
              lastStepTime = now;
              synth?.triggerAttackRelease("C2", "16n");

              stepTimes.push(now);
              if (stepTimes.length > 8) stepTimes.shift();
              if (stepTimes.length >= 3) {
                const gaps = stepTimes.slice(1).map((t, i) => t - stepTimes[i]).sort((a, b) => a - b);
                cadence = 60000 / gaps[Math.floor(gaps.length / 2)];

                // 5. Auto-lock the beat once the last 8 steps are steady
                if (armed && !beatOn && stepTimes.length >= 8) {
                  // ignore the single fastest + slowest gap (one odd step shouldn't block it)
                  const spread = (gaps[gaps.length - 2] - gaps[1]) / gaps[Math.floor(gaps.length / 2)];
                  beatBtn.textContent = `Armed… steadiness ${spread.toFixed(2)} (need < 0.35)`;
                  if (spread < 0.35) startBeat();
                }
              }
            }
            if (footUp && smooth < lo) footUp = false;
          }

          status.textContent = `Steps: ${steps} | Cadence: ${cadence.toFixed(0)} spm`;
          history.push({ v: smooth, hi, lo, step: stepped });
          if (history.length > 300) history.shift();
          drawGraph();
        }
      }
    }
    requestAnimationFrame(loop);
  }
  loop();
}

main().catch((e) => (status.textContent = "Error: " + e.message));