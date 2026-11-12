import { PoseLandmarker, FilesetResolver, DrawingUtils } from "@mediapipe/tasks-vision";

const video = document.getElementById("video") as HTMLVideoElement;
const canvas = document.getElementById("canvas") as HTMLCanvasElement;
const status = document.getElementById("status")!;
const ctx = canvas.getContext("2d")!;

async function main() {
  // Camera
  status.textContent = "Starting camera…";
  video.srcObject = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480 },
  });
  await video.play();

  // Pose model
  status.textContent = "Loading pose model…";
  const vision = await FilesetResolver.forVisionTasks("/node_modules/@mediapipe/tasks-vision/wasm");
  const pose = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numPoses: 1,
    minPoseDetectionConfidence: 0.3,
    minTrackingConfidence: 0.3,
  });

  const draw = new DrawingUtils(ctx);
  let lastVideoTime = -1;

  // Runs every animation frame
  function loop() {
    if (video.currentTime !== lastVideoTime) {
      lastVideoTime = video.currentTime;
      const result = pose.detectForVideo(video, performance.now());
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const lm = result.landmarks[0];
      if (lm) {
        draw.drawConnectors(lm, PoseLandmarker.POSE_CONNECTIONS, { color: "#4ade80", lineWidth: 3 });
        draw.drawLandmarks(lm, { color: "#f97316", radius: 3 });
        // 27 = left ankle, 28 = right ankle
        const ankleVis = Math.min(lm[27].visibility ?? 0, lm[28].visibility ?? 0);
        if (ankleVis < 0.5) {
          status.textContent = `Ankles not visible (${ankleVis.toFixed(2)})`;
        } else {
          const sep = lm[27].x - lm[28].x;
          status.textContent = `Ankle separation: ${sep.toFixed(3)}`;
        }
      } else {
        status.textContent = "No person detected";
      }
    }
    requestAnimationFrame(loop);
  }
  loop();
}

main().catch((e) => (status.textContent = "Error: " + e.message));