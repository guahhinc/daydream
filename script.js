// Ultra-robust MediaPipe Loader
const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const loadingOverlay = document.getElementById("loading-overlay");
const loadingText = document.getElementById("loading-text");
const cameraBtn = document.getElementById("camera-toggle");
const captureBtn = document.getElementById("capture-btn");
const scannerLine = document.querySelector(".scanner-line");

let faceLandmarker;
let webcamRunning = false;
let lastVideoTime = -1;
let results = undefined;

// UI Elements
const mogScoreText = document.getElementById("mog-score");
const scoreRank = document.getElementById("score-rank");
const faceWidthText = document.getElementById("face-width");
const faceRatioText = document.getElementById("face-ratio");

// Error display
function showError(msg) {
    console.error("Mogger Error:", msg);
    loadingText.innerHTML = `<span style="color:#ff4444; font-weight:bold">AI INITIALIZATION FAILED</span><br><small style="font-size:0.6rem; opacity:0.8">${msg}</small><br><button onclick="location.reload()" style="background:var(--accent-color); border:none; color:#000; padding:8px 20px; border-radius:5px; margin-top:15px; cursor:pointer; font-weight:bold">RETRY</button>`;
}

window.addEventListener('unhandledrejection', event => {
    showError("Promise rejected: " + event.reason);
});

// Initialization with Retry and Fallback
async function init() {
    let attempts = 0;
    const maxAttempts = 3;
    
    while (attempts < maxAttempts) {
        try {
            loadingText.innerText = `STAGE 1: CONNECTING TO AI NODE (Attempt ${attempts + 1}/${maxAttempts})...`;
            
            // Wait for 'vision' global to be available
            if (typeof vision === 'undefined') {
                await new Promise(resolve => setTimeout(resolve, 1000));
                if (typeof vision === 'undefined') throw new Error("MediaPipe Library not found. Check your internet connection.");
            }

            const filesetResolver = await vision.FilesetResolver.forVisionTasks(
                "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
            );
            
            loadingText.innerText = "STAGE 2: DOWNLOADING NEURAL NETWORKS...";
            
            faceLandmarker = await vision.FaceLandmarker.createFromOptions(filesetResolver, {
                baseOptions: {
                    modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
                    delegate: "GPU"
                },
                outputFaceBlendshapes: true,
                runningMode: "VIDEO",
                numFaces: 1
            });
            
            loadingText.innerText = "STAGE 3: SYNCING CORE...";
            console.log("AI Ready.");
            
            loadingOverlay.style.opacity = "0";
            setTimeout(() => {
                loadingOverlay.style.display = "none";
            }, 800);
            return; // Success!
            
        } catch (error) {
            attempts++;
            console.warn(`Init attempt ${attempts} failed:`, error);
            if (attempts === maxAttempts) {
                // Final fallback to CPU
                try {
                    loadingText.innerText = "STAGE 1: FALLBACK TO CPU MODE...";
                    const filesetResolver = await vision.FilesetResolver.forVisionTasks(
                        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
                    );
                    faceLandmarker = await vision.FaceLandmarker.createFromOptions(filesetResolver, {
                        baseOptions: {
                            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
                            delegate: "CPU"
                        },
                        outputFaceBlendshapes: true,
                        runningMode: "VIDEO",
                        numFaces: 1
                    });
                    loadingOverlay.style.opacity = "0";
                    setTimeout(() => {
                        loadingOverlay.style.display = "none";
                    }, 800);
                    return;
                } catch (cpuError) {
                    showError(cpuError.message);
                }
            }
            await new Promise(resolve => setTimeout(resolve, 2000));
        }
    }
}

// Start Init
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}

// Camera Toggle
cameraBtn.addEventListener("click", async () => {
    if (!faceLandmarker) {
        alert("AI is still initializing...");
        return;
    }

    if (webcamRunning) {
        webcamRunning = false;
        cameraBtn.innerHTML = '<span class="icon">📷</span> START CAMERA';
        scannerLine.style.display = "none";
        if (video.srcObject) {
            video.srcObject.getTracks().forEach(track => track.stop());
            video.srcObject = null;
        }
    } else {
        try {
            const constraints = { video: { width: 1280, height: 720 } };
            const stream = await navigator.mediaDevices.getUserMedia(constraints);
            video.srcObject = stream;
            video.addEventListener("loadeddata", predictWebcam);
            webcamRunning = true;
            cameraBtn.innerHTML = '<span class="icon">⏹</span> STOP CAMERA';
            scannerLine.style.display = "block";
        } catch (err) {
            console.error("Camera error:", err);
            showError("Camera Access Denied: " + err.message);
        }
    }
});

async function predictWebcam() {
    if (!webcamRunning) return;

    if (video.videoWidth > 0) {
        canvasElement.width = video.videoWidth;
        canvasElement.height = video.videoHeight;
        
        let startTimeMs = performance.now();
        if (lastVideoTime !== video.currentTime) {
            lastVideoTime = video.currentTime;
            results = faceLandmarker.detectForVideo(video, startTimeMs);
        }

        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
        const drawingUtils = new vision.DrawingUtils(canvasCtx);

        if (results && results.faceLandmarks) {
            for (const landmarks of results.faceLandmarks) {
                drawingUtils.drawConnectors(landmarks, vision.FaceLandmarker.FACE_LANDMARKS_TESSELATION, { color: "#00f2ff22", lineWidth: 0.5 });
                drawingUtils.drawConnectors(landmarks, vision.FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE, { color: "#00f2ff", lineWidth: 1 });
                drawingUtils.drawConnectors(landmarks, vision.FaceLandmarker.FACE_LANDMARKS_LEFT_EYE, { color: "#00f2ff", lineWidth: 1 });
                drawingUtils.drawConnectors(landmarks, vision.FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, { color: "#00f2ff", lineWidth: 1.5 });
                processMetrics(landmarks);
            }
        }
    }
    window.requestAnimationFrame(predictWebcam);
}

function processMetrics(landmarks) {
    const leftEye = landmarks[33];
    const rightEye = landmarks[263];
    const noseTip = landmarks[4];
    const mouth = landmarks[13];
    const leftCheek = landmarks[234];
    const rightCheek = landmarks[454];
    const chin = landmarks[152];

    // Symmetry
    const distL = Math.hypot(leftEye.x - noseTip.x, leftEye.y - noseTip.y);
    const distR = Math.hypot(rightEye.x - noseTip.x, rightEye.y - noseTip.y);
    const symmetry = Math.max(0, 100 - Math.abs(distL - distR) * 2000);
    
    // Jawline
    const faceWidth = Math.hypot(leftCheek.x - rightCheek.x, leftCheek.y - rightCheek.y);
    const jawScore = Math.min(100, (1 - (leftCheek.y + rightCheek.y) / 2 + chin.y) * 150);

    // Canthal Tilt
    const tilt = (landmarks[133].y - leftEye.y + landmarks[362].y - rightEye.y) * 1000;
    const tiltScore = Math.min(100, Math.max(0, 50 + tilt));

    // Midface Ratio
    const eyeDist = Math.hypot(leftEye.x - rightEye.x, leftEye.y - rightEye.y);
    const noseToMouth = Math.hypot(noseTip.x - mouth.x, noseTip.y - mouth.y);
    const ratio = (eyeDist / (noseToMouth || 1)).toFixed(2);

    updateStat("symmetry", symmetry);
    updateStat("jawline", jawScore);
    updateStat("eye", tiltScore);
    
    faceWidthText.innerText = (faceWidth * 100).toFixed(0) + "px";
    faceRatioText.innerText = ratio;

    const baseScore = (symmetry * 0.3 + jawScore * 0.3 + tiltScore * 0.4) / 10;
    const currentScore = parseFloat(mogScoreText.innerText) || 0;
    const smoothScore = (1 - 0.1) * currentScore + 0.1 * baseScore;
    
    mogScoreText.innerText = smoothScore.toFixed(1);

    if (smoothScore > 8.5) scoreRank.innerText = "ELITE MOGGER";
    else if (smoothScore > 7) scoreRank.innerText = "CHAD STATUS";
    else if (smoothScore > 5) scoreRank.innerText = "ABOVE AVERAGE";
    else scoreRank.innerText = "NORMIE";
}

function updateStat(id, value) {
    const bar = document.getElementById(`${id}-bar`);
    const text = document.getElementById(`${id}-val`);
    if (bar && text) {
        const clamped = Math.min(100, Math.max(0, value));
        bar.style.width = clamped + "%";
        text.innerText = clamped.toFixed(0) + "%";
    }
}

// Capture
captureBtn.addEventListener("click", () => {
    if (!webcamRunning) return;
    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = video.videoWidth;
    tempCanvas.height = video.videoHeight;
    const ctx = tempCanvas.getContext("2d");
    ctx.translate(tempCanvas.width, 0); ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0); ctx.drawImage(canvasElement, 0, 0);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = "rgba(0,0,0,0.6)"; ctx.fillRect(0, tempCanvas.height-80, tempCanvas.width, 80);
    ctx.fillStyle = "#00f2ff"; ctx.font = "bold 30px Outfit"; ctx.fillText("MOGGER AI | SCORE: " + mogScoreText.innerText, 30, tempCanvas.height-30);
    
    const link = document.createElement("a");
    link.download = "mogger-result.png";
    link.href = tempCanvas.toDataURL("image/png");
    link.click();
    
    // Add to log
    const log = document.getElementById("mog-log");
    const empty = log.querySelector(".empty-log"); if (empty) empty.remove();
    const thumb = document.createElement("div"); thumb.className = "mog-thumb";
    thumb.style.backgroundImage = `url(${link.href})`;
    log.prepend(thumb); if (log.children.length > 6) log.lastElementChild.remove();
});
