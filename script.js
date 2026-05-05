import {
    FaceLandmarker,
    FilesetResolver,
    DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const loadingOverlay = document.getElementById("loading-overlay");
const cameraBtn = document.getElementById("camera-toggle");
const scannerLine = document.querySelector(".scanner-line");

let faceLandmarker;
let webcamRunning = false;
let lastVideoTime = -1;
let results = undefined;

// UI Elements
const symmetryBar = document.getElementById("symmetry-bar");
const symmetryVal = document.getElementById("symmetry-val");
const jawlineBar = document.getElementById("jawline-bar");
const jawlineVal = document.getElementById("jawline-val");
const eyeBar = document.getElementById("eye-bar");
const eyeVal = document.getElementById("eye-val");
const mogScoreText = document.getElementById("mog-score");
const scoreRank = document.getElementById("score-rank");
const faceWidthText = document.getElementById("face-width");
const faceRatioText = document.getElementById("face-ratio");

// Initialize Face Landmarker
async function createFaceLandmarker() {
    const loadingText = loadingOverlay.querySelector("p");
    
    try {
        console.log("Loading AI Models... This may take a few seconds.");
        loadingText.innerText = "INITIALIZING CORE... (STAGE 1/3)";
        
        const filesetResolver = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
        );
        
        loadingText.innerText = "DOWNLOADING NEURAL NETWORKS... (STAGE 2/3)";
        console.log("FilesetResolver loaded.");

        faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
            baseOptions: {
                modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
                delegate: "GPU"
            },
            outputFaceBlendshapes: true,
            runningMode: "VIDEO",
            numFaces: 1
        });
        
        loadingText.innerText = "CALIBRATING... (STAGE 3/3)";
        console.log("FaceLandmarker initialized with GPU.");
        
        loadingOverlay.style.opacity = "0";
        setTimeout(() => {
            loadingOverlay.style.display = "none";
        }, 800);
    } catch (gpuError) {
        console.warn("GPU Initialization failed, falling back to CPU:", gpuError);
        try {
            const filesetResolver = await FilesetResolver.forVisionTasks(
                "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
            );
            faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
                baseOptions: {
                    modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
                    delegate: "CPU"
                },
                outputFaceBlendshapes: true,
                runningMode: "VIDEO",
                numFaces: 1
            });
            console.log("FaceLandmarker initialized with CPU.");
            loadingOverlay.style.opacity = "0";
            setTimeout(() => {
                loadingOverlay.style.display = "none";
            }, 800);
        } catch (cpuError) {
            console.error("Critical AI Error:", cpuError);
            loadingText.innerHTML = `<span style="color:#ff4444">AI INITIALIZATION FAILED</span><br><small style="font-size:0.6rem; opacity:0.6; letter-spacing:1px">${cpuError.message}</small><br><button onclick="location.reload()" style="background:none; border:1px solid #444; color:#fff; padding:5px 15px; margin-top:10px; cursor:pointer; font-size:0.6rem">RETRY</button>`;
        }
    }
}
createFaceLandmarker();

// Enable the live webcam view and start detection.
cameraBtn.addEventListener("click", () => {
    if (!faceLandmarker) {
        alert("AI models are still loading. Please wait a moment.");
        return;
    }

    if (webcamRunning === true) {
        webcamRunning = false;
        cameraBtn.innerHTML = '<span class="icon">📷</span> START CAMERA';
        scannerLine.style.display = "none";
        stopCamera();
    } else {
        webcamRunning = true;
        cameraBtn.innerHTML = '<span class="icon">⏹</span> STOP CAMERA';
        scannerLine.style.display = "block";
        startCamera();
    }
});

function startCamera() {
    const constraints = {
        video: {
            width: { ideal: 1280 },
            height: { ideal: 720 },
            facingMode: "user"
        }
    };
    
    navigator.mediaDevices.getUserMedia(constraints)
        .then((stream) => {
            video.srcObject = stream;
            video.addEventListener("loadeddata", () => {
                console.log("Webcam active.");
                predictWebcam();
            });
        })
        .catch((err) => {
            console.error("Camera access error:", err);
            alert("Camera access denied or unavailable. Please enable permissions in your browser settings.");
            webcamRunning = false;
            cameraBtn.innerHTML = '<span class="icon">📷</span> START CAMERA';
            scannerLine.style.display = "none";
        });
}

function stopCamera() {
    const stream = video.srcObject;
    if (stream) {
        const tracks = stream.getTracks();
        tracks.forEach((track) => track.stop());
        video.srcObject = null;
    }
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
}

async function predictWebcam() {
    if (!webcamRunning) return;

    if (video.videoWidth > 0) {
        canvasElement.style.width = video.clientWidth + "px";
        canvasElement.style.height = video.clientHeight + "px";
        canvasElement.width = video.videoWidth;
        canvasElement.height = video.videoHeight;

        let startTimeMs = performance.now();
        if (lastVideoTime !== video.currentTime) {
            lastVideoTime = video.currentTime;
            results = faceLandmarker.detectForVideo(video, startTimeMs);
        }

        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
        const drawingUtils = new DrawingUtils(canvasCtx);

        if (results && results.faceLandmarks) {
            for (const landmarks of results.faceLandmarks) {
                drawingUtils.drawConnectors(
                    landmarks,
                    FaceLandmarker.FACE_LANDMARKS_TESSELATION,
                    { color: "#00f2ff11", lineWidth: 0.5 }
                );
                drawingUtils.drawConnectors(
                    landmarks,
                    FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE,
                    { color: "#00f2ff", lineWidth: 1 }
                );
                drawingUtils.drawConnectors(
                    landmarks,
                    FaceLandmarker.FACE_LANDMARKS_LEFT_EYE,
                    { color: "#00f2ff", lineWidth: 1 }
                );
                drawingUtils.drawConnectors(
                    landmarks,
                    FaceLandmarker.FACE_LANDMARKS_FACE_OVAL,
                    { color: "#00f2ff", lineWidth: 1.5 }
                );

                processMetrics(landmarks);
            }
        }
    }

    window.requestAnimationFrame(predictWebcam);
}

function processMetrics(landmarks) {
    const leftEyeInner = landmarks[133];
    const leftEyeOuter = landmarks[33];
    const rightEyeInner = landmarks[362];
    const rightEyeOuter = landmarks[263];
    const noseTip = landmarks[4];
    const chin = landmarks[152];
    const topHead = landmarks[10];
    const leftCheek = landmarks[234];
    const rightCheek = landmarks[454];
    const mouthCenter = landmarks[13];

    // 1. Symmetry
    const distL = Math.hypot(leftEyeInner.x - noseTip.x, leftEyeInner.y - noseTip.y);
    const distR = Math.hypot(rightEyeInner.x - noseTip.x, rightEyeInner.y - noseTip.y);
    const symmetry = Math.max(0, 100 - Math.abs(distL - distR) * 2000);
    
    // 2. Jawline / Face Width
    const faceWidth = Math.hypot(leftCheek.x - rightCheek.x, leftCheek.y - rightCheek.y);
    const jawScore = Math.min(100, (1 - (leftCheek.y + rightCheek.y) / 2 + chin.y) * 150);

    // 3. Canthal Tilt (Angle of eyes)
    const leftTilt = (leftEyeInner.y - leftEyeOuter.y) * 100;
    const rightTilt = (rightEyeInner.y - rightEyeOuter.y) * 100;
    const tiltScore = Math.min(100, Math.max(0, 50 + (leftTilt + rightTilt) * 20));

    // 4. Midface Ratio (Distance between pupils vs eyes-to-mouth)
    const eyeDist = Math.hypot(leftEyeOuter.x - rightEyeOuter.x, leftEyeOuter.y - rightEyeOuter.y);
    const eyeToMouth = Math.hypot(noseTip.x - mouthCenter.x, noseTip.y - mouthCenter.y);
    const midfaceRatio = (eyeDist / (eyeToMouth || 1)).toFixed(2);
    const midfaceScore = Math.max(0, 100 - Math.abs(1.0 - (eyeDist / eyeToMouth / 5)) * 100);

    // Update UI
    updateStat("symmetry", symmetry);
    updateStat("jawline", jawScore);
    updateStat("eye", tiltScore); // Eye spacing replaced with Canthal Tilt for better 'Mog' vibes
    
    faceWidthText.innerText = (faceWidth * 100).toFixed(0) + "px";
    faceRatioText.innerText = midfaceRatio;

    // Mog Score Calculation
    const baseScore = (symmetry * 0.3 + jawScore * 0.3 + tiltScore * 0.2 + midfaceScore * 0.2) / 10;
    const currentScore = parseFloat(mogScoreText.innerText) || 0;
    const smoothScore = lerp(currentScore, baseScore, 0.1);
    
    mogScoreText.innerText = smoothScore.toFixed(1);

    let rank = "CALIBRATING...";
    if (smoothScore > 8.8) rank = "ELITE MOGGER";
    else if (smoothScore > 7.5) rank = "CHAD STATUS";
    else if (smoothScore > 6) rank = "ABOVE AVERAGE";
    else if (smoothScore > 4) rank = "NORMIE";
    else rank = "BRUTALLY MOGGED";
    
    scoreRank.innerText = rank;
}

function updateStat(id, value) {
    const bar = document.getElementById(`${id}-bar`);
    const text = document.getElementById(`${id}-val`);
    if (!bar || !text) return;
    const clampedVal = Math.min(100, Math.max(0, value));
    bar.style.width = clampedVal + "%";
    text.innerText = clampedVal.toFixed(0) + "%";
}

const captureBtn = document.getElementById("capture-btn");

captureBtn.addEventListener("click", () => {
    if (!webcamRunning) {
        alert("Please start the camera first.");
        return;
    }

    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = video.videoWidth;
    tempCanvas.height = video.videoHeight;
    const ctx = tempCanvas.getContext("2d");

    // 1. Draw Video Frame (Mirrored)
    ctx.save();
    ctx.translate(tempCanvas.width, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0);
    ctx.restore();

    // 2. Draw Landmarks (Mirrored)
    ctx.save();
    ctx.translate(tempCanvas.width, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(canvasElement, 0, 0);
    ctx.restore();

    // 3. Add Watermark and Score
    ctx.fillStyle = "rgba(0, 0, 0, 0.5)";
    ctx.fillRect(0, tempCanvas.height - 100, tempCanvas.width, 100);
    
    ctx.fillStyle = "#00f2ff";
    ctx.font = "bold 40px Outfit, sans-serif";
    ctx.fillText("MOGGER AI", 40, tempCanvas.height - 40);
    
    ctx.fillStyle = "#fff";
    ctx.font = "30px Inter, sans-serif";
    const scoreText = `SCORE: ${mogScoreText.innerText} / 10`;
    const rankText = scoreRank.innerText;
    ctx.fillText(`${scoreText} - ${rankText}`, tempCanvas.width - 450, tempCanvas.height - 40);

    // 4. Download
    const dataUrl = tempCanvas.toDataURL("image/png");
    const link = document.createElement("a");
    link.download = `mogger-score-${Date.now()}.png`;
    link.href = dataUrl;
    link.click();

    // 5. Update Mog Log
    addToLog(dataUrl);
});

function addToLog(imgData) {
    const log = document.getElementById("mog-log");
    const empty = log.querySelector(".empty-log");
    if (empty) empty.remove();

    const thumb = document.createElement("div");
    thumb.className = "mog-thumb";
    thumb.style.backgroundImage = `url(${imgData})`;
    
    // Add to start
    log.prepend(thumb);

    // Limit to 6
    if (log.children.length > 6) {
        log.lastElementChild.remove();
    }
}

function lerp(start, end, amt) {
    return (1 - amt) * start + amt * end;
}
