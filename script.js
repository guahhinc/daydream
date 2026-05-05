import {
    FaceLandmarker,
    FilesetResolver,
    DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/vision_bundle.mjs";

const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const loadingOverlay = document.getElementById("loading-overlay");
const cameraBtn = document.getElementById("camera-toggle");
const scannerLine = document.querySelector(".scanner-line");

let faceLandmarker;
let runningMode = "VIDEO";
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
    try {
        console.log("Initializing FilesetResolver...");
        const filesetResolver = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
        );
        
        console.log("Creating FaceLandmarker...");
        faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
            baseOptions: {
                modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
                delegate: "GPU"
            },
            outputFaceBlendshapes: true,
            runningMode: "VIDEO",
            numFaces: 1
        });
        
        console.log("FaceLandmarker created successfully.");
        loadingOverlay.style.opacity = "0";
        setTimeout(() => {
            loadingOverlay.style.display = "none";
        }, 500);
    } catch (error) {
        console.error("Error initializing FaceLandmarker:", error);
        const loadingText = loadingOverlay.querySelector("p");
        if (loadingText) {
            loadingText.innerText = "ERROR LOADING AI. CHECK CONSOLE.";
            loadingText.style.color = "#ff4444";
        }
    }
}
createFaceLandmarker();

// Enable the live webcam view and start detection.
cameraBtn.addEventListener("click", () => {
    if (!faceLandmarker) {
        console.warn("Face Landmarker not ready.");
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
            width: 1280,
            height: 720
        }
    };
    navigator.mediaDevices.getUserMedia(constraints)
        .then((stream) => {
            video.srcObject = stream;
            video.addEventListener("loadeddata", () => {
                console.log("Webcam stream loaded.");
                predictWebcam();
            });
        })
        .catch((err) => {
            console.error("Error accessing webcam:", err);
            alert("Could not access camera. Please check permissions.");
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
                { color: "#00f2ff22", lineWidth: 0.5 }
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

    window.requestAnimationFrame(predictWebcam);
}

function processMetrics(landmarks) {
    const leftEye = landmarks[33];
    const rightEye = landmarks[263];
    const noseTip = landmarks[4];
    const chin = landmarks[152];
    const topHead = landmarks[10];
    const leftCheek = landmarks[234];
    const rightCheek = landmarks[454];

    // 1. Symmetry
    const distL = Math.hypot(leftEye.x - noseTip.x, leftEye.y - noseTip.y);
    const distR = Math.hypot(rightEye.x - noseTip.x, rightEye.y - noseTip.y);
    const symmetry = Math.max(0, 100 - Math.abs(distL - distR) * 1500);
    
    // 2. Jawline
    const faceWidth = Math.hypot(leftCheek.x - rightCheek.x, leftCheek.y - rightCheek.y);
    const jawScore = Math.min(100, (1 - (leftCheek.y + rightCheek.y) / 2 + chin.y) * 150);

    // 3. Eye Spacing
    const eyeDist = Math.hypot(leftEye.x - rightEye.x, leftEye.y - rightEye.y);
    const eyeSpacing = (eyeDist / faceWidth) * 200;
    const eyeScore = Math.max(0, 100 - Math.abs(45 - eyeSpacing) * 3);

    // 4. Face Ratio
    const faceHeight = Math.hypot(topHead.x - chin.x, topHead.y - chin.y);
    const ratio = (faceHeight / faceWidth).toFixed(2);

    // Update UI
    updateStat("symmetry", symmetry);
    updateStat("jawline", jawScore);
    updateStat("eye", eyeScore);
    
    faceWidthText.innerText = (faceWidth * 100).toFixed(0) + "px";
    faceRatioText.innerText = ratio;

    // Mog Score Calculation
    const baseScore = (symmetry * 0.4 + jawScore * 0.3 + eyeScore * 0.3) / 10;
    const currentScore = parseFloat(mogScoreText.innerText) || 0;
    const smoothScore = lerp(currentScore, baseScore, 0.1);
    
    mogScoreText.innerText = smoothScore.toFixed(1);

    let rank = "CALIBRATING...";
    if (smoothScore > 8.5) rank = "ELITE MOGGER";
    else if (smoothScore > 7) rank = "CHAD STATUS";
    else if (smoothScore > 5.5) rank = "ABOVE AVERAGE";
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

function lerp(start, end, amt) {
    return (1 - amt) * start + amt * end;
}
