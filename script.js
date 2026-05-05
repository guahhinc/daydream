// Using global 'vision' from index.html bundle
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

// Global error handling
window.onerror = function(msg, url, line) {
    if (loadingText) {
        loadingText.innerHTML = `<span style="color:#ff4444">ERROR: ${msg}</span><br><small>${url}:${line}</small>`;
    }
    return false;
};

// Initialize Face Landmarker
async function createFaceLandmarker() {
    try {
        console.log("Loading FilesetResolver...");
        loadingText.innerText = "STAGE 1: RESOLVING AI CORE...";
        
        const filesetResolver = await vision.FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
        );
        
        loadingText.innerText = "STAGE 2: DOWNLOADING NEURAL NETWORKS...";
        console.log("Creating FaceLandmarker...");
        
        faceLandmarker = await vision.FaceLandmarker.createFromOptions(filesetResolver, {
            baseOptions: {
                modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
                delegate: "GPU"
            },
            outputFaceBlendshapes: true,
            runningMode: "VIDEO",
            numFaces: 1
        });
        
        loadingText.innerText = "STAGE 3: CALIBRATING...";
        console.log("FaceLandmarker ready.");
        
        loadingOverlay.style.opacity = "0";
        setTimeout(() => {
            loadingOverlay.style.display = "none";
        }, 800);
    } catch (error) {
        console.error("Initialization error, trying CPU fallback...", error);
        try {
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
        } catch (cpuError) {
            loadingText.innerHTML = `<span style="color:#ff4444">CRITICAL AI FAILURE</span><br><small>${cpuError.message}</small>`;
        }
    }
}
createFaceLandmarker();

// Camera Logic
cameraBtn.addEventListener("click", () => {
    if (!faceLandmarker) return;

    if (webcamRunning) {
        webcamRunning = false;
        cameraBtn.innerHTML = '<span class="icon">📷</span> START CAMERA';
        scannerLine.style.display = "none";
        if (video.srcObject) {
            video.srcObject.getTracks().forEach(t => t.stop());
            video.srcObject = null;
        }
    } else {
        webcamRunning = true;
        cameraBtn.innerHTML = '<span class="icon">⏹</span> STOP CAMERA';
        scannerLine.style.display = "block";
        navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } })
            .then(stream => {
                video.srcObject = stream;
                video.addEventListener("loadeddata", predictWebcam);
            })
            .catch(err => {
                alert("Camera Access Error: " + err.message);
                webcamRunning = false;
            });
    }
});

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
    const drawingUtils = new vision.DrawingUtils(canvasCtx);

    if (results && results.faceLandmarks) {
        for (const landmarks of results.faceLandmarks) {
            drawingUtils.drawConnectors(landmarks, vision.FaceLandmarker.FACE_LANDMARKS_TESSELATION, { color: "#00f2ff11", lineWidth: 0.5 });
            drawingUtils.drawConnectors(landmarks, vision.FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE, { color: "#00f2ff", lineWidth: 1 });
            drawingUtils.drawConnectors(landmarks, vision.FaceLandmarker.FACE_LANDMARKS_LEFT_EYE, { color: "#00f2ff", lineWidth: 1 });
            drawingUtils.drawConnectors(landmarks, vision.FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, { color: "#00f2ff", lineWidth: 1.5 });
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
    const leftCheek = landmarks[234];
    const rightCheek = landmarks[454];
    const mouth = landmarks[13];

    const distL = Math.hypot(leftEye.x - noseTip.x, leftEye.y - noseTip.y);
    const distR = Math.hypot(rightEye.x - noseTip.x, rightEye.y - noseTip.y);
    const symmetry = Math.max(0, 100 - Math.abs(distL - distR) * 2000);
    
    const faceWidth = Math.hypot(leftCheek.x - rightCheek.x, leftCheek.y - rightCheek.y);
    const jawScore = Math.min(100, (1 - (leftCheek.y + rightCheek.y) / 2 + chin.y) * 150);

    const tiltScore = Math.min(100, Math.max(0, 50 + (landmarks[133].y - leftEye.y + landmarks[362].y - rightEye.y) * 1000));
    const midfaceRatio = (Math.hypot(leftEye.x - rightEye.x, leftEye.y - rightEye.y) / (Math.hypot(noseTip.x - mouth.x, noseTip.y - mouth.y) || 1)).toFixed(2);

    updateStat("symmetry", symmetry);
    updateStat("jawline", jawScore);
    updateStat("eye", tiltScore);
    
    faceWidthText.innerText = (faceWidth * 100).toFixed(0) + "px";
    faceRatioText.innerText = midfaceRatio;

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
    if (!bar || !text) return;
    const clampedVal = Math.min(100, Math.max(0, value));
    bar.style.width = clampedVal + "%";
    text.innerText = clampedVal.toFixed(0) + "%";
}

// Capture Logic
captureBtn.addEventListener("click", () => {
    if (!webcamRunning) return;
    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = video.videoWidth;
    tempCanvas.height = video.videoHeight;
    const ctx = tempCanvas.getContext("2d");
    ctx.translate(tempCanvas.width, 0); ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0); ctx.drawImage(canvasElement, 0, 0);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = "rgba(0,0,0,0.5)"; ctx.fillRect(0, tempCanvas.height-60, tempCanvas.width, 60);
    ctx.fillStyle = "#00f2ff"; ctx.font = "20px Outfit"; ctx.fillText("MOGGER AI | SCORE: " + mogScoreText.innerText, 20, tempCanvas.height-25);
    const dataUrl = tempCanvas.toDataURL("image/png");
    const link = document.createElement("a"); link.download = "mog.png"; link.href = dataUrl; link.click();
    
    const log = document.getElementById("mog-log");
    const empty = log.querySelector(".empty-log"); if (empty) empty.remove();
    const thumb = document.createElement("div"); thumb.className = "mog-thumb"; thumb.style.backgroundImage = `url(${dataUrl})`;
    log.prepend(thumb); if (log.children.length > 6) log.lastElementChild.remove();
});
