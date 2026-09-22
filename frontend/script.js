const video = document.getElementById("camera");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");

const startButton = document.getElementById("startCamera");
const placeholder = document.getElementById("cameraPlaceholder");
const cameraContainer = document.querySelector(".camera-container");
const cameraStatus = document.getElementById("cameraStatus");
const systemStatus = document.getElementById("systemStatus");

const signName = document.getElementById("signName");
const confidence = document.getElementById("confidence");
const confidenceFill = document.getElementById("confidenceFill");
const predictionStatus = document.getElementById("predictionStatus");

const audioText = document.getElementById("audioText");
const speakButton = document.getElementById("speakButton");
const history = document.getElementById("history");

let stream = null;
let running = false;
let lastPrediction = null;
let lastSpoken = "";
let lastSpokenTime = 0;

const voiceMessages = {
    no_entry: "Warning. No entry ahead.",
    no_left_turn: "No left turn ahead.",
    no_right_turn: "No right turn ahead.",
    speed_90: "Speed limit is 90 kilometers per hour.",
    speed_110: "Speed limit is 110 kilometers per hour.",
    left_turn: "Left turn ahead.",
    right_turn: "Right turn ahead.",
    roundabout: "Roundabout ahead.",
    guarded_railway_crossing:
        "Railway crossing ahead. Guarded crossing.",
    unguarded_railway_crossing:
        "Warning. Unguarded railway crossing ahead.",
    parking: "Parking area ahead.",
    bus_stop: "Bus stop ahead."
};

const signSymbols = {
    no_entry: "⛔",
    no_left_turn: "↩",
    no_right_turn: "↪",
    speed_90: "90",
    speed_110: "110",
    left_turn: "←",
    right_turn: "→",
    roundabout: "↻",
    guarded_railway_crossing: "🚂",
    unguarded_railway_crossing: "🚂",
    parking: "P",
    bus_stop: "BUS"
};

startButton.addEventListener("click", startCamera);

speakButton.addEventListener("click", () => {
    if (!lastPrediction) return;

    const message =
        voiceMessages[lastPrediction.class_name] ||
        lastPrediction.class_name.replaceAll("_", " ");

    speak(message);
});

async function startCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: "environment",
                width: { ideal: 1280 },
                height: { ideal: 720 }
            },
            audio: false
        });

        video.srcObject = stream;

        await video.play();

        video.style.display = "block";
        placeholder.style.display = "none";

        cameraContainer.classList.add("active");

        cameraStatus.textContent = "CAMERA LIVE";
        cameraStatus.style.color = "var(--green)";

        systemStatus.textContent = "SYSTEM LIVE";

        running = true;

        resizeCanvas();

        requestAnimationFrame(processFrame);

    } catch (error) {
        console.error(error);

        predictionStatus.textContent =
            "Camera permission was denied or unavailable.";

        cameraStatus.textContent = "CAMERA ERROR";
        cameraStatus.style.color = "var(--danger)";
    }
}

function resizeCanvas() {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
}

async function processFrame() {
    if (!running) return;

    if (
        video.readyState >= 2 &&
        video.videoWidth > 0 &&
        video.videoHeight > 0
    ) {
        await sendFrame();
    }

    setTimeout(() => {
        requestAnimationFrame(processFrame);
    }, 350);
}

async function sendFrame() {
    const tempCanvas = document.createElement("canvas");

    tempCanvas.width = video.videoWidth;
    tempCanvas.height = video.videoHeight;

    const tempCtx = tempCanvas.getContext("2d");

    tempCtx.drawImage(
        video,
        0,
        0,
        tempCanvas.width,
        tempCanvas.height
    );

    tempCanvas.toBlob(
        async (blob) => {
            if (!blob) return;

            const formData = new FormData();

            formData.append(
                "file",
                blob,
                "frame.jpg"
            );

            try {
                const response = await fetch(
                    "/predict",
                    {
                        method: "POST",
                        body: formData
                    }
                );

                const result = await response.json();

                handlePrediction(result);

            } catch (error) {
                console.error(error);
                predictionStatus.textContent =
                    "Backend connection unavailable.";
            }
        },
        "image/jpeg",
        0.75
    );
}

function handlePrediction(result) {
    ctx.clearRect(
        0,
        0,
        canvas.width,
        canvas.height
    );

    if (!result.detected) {
        signName.textContent = "Waiting for detection";
        confidence.textContent = "—";
        confidenceFill.style.width = "0%";

        predictionStatus.textContent =
            result.message || "Scanning...";

        speakButton.disabled = true;

        lastPrediction = null;

        return;
    }

    lastPrediction = result;

    const className = result.class_name;

    const displayName = className
        .replaceAll("_", " ")
        .toUpperCase();

    const classConfidence =
        result.classification_confidence * 100;

    signName.textContent = displayName;

    confidence.textContent =
        `${classConfidence.toFixed(1)}%`;

    confidenceFill.style.width =
        `${classConfidence}%`;

    predictionStatus.textContent =
        `Detected and classified by AI`;

    speakButton.disabled = false;

    drawBoundingBox(result);

    const now = Date.now();

    if (
        className !== lastSpoken ||
        now - lastSpokenTime > 5000
    ) {
        const message =
            voiceMessages[className] ||
            displayName;

        audioText.textContent =
            `"${message}"`;

        speak(message);

        addHistory(
            displayName,
            classConfidence
        );

        lastSpoken = className;
        lastSpokenTime = now;
    }
}

function drawBoundingBox(result) {
    const [x1, y1, x2, y2] = result.box;

    const scaleX =
        canvas.width / video.videoWidth;

    const scaleY =
        canvas.height / video.videoHeight;

    const x = x1 * scaleX;
    const y = y1 * scaleY;
    const width = (x2 - x1) * scaleX;
    const height = (y2 - y1) * scaleY;

    ctx.strokeStyle = "#63ffad";
    ctx.lineWidth = 4;

    ctx.strokeRect(
        x,
        y,
        width,
        height
    );

    ctx.fillStyle = "rgba(6, 17, 13, 0.85)";

    const label =
        `${result.class_name.replaceAll("_", " ")} ` +
        `${(result.classification_confidence * 100).toFixed(0)}%`;

    ctx.font = "bold 16px Arial";

    const textWidth =
        ctx.measureText(label).width + 20;

    ctx.fillRect(
        x,
        Math.max(0, y - 32),
        textWidth,
        32
    );

    ctx.fillStyle = "#63ffad";

    ctx.fillText(
        label,
        x + 10,
        Math.max(21, y - 10)
    );
}

function speak(message) {
    if (!("speechSynthesis" in window)) {
        return;
    }

    window.speechSynthesis.cancel();

    const utterance =
        new SpeechSynthesisUtterance(message);

    utterance.rate = 0.95;
    utterance.pitch = 1.0;
    utterance.volume = 1.0;

    window.speechSynthesis.speak(
        utterance
    );
}

function addHistory(name, score) {
    const empty =
        history.querySelector(".empty-history");

    if (empty) {
        empty.remove();
    }

    const item =
        document.createElement("div");

    item.className = "history-item";

    const time =
        new Date().toLocaleTimeString(
            [],
            {
                hour: "2-digit",
                minute: "2-digit"
            }
        );

    item.innerHTML = `
        <strong>${name}</strong>
        <span>${score.toFixed(1)}% • ${time}</span>
    `;

    history.prepend(item);

    while (history.children.length > 6) {
        history.lastElementChild.remove();
    }
}
