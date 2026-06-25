const video = document.getElementById("video");
const bg = document.getElementById("bg");

const card = document.getElementById("card");
const title = document.getElementById("title");
const subtitle = document.getElementById("subtitle");

let processing = false;

/* CAMERA */
async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "user" },
    audio: false
  });

  video.srcObject = stream;
  bg.srcObject = stream;
}

/* FRAME CAPTURE */
function getFrame() {
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0);

  return canvas.toDataURL("image/jpeg", 0.7);
}

/* REAL FACE API CALL */
async function recognizeFrame(frame) {
  console.log("SENDING FRAME SIZE:", frame.length);
  const res = await fetch("http://127.0.0.1:8000/upload", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ image: frame })
  });

  return await res.json();
}

/* UI UPDATE */
function updateUI(result) {
  card.className = "card " + result.status;

  title.innerText = result.name;

  if (result.status === "known") {
    subtitle.innerText = `Pouzdanost: ${result.confidence}%`;
  } else if (result.status === "unknown") {
    subtitle.innerText = "Osoba nije u bazi";
  } else if (result.status === "spoof") {
    subtitle.innerText = "Nije pravo lice";
  } else if (result.status === "quality") {
    subtitle.innerText = "Loš kvalitet slike";
  }
}

/* LOOP (FRAME PIPELINE) */
async function processLoop() {
  if (processing) return;

  processing = true;

  try {
    card.className = "card processing";
    title.innerText = "Obrada...";
    subtitle.innerText = "Analiza lica";

    const frame = getFrame();

    const result = await recognizeFrame(frame);

    console.log("FRAME RESULT:", result);

    updateUI(result);
  } catch (err) {
    console.error(err);
  }

  processing = false;
}

/* START STREAM */

console.log("This works!")
startCamera();

/* RUN EVERY 500ms (real-time feel) */
setInterval(processLoop, 500);

