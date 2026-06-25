const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const card = document.getElementById("card");
const btn = document.getElementById("btn");

async function startCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
}

function setUI(result) {
    card.className = "card";

    const status = result.status || "processing";

    card.classList.add(status);

    if (status === "known") {
        card.innerHTML = `✔ ${result.name}<br>${result.confidence || ""}`;
    }

    if (status === "unknown") {
        card.innerHTML = "Nepoznata osoba";
    }

    if (status === "spoof") {
        card.innerHTML = "Spoof detektovan";
    }

    if (status === "bad_quality") {
        card.innerHTML = "Loš kvalitet slike";
    }

    if (status === "processing") {
        card.innerHTML = "Obrada...";
    }
}

async function capture() {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    ctx.drawImage(video, 0, 0);

    const image = canvas.toDataURL("image/jpeg");

    setUI({ status: "processing" });

    const res = await fetch("http://127.0.0.1:8000/upload", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image })
    });

    const data = await res.json();

    const result = data.faces?.[0];

    if (!result) {
        setUI({ status: "unknown" });
        return;
    }

    setUI(result);
}

btn.onclick = capture;

setInterval(capture, 30000);

startCamera();