let video = document.getElementById('video');
let canvas = document.getElementById('canvas');
let preview = document.getElementById('preview');
let capturedBlob = null;

function previewFile() {
  const fileInput = document.getElementById('fileInput');
  const file = fileInput.files[0];
  const allowedTypes = ['image/jpeg', 'image/png'];

  if (file && allowedTypes.includes(file.type)) {
    const reader = new FileReader();
    reader.onloadend = () => {
      preview.src = reader.result;
      capturedBlob = null;
    };
    reader.readAsDataURL(file);
  } else {
    alert("Please upload a valid image (JPG or PNG).");
  }
}

function startCamera() {
  navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
      video.style.display = 'block';
      video.srcObject = stream;
    })
    .catch(err => {
      alert("Camera access denied or unavailable.");
      console.error(err);
    });
}

function captureImage() {
  if (!video.srcObject) {
    alert("Camera not started.");
    return;
  }

  canvas.style.display = 'block';
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);
  preview.src = canvas.toDataURL('image/jpeg');

  canvas.toBlob(blob => {
    capturedBlob = blob;
  }, 'image/jpeg');
}

async function uploadImage() {
  const resultText = document.getElementById('result');
  const formData = new FormData();

  const fileInput = document.getElementById("fileInput");
  const file = fileInput.files[0];

  if (capturedBlob) {
    formData.append("file", capturedBlob, "captured.jpg");
  } else if (file && ['image/jpeg', 'image/png'].includes(file.type)) {
    formData.append("file", file);
  } else {
    resultText.innerText = "⚠️ Please upload or capture a valid image.";
    return;
  }

  resultText.innerText = "🔄 Predicting...";

  try {
    const response = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      body: formData
    });

    if (!response.ok) {
      const errData = await response.json();
      resultText.innerText = `❌ Error: ${errData.error || "Unknown error"}`;
      return;
    }

    const data = await response.json();
    resultText.innerText = "✅ Prediction: " + data.prediction;
  } catch (err) {
    console.error(err);
    resultText.innerText = "❌ Could not connect to server.";
  }
}
