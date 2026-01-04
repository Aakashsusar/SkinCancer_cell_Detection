let video = document.getElementById('video');
let canvas = document.getElementById('canvas');
let preview = document.getElementById('preview');
let capturedBlob = null;
let cameraStream = null;

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

function toggleCamera() {
  const snapBtn = document.getElementById('snapBtn');
  const captureBtnText = document.getElementById('captureBtnText');
  
  if (video.style.display === 'block') {
    // Stop camera
    stopCamera();
    video.style.display = 'none';
    snapBtn.style.display = 'none';
    captureBtnText.innerText = 'CAPTURE IMAGE';
  } else {
    // Start camera
    startCamera();
  }
}

function startCamera() {
  const snapBtn = document.getElementById('snapBtn');
  const captureBtnText = document.getElementById('captureBtnText');
  
  navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } })
    .then(stream => {
      cameraStream = stream;
      video.style.display = 'block';
      video.srcObject = stream;
      snapBtn.style.display = 'flex';
      captureBtnText.innerText = 'STOP CAMERA';
    })
    .catch(err => {
      alert("Camera access denied or unavailable.");
      console.error(err);
    });
}

function stopCamera() {
  if (cameraStream) {
    cameraStream.getTracks().forEach(track => track.stop());
    video.srcObject = null;
    cameraStream = null;
  }
}

function captureImage() {
  if (!video.srcObject) {
    alert("Camera not started.");
    return;
  }

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);
  
  // Show preview
  preview.src = canvas.toDataURL('image/jpeg');

  // Convert to blob for upload
  canvas.toBlob(blob => {
    capturedBlob = blob;
  }, 'image/jpeg');
  
  // Stop camera after capture
  stopCamera();
  video.style.display = 'none';
  document.getElementById('snapBtn').style.display = 'none';
  document.getElementById('captureBtnText').innerText = 'CAPTURE IMAGE';
  
  alert("Photo captured! Click 'GET RESULT' to analyze.");
}

async function uploadImage() {
  const resultText = document.getElementById('result');
  const confidenceInfo = document.querySelector('.confidence-info');
  const resultBox = document.querySelector('.result-box');
  const checkmarkIcon = document.querySelector('.checkmark-icon');
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

  // Reset styles
  resultBox.classList.remove('result-benign', 'result-malignant', 'result-invalid');
  checkmarkIcon.classList.remove('success-pulse');
  resultText.innerText = "ANALYZING...";
  confidenceInfo.innerHTML = '<p>CONFIDENCE SCORE:</p><p>RECOMMENDATION:</p>';

  try {
    const response = await fetch("https://skincancer-cell-detection.onrender.com/predict", {
      method: "POST",
      body: formData
    });

    if (!response.ok) {
      const errData = await response.json();
      resultText.innerText = `ERROR: ${errData.error || "Unknown error"}`;
      return;
    }

    const data = await response.json();
    
    // Handle invalid/non-skin images
    if (data.is_invalid) {
      resultBox.classList.add('result-invalid');
      resultText.innerHTML = `⚠️ INVALID IMAGE`;
      confidenceInfo.innerHTML = `
        <p class="confidence-label">DETECTION CONFIDENCE:</p>
        <p class="confidence-value">${data.confidence}%</p>
        <p class="recommendation-label">NOTICE:</p>
        <p class="recommendation-text">${data.recommendation}</p>
      `;
      return;
    }
    
    // Add animation class based on result
    if (data.is_cancerous) {
      resultBox.classList.add('result-malignant');
      resultText.innerHTML = `⚠️ MALIGNANT DETECTED`;
    } else {
      resultBox.classList.add('result-benign');
      resultText.innerHTML = `✓ BENIGN - SAFE`;
      checkmarkIcon.classList.add('success-pulse');
    }
    
    // Update confidence and recommendation
    confidenceInfo.innerHTML = `
      <p class="confidence-label">CONFIDENCE SCORE:</p>
      <p class="confidence-value">${data.confidence}%</p>
      <p class="recommendation-label">RECOMMENDATION:</p>
      <p class="recommendation-text">${data.recommendation}</p>
    `;
    
  } catch (err) {
    console.error(err);
    resultText.innerText = "❌ Could not connect to server.";
  }
}
