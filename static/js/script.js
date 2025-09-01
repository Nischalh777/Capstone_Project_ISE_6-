document.addEventListener('DOMContentLoaded', function() {
    // --- DOM Element Selection ---
    const fileInput = document.getElementById('file-upload');
    const fileNameSpan = document.getElementById('file-name');
    const fileLabel = document.querySelector('.file-label');
    const startCameraBtn = document.getElementById('start-camera-btn');
    const cameraModal = document.getElementById('camera-modal');
    const closeModalBtn = document.querySelector('.close-btn');
    const webcamStream = document.getElementById('webcam-stream');
    const capturePhotoBtn = document.getElementById('capture-photo-btn');
    const resultsSection = document.getElementById('results-section');
    const loader = document.getElementById('loader');
    const resultContent = document.getElementById('result-content');
    const resultImage = document.getElementById('result-image');
    const predictedCropSpan = document.getElementById('predicted-crop');
    const predictedDiseaseSpan = document.getElementById('predicted-disease');
    const confidenceScoreSpan = document.getElementById('confidence-score');
    const chatbotResponseDiv = document.getElementById('chatbot-response');
    const chatbotContainer = document.querySelector('.chatbot-container');
    let stream;

    // --- Event Listeners ---
    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            const file = fileInput.files[0];
            fileNameSpan.textContent = file.name;
            const formData = new FormData();
            formData.append('file', file);
            handlePrediction(formData, '/predict');
        }
    });
    fileLabel.addEventListener('dragover', (e) => { e.preventDefault(); fileLabel.classList.add('dragover'); });
    fileLabel.addEventListener('dragleave', () => { fileLabel.classList.remove('dragover'); });
    fileLabel.addEventListener('drop', (e) => {
        e.preventDefault();
        fileLabel.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            fileInput.files = e.dataTransfer.files;
            fileNameSpan.textContent = e.dataTransfer.files[0].name;
            const formData = new FormData();
            formData.append('file', e.dataTransfer.files[0]);
            handlePrediction(formData, '/predict');
        }
    });

    // --- Camera Logic ---
    startCameraBtn.addEventListener('click', async () => {
        cameraModal.style.display = 'block';
        try {
            stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
            webcamStream.srcObject = stream;
        } catch (err) {
            console.error("Error accessing camera: ", err);
            alert("Could not access the camera. Please ensure permissions are granted and that your connection is secure (https).");
            cameraModal.style.display = 'none';
        }
    });

    function stopCamera() {
        if (stream) { stream.getTracks().forEach(track => track.stop()); }
        cameraModal.style.display = 'none';
    }

    closeModalBtn.addEventListener('click', stopCamera);
    window.addEventListener('click', (event) => { if (event.target == cameraModal) stopCamera(); });

    capturePhotoBtn.addEventListener('click', () => {
        const canvas = document.createElement('canvas');
        canvas.width = webcamStream.videoWidth;
        canvas.height = webcamStream.videoHeight;
        canvas.getContext('2d').drawImage(webcamStream, 0, 0);
        const imageDataURL = canvas.toDataURL('image/png');
        handlePrediction(JSON.stringify({ image: imageDataURL }), '/capture');
        stopCamera();
    });

    // --- ASYNCHRONOUS WORKFLOW ---
    async function handlePrediction(data, endpoint) {
        resultsSection.style.display = 'block';
        loader.style.display = 'block';
        resultContent.style.display = 'none';
        chatbotResponseDiv.innerHTML = '';
        window.scrollTo({ top: resultsSection.offsetTop, behavior: 'smooth' });

        try {
            const response = await fetch(endpoint, {
                method: 'POST',
                headers: (typeof data === 'string') ? {'Content-Type': 'application/json'} : {},
                body: data,
            });
            const result = await response.json();
            if (!response.ok) { throw new Error(result.error || `Server error: ${response.status}`); }
            
            displayInitialResults(result);

        } catch (error) {
            console.error('Prediction failed:', error);
            displayError(error.message);
        }
    }

    function displayInitialResults(data) {
        const confidence = parseFloat(data.confidence);
        const CONFIDENCE_THRESHOLD = 50.0;

        resultImage.src = data.image_url;
        loader.style.display = 'none';
        resultContent.style.display = 'block';
        confidenceScoreSpan.innerHTML = `${confidence.toFixed(2)}%`;

        if (confidence < CONFIDENCE_THRESHOLD) {
            predictedCropSpan.textContent = "Unknown Image";
            predictedDiseaseSpan.textContent = "Please use a clearer image.";
            confidenceScoreSpan.innerHTML += " <strong style='color: #c62828;'>(Low Confidence)</strong>";
            updateChatbotBox({
                gemini_success: false,
                chatbot_response: "The model is not confident in this prediction. This may not be a plant leaf, or the image could be blurry. Please try again with a clearer picture of a single leaf."
            });
        } else {
            const rawPrediction = data.prediction;
            let cropName = "Unknown";
            let diseaseName = rawPrediction.replace(/_/g, ' ');
            if (rawPrediction.includes('___')) {
                [cropName, diseaseName] = rawPrediction.split('___');
                cropName = cropName.replace(/_/g, ' ');
                diseaseName = diseaseName.replace(/_/g, ' ');
            }
            predictedCropSpan.textContent = cropName;
            predictedDiseaseSpan.textContent = diseaseName;
            
            chatbotResponseDiv.innerHTML = '<p><em>Contacting AI assistant for more information...</em></p>';
            chatbotContainer.style.backgroundColor = '#f5f5f5';
            chatbotContainer.style.borderColor = '#ddd';
            
            fetchChatbotInfo(data.prediction);
        }
    }

    async function fetchChatbotInfo(prediction) {
        try {
            const response = await fetch('/get_info', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ prediction: prediction }),
            });
            const result = await response.json();
            if (!response.ok) { throw new Error(result.error || "Failed to get chatbot info."); }
            updateChatbotBox(result);
        } catch (error) {
            console.error('Chatbot info fetch failed:', error);
            updateChatbotBox({ gemini_success: false, chatbot_response: error.message });
        }
    }

    function updateChatbotBox(data) {
        if (data.gemini_success) {
            chatbotResponseDiv.innerHTML = marked.parse(data.chatbot_response);
            chatbotContainer.style.backgroundColor = 'var(--light-green)';
            chatbotContainer.style.borderColor = 'var(--secondary-color)';
        } else {
            chatbotResponseDiv.innerHTML = `<p style="font-weight: bold; color: #c62828;">${data.chatbot_response}</p>`;
            chatbotContainer.style.backgroundColor = '#ffebee';
            chatbotContainer.style.borderColor = '#c62828';
        }
    }

    function displayError(errorMessage) {
        loader.style.display = 'none';
        resultContent.style.display = 'block';
        predictedCropSpan.textContent = "Error";
        predictedDiseaseSpan.textContent = "Analysis Failed";
        confidenceScoreSpan.textContent = "N/A";
        updateChatbotBox({ gemini_success: false, chatbot_response: errorMessage });
    }
});