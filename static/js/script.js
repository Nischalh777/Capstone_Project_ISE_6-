document.addEventListener('DOMContentLoaded', function() {
    // DOM Element Selection
    const uploadForm = document.getElementById('upload-form');
    const fileInput = document.getElementById('file-upload');
    const fileLabel = document.querySelector('.file-label');
    const resultsSection = document.getElementById('results-section');
    const loader = document.getElementById('loader');
    const resultContent = document.getElementById('result-content');
    const resultImage = document.getElementById('result-image');
    const predictedCropSpan = document.getElementById('predicted-crop');
    const predictedDiseaseSpan = document.getElementById('predicted-disease');
    const confidenceScoreSpan = document.getElementById('confidence-score');
    const chatbotResponseDiv = document.getElementById('chatbot-response');
    const chatbotContainer = document.querySelector('.chatbot-container');

    // Prevent default form submission which can cause page reloads
    uploadForm.addEventListener('submit', function(event) {
        event.preventDefault();
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            const file = fileInput.files[0];
            const formData = new FormData();
            formData.append('file', file);
            handlePrediction(formData);
        }
    });

    async function handlePrediction(data) {
        console.log("JS: Starting prediction...");
        resultsSection.style.display = 'block';
        loader.style.display = 'block';
        resultContent.style.display = 'none';

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: data,
            });
            
            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.error || `Server Error: ${response.status}`);
            }

            console.log("JS: Received data from server:", result);
            displayResults(result);

        } catch (error) {
            console.error('JS: A critical error occurred:', error);
            displayError(error.message);
        }
    }

    function displayResults(data) {
        // This function displays the data received from the server.
        console.log("JS: Displaying results...");
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
        resultImage.src = data.image_url;
        confidenceScoreSpan.textContent = `${data.confidence}%`;
        
        // Use the marked library for the chatbot response
        if (data.chatbot_response) {
            chatbotResponseDiv.innerHTML = marked.parse(data.chatbot_response);
        }

        if (data.gemini_success) {
            chatbotContainer.style.backgroundColor = 'var(--light-green)';
        } else {
            chatbotContainer.style.backgroundColor = '#ffebee';
        }
        
        loader.style.display = 'none';
        resultContent.style.display = 'block';
    }

    function displayError(errorMessage) {
        console.log("JS: Displaying error message.");
        loader.style.display = 'none';
        resultContent.style.display = 'block';
        predictedCropSpan.textContent = "Error";
        predictedDiseaseSpan.textContent = "Analysis Failed";
        confidenceScoreSpan.textContent = "N/A";
        chatbotResponseDiv.innerHTML = `<p style="color: red;">${errorMessage}</p>`;
    }
});