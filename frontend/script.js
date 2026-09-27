// ============================================================
// RETINACARE - FRONTEND LOGIC & INTERACTION HANDLERS
// ============================================================

const API_URL = "http://127.0.0.1:8000";


// ============================================================
// DOM ELEMENTS
// ============================================================

const imageInput = document.getElementById("imageInput");
const chooseBtn = document.getElementById("chooseBtn");
const dropZone = document.getElementById("dropZone");

const previewContainer =
    document.getElementById("previewContainer");

const imagePreview =
    document.getElementById("imagePreview");

const fileName =
    document.getElementById("fileName");

const removeBtn =
    document.getElementById("removeBtn");

const analyzeBtn =
    document.getElementById("analyzeBtn");

const analyzeText =
    document.getElementById("analyzeText");

const spinner =
    document.getElementById("spinner");

const emptyResult =
    document.getElementById("emptyResult");

const actualResult =
    document.getElementById("actualResult");

const errorResult =
    document.getElementById("errorResult");

const errorMessage =
    document.getElementById("errorMessage");

const resultStatus =
    document.getElementById("resultStatus");

const predictedStage =
    document.getElementById("predictedStage");

const predictedClass =
    document.getElementById("predictedClass");

const confidenceValue =
    document.getElementById("confidenceValue");

const confidenceBar =
    document.getElementById("confidenceBar");

const probabilityList =
    document.getElementById("probabilityList");


// ============================================================
// CURRENT FILE
// ============================================================

let selectedFile = null;


// ============================================================
// CHOOSE IMAGE BUTTON & DROPZONE CLICK
// ============================================================

chooseBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    imageInput.click();
});

dropZone.addEventListener("click", () => {
    imageInput.click();
});


// ============================================================
// FILE INPUT
// ============================================================

imageInput.addEventListener("change", (event) => {

    const file = event.target.files[0];

    if (file) {
        handleFile(file);
    }

});


// ============================================================
// DRAG & DROP
// ============================================================

dropZone.addEventListener("dragover", (event) => {

    event.preventDefault();

    dropZone.classList.add("drag-over");

});


dropZone.addEventListener("dragleave", () => {

    dropZone.classList.remove("drag-over");

});


dropZone.addEventListener("drop", (event) => {

    event.preventDefault();

    dropZone.classList.remove("drag-over");

    const file = event.dataTransfer.files[0];

    if (file) {
        handleFile(file);
    }

});


// ============================================================
// HANDLE FILE
// ============================================================

function handleFile(file) {

    const allowedTypes = [
        "image/jpeg",
        "image/png"
    ];

    if (!allowedTypes.includes(file.type)) {

        showError(
            "Please select a valid JPG, JPEG or PNG retinal image."
        );

        return;
    }


    selectedFile = file;


    // Create preview URL

    const previewURL =
        URL.createObjectURL(file);

    imagePreview.src = previewURL;

    fileName.textContent = file.name;


    // Show preview

    previewContainer.classList.remove("hidden");

    dropZone.classList.add("hidden");

    analyzeBtn.disabled = false;


    // Reset previous result

    resetResult();

}


// ============================================================
// REMOVE IMAGE
// ============================================================

removeBtn.addEventListener("click", () => {

    selectedFile = null;

    imageInput.value = "";

    imagePreview.src = "";

    previewContainer.classList.add("hidden");

    dropZone.classList.remove("hidden");

    analyzeBtn.disabled = true;

    resetResult();

});


// ============================================================
// ANALYZE BUTTON
// ============================================================

analyzeBtn.addEventListener("click", async () => {

    if (!selectedFile) {
        return;
    }

    await analyzeImage();

});


// ============================================================
// ANALYZE IMAGE (API CALL)
// ============================================================

async function analyzeImage() {

    setLoading(true);

    hideAllResults();


    const formData = new FormData();

    formData.append(
        "file",
        selectedFile
    );


    try {

        const response = await fetch(
            `${API_URL}/predict`,
            {
                method: "POST",
                body: formData
            }
        );


        const data = await response.json();


        if (!response.ok) {

            throw new Error(
                data.detail ||
                "Prediction failed."
            );

        }


        displayResult(data);

    }

    catch (error) {

        console.error(
            "Prediction error:",
            error
        );

        showError(
            error.message ||
            "Unable to connect to the prediction server."
        );

    }

    finally {

        setLoading(false);

    }

}


// ============================================================
// DISPLAY RESULT
// ============================================================

function displayResult(data) {

    emptyResult.classList.add("hidden");

    errorResult.classList.add("hidden");

    actualResult.classList.remove("hidden");


    // Status badge update

    resultStatus.textContent =
        "Analysis Complete";

    resultStatus.className =
        "result-status complete";


    // Dynamic stage styling attribute

    actualResult.setAttribute(
        "data-stage",
        data.predicted_class
    );


    // Prediction

    predictedStage.textContent =
        data.predicted_stage;


    predictedClass.textContent =
        data.predicted_class;


    // Confidence

    const confidence =
        Number(data.confidence);


    confidenceValue.textContent =
        `${confidence.toFixed(2)}%`;


    confidenceBar.style.width =
        `${confidence}%`;


    // Probabilities

    probabilityList.innerHTML = "";


    const probabilities =
        data.probabilities;


    Object.entries(probabilities)
        .forEach(([stage, probability]) => {

            const row =
                document.createElement("div");

            const isPredicted =
                stage === data.predicted_stage;

            row.className = isPredicted
                ? "probability-row highlight"
                : "probability-row";


            row.innerHTML = `
                <span class="probability-name">
                    ${stage}
                </span>

                <div class="probability-track">
                    <div
                        class="probability-fill"
                        style="width: ${probability}%"
                    ></div>
                </div>

                <span class="probability-value">
                    ${Number(probability).toFixed(2)}%
                </span>
            `;


            probabilityList.appendChild(row);

        });

}


// ============================================================
// SHOW ERROR
// ============================================================

function showError(message) {

    emptyResult.classList.add("hidden");

    actualResult.classList.add("hidden");

    errorResult.classList.remove("hidden");

    resultStatus.textContent =
        "Error";

    resultStatus.className =
        "result-status error";

    errorMessage.textContent =
        message;

}


// ============================================================
// RESET RESULT
// ============================================================

function resetResult() {

    emptyResult.classList.remove("hidden");

    actualResult.classList.add("hidden");

    errorResult.classList.add("hidden");

    actualResult.removeAttribute("data-stage");

    resultStatus.textContent =
        "Awaiting Image";

    resultStatus.className =
        "result-status awaiting";

    confidenceBar.style.width =
        "0%";

    probabilityList.innerHTML = "";

}


// ============================================================
// HIDE RESULTS
// ============================================================

function hideAllResults() {

    emptyResult.classList.add("hidden");

    actualResult.classList.add("hidden");

    errorResult.classList.add("hidden");

}


// ============================================================
// LOADING STATE
// ============================================================

function setLoading(isLoading) {

    if (isLoading) {

        analyzeBtn.disabled = true;

        analyzeText.textContent =
            "Analyzing...";

        spinner.classList.remove("hidden");

    }

    else {

        analyzeBtn.disabled =
            !selectedFile;

        analyzeText.textContent =
            "Analyze Retina";

        spinner.classList.add("hidden");

    }

}


// ============================================================
// BACKEND STATUS VERIFICATION (OPTIONAL HEALTH CHECK)
// ============================================================

async function checkBackendStatus() {

    const statusDot = document.querySelector(".status-dot");
    const statusText = document.getElementById("statusText");

    try {

        const response = await fetch(`${API_URL}/health`);

        if (response.ok) {

            if (statusDot) statusDot.classList.remove("offline");
            if (statusText) statusText.textContent = "AI Model Ready";

        } else {

            if (statusDot) statusDot.classList.add("offline");
            if (statusText) statusText.textContent = "API Degraded";

        }

    } catch (e) {

        if (statusDot) statusDot.classList.add("offline");
        if (statusText) statusText.textContent = "Backend Offline";

    }

}

// Initial status check
checkBackendStatus();