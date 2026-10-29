// ============================================
// SpiralSense - Dashboard Navigation
// ============================================

const form = document.getElementById("upload-form");
const input = document.getElementById("image-input");
const dropzone = document.getElementById("dropzone");
const selectedFile = document.getElementById("selected-file");
const selectedThumb = document.getElementById("selected-thumb");
const largePreview = document.getElementById("large-preview");
const analyzeButton = document.getElementById("analyze-button");
const buttonLabel = document.getElementById("button-label");
const uploadAnother = document.getElementById("upload-another");

let previewUrl = null;


// ============================================
// PAGE / TAB NAVIGATION
// ============================================

const tabPanels = document.querySelectorAll("[data-tab-panel]");
const tabLinks = document.querySelectorAll("[data-tab-target]");

function activateTab(tabName) {
  const targetPanel = document.querySelector(
    `[data-tab-panel="${tabName}"]`
  );

  if (!targetPanel) return;

  tabPanels.forEach(panel => {
    const isActive = panel === targetPanel;

    panel.hidden = !isActive;
    panel.classList.toggle("active", isActive);
  });

  // Update sidebar active state.
  document.querySelectorAll(".nav-link[data-tab-target]").forEach(link => {
    const isActive = link.dataset.tabTarget === tabName;

    link.classList.toggle("active", isActive);

    if (isActive) {
      link.setAttribute("aria-current", "page");
    } else {
      link.removeAttribute("aria-current");
    }
  });

  // Keep the page at the top after navigation.
  window.scrollTo({
    top: 0,
    behavior: "auto"
  });
}

tabLinks.forEach(link => {
  link.addEventListener("click", event => {
    event.preventDefault();

    const tabName = link.dataset.tabTarget;

    // Allow navigation to Results and Grad-CAM even before
    // an analysis has been completed, so their empty states work.
    activateTab(tabName);
  });
});


// ============================================
// RESULT STATES
// ============================================

const states = {
  empty: document.getElementById("empty-state"),
  loading: document.getElementById("loading-state"),
  error: document.getElementById("error-state"),
  result: document.getElementById("result-state")
};

function setState(name) {
  Object.entries(states).forEach(([key, element]) => {
    if (element) {
      element.hidden = key !== name;
    }
  });

  const labels = {
    empty: "AWAITING IMAGE",
    loading: "ANALYZING",
    error: "ERROR",
    result: "COMPLETE"
  };

  const status = document.getElementById("result-status");

  if (status) {
    status.textContent = labels[name] || "AWAITING IMAGE";
  }
}


// ============================================
// FORMATTING HELPERS
// ============================================

function percent(value) {
  if (value === null || value === undefined || value === "") {
    return "N/A";
  }

  const number = Number(value);

  return Number.isFinite(number)
    ? `${(number * 100).toFixed(1)}%`
    : "N/A";
}

function decimal(value, places = 4) {
  if (value === null || value === undefined || value === "") {
    return "N/A";
  }

  const number = Number(value);

  return Number.isFinite(number)
    ? number.toFixed(places)
    : "N/A";
}

function showError(message) {
  document.getElementById("error-message").textContent = message;
  setState("error");
}


// ============================================
// GRAD-CAM RESET
// ============================================

function resetGradCAM() {
  const result = document.getElementById("gradcam-result");
  const image = document.getElementById("gradcam-image");
  const placeholder = document.getElementById("gradcam-placeholder");

  result.hidden = true;
  image.removeAttribute("src");
  placeholder.hidden = false;

  placeholder.querySelector("p").textContent =
    "Analyze an image first, then return here to view its explanation.";
}


// ============================================
// IMAGE SELECTION AND PREVIEW
// ============================================

function selectImage(file) {
  if (!file) return;

  const acceptedTypes = [
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/bmp"
  ];

  if (!acceptedTypes.includes(file.type)) {
    showError("Choose a PNG, JPG, JPEG, WEBP, or BMP image.");
    return;
  }

  if (file.size > 10 * 1024 * 1024) {
    showError("Image must be 10 MB or smaller.");
    return;
  }

  const transfer = new DataTransfer();
  transfer.items.add(file);
  input.files = transfer.files;

  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
  }

  previewUrl = URL.createObjectURL(file);

  selectedThumb.src = previewUrl;
  largePreview.src = previewUrl;

  selectedThumb.hidden = false;
  largePreview.hidden = false;

  document.querySelector(".preview-placeholder").hidden = true;

  document.getElementById("file-name").textContent = file.name;

  document.getElementById("file-size").textContent =
    file.size < 1048576
      ? `${Math.round(file.size / 1024)} KB`
      : `${(file.size / 1048576).toFixed(2)} MB`;

  selectedFile.hidden = false;
  uploadAnother.hidden = false;
  dropzone.hidden = true;

  analyzeButton.disabled = false;
  buttonLabel.textContent = "Analyze image";

  document.getElementById("preview-status").textContent =
    "Image ready for analysis";

  document.querySelector(".file-status-dot").classList.add("ready");

  closeImageModal();
  resetGradCAM();
  setState("empty");

  // After choosing a new image, return to Dashboard.
  activateTab("dashboard");
}

function clearImage() {
  input.value = "";

  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
    previewUrl = null;
  }

  selectedThumb.removeAttribute("src");
  largePreview.removeAttribute("src");

  selectedThumb.hidden = true;
  largePreview.hidden = true;

  document.querySelector(".preview-placeholder").hidden = false;

  selectedFile.hidden = true;
  uploadAnother.hidden = true;
  dropzone.hidden = false;

  analyzeButton.disabled = true;
  buttonLabel.textContent = "Analyze image";

  document.getElementById("preview-status").textContent =
    "Waiting for image";

  document.querySelector(".file-status-dot").classList.remove("ready");

  document.getElementById("probability-meter").style.width = "0%";

  closeImageModal();
  resetGradCAM();
  setState("empty");

  activateTab("dashboard");
}


// ============================================
// IMAGE INPUT EVENTS
// ============================================

input.addEventListener("change", () => {
  if (input.files && input.files.length > 0) {
    selectImage(input.files[0]);
  }
});

document.getElementById("remove-image").addEventListener(
  "click",
  clearImage
);


// ============================================
// DRAG AND DROP
// ============================================

["dragenter", "dragover"].forEach(type => {
  dropzone.addEventListener(type, event => {
    event.preventDefault();
    dropzone.classList.add("dragover");
  });
});

["dragleave", "drop"].forEach(type => {
  dropzone.addEventListener(type, event => {
    event.preventDefault();
    dropzone.classList.remove("dragover");
  });
});

dropzone.addEventListener("drop", event => {
  const file = event.dataTransfer.files[0];

  if (file) {
    selectImage(file);
  }
});


// ============================================
// FULL-SIZE IMAGE MODAL
// ============================================

const imageModal = document.getElementById("image-modal");
const modalImage = document.getElementById("modal-image");
const modalImageName = document.getElementById("modal-image-name");

function openImageModal() {
  if (!previewUrl || !input.files[0]) return;

  modalImage.src = previewUrl;

  modalImageName.textContent =
    document.getElementById("file-name").textContent;

  imageModal.hidden = false;
  document.body.style.overflow = "hidden";

  document.getElementById("image-modal-close").focus();
}

function closeImageModal() {
  if (!imageModal) return;

  imageModal.hidden = true;
  document.body.style.overflow = "";

  modalImage.removeAttribute("src");
}

largePreview.addEventListener("click", openImageModal);
selectedThumb.addEventListener("click", openImageModal);

document.getElementById("image-modal-close").addEventListener(
  "click",
  closeImageModal
);

document.getElementById("image-modal-backdrop").addEventListener(
  "click",
  closeImageModal
);

document.addEventListener("keydown", event => {
  if (event.key === "Escape" && !imageModal.hidden) {
    closeImageModal();
  }
});


// ============================================
// GRAD-CAM RENDERING
// ============================================

function renderGradCAM(data) {
  const placeholder = document.getElementById("gradcam-placeholder");
  const result = document.getElementById("gradcam-result");
  const image = document.getElementById("gradcam-image");
  const caption = document.getElementById("gradcam-caption");

  result.hidden = true;
  image.removeAttribute("src");
  placeholder.hidden = false;

  if (!data.gradcam_url) {
    placeholder.querySelector("p").textContent =
      "Grad-CAM could not be generated. Check the Flask terminal for details.";

    if (data.gradcam_error) {
      console.error("Grad-CAM error:", data.gradcam_error);
    }

    return;
  }

  const separator = data.gradcam_url.includes("?") ? "&" : "?";

  image.onload = () => {
    result.hidden = false;
    placeholder.hidden = true;

    caption.textContent =
      "Grad-CAM overlay — research visualization, not a diagnosis.";
  };

  image.onerror = () => {
    result.hidden = true;
    placeholder.hidden = false;

    placeholder.querySelector("p").textContent =
      "The heatmap could not be loaded. Check the Flask static file path.";
  };

  image.src = `${data.gradcam_url}${separator}t=${Date.now()}`;
}


// ============================================
// PREDICTION RESULT RENDERING
// ============================================

function renderResult(data) {
  const prediction = String(data.prediction || "Unavailable");
  const isPositive = prediction.toUpperCase().includes("PARKINSON");

  document.getElementById("prediction-value").textContent = prediction;

  document.getElementById("prediction-caption").textContent =
    isPositive
      ? "Model classification at the configured threshold."
      : "Model classification below the configured threshold.";

  document.getElementById("probability-value").textContent =
    percent(data.calibrated_pd_probability);

  const probability = Number(data.calibrated_pd_probability);

  const meterWidth = Number.isFinite(probability)
    ? Math.max(0, Math.min(100, probability * 100))
    : 0;

  document.getElementById("probability-meter").style.width =
    `${meterWidth}%`;

  document.getElementById("uncertainty-value").textContent =
    percent(data.uncertainty_proxy);

  document.getElementById("mc-std").textContent =
    decimal(data.mc_dropout_standard_deviation);

  document.getElementById("threshold-value").textContent =
    percent(data.decision_threshold);

  document.getElementById("raw-value").textContent =
    percent(data.raw_pd_probability);

  document.getElementById("passes-value").textContent =
    data.mc_dropout_passes ?? "N/A";

  document.getElementById("model-value").textContent =
    data.model || "N/A";

  renderGradCAM(data);
  setState("result");
}


// ============================================
// ANALYZE IMAGE
// ============================================

analyzeButton.addEventListener("click", async () => {
  if (!input.files[0]) return;

  const fileToAnalyze = input.files[0];

  // Show the user the prediction page while inference runs.
  activateTab("results");
  setState("loading");

  analyzeButton.disabled = true;
  buttonLabel.textContent = "Analyzing…";

  document.getElementById("preview-status").textContent =
    "Analysis in progress";

  try {
    const payload = new FormData();
    payload.append("image", fileToAnalyze);

    const response = await fetch("/predict", {
      method: "POST",
      body: payload
    });

    const data = await response.json();

    console.log("Prediction API response:", data);

    if (!response.ok) {
      throw new Error(
        data.error || "The server could not analyze this image."
      );
    }

    renderResult(data);

    document.getElementById("preview-status").textContent =
      "Analysis complete";

    // Successful analysis automatically opens Prediction Results.
    activateTab("results");

  } catch (error) {
    console.error("Analysis error:", error);

    showError(
      error.message || "Could not connect to the local Flask server."
    );

    // Keep the error visible on the Results page.
    activateTab("results");

  } finally {
    analyzeButton.disabled = !input.files[0];
    buttonLabel.textContent = "Analyze image";
  }
});


// ============================================
// INITIAL PAGE
// ============================================

activateTab("dashboard");