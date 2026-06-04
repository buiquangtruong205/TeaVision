const API_BASE =
  window.location.port === "8000" ? "" : "http://127.0.0.1:8000";

const diseaseLabels = {
  algal_spot: "Đốm tảo",
  brown_blight: "Cháy lá nâu",
  gray_blight: "Cháy lá xám",
  healthy: "Lá khỏe mạnh",
  helopeltis: "Bọ xít muỗi",
  red_spot: "Đốm đỏ",
};

const diseaseDescriptions = {
  algal_spot: "Nhận diện dấu hiệu đốm tảo trên bề mặt lá.",
  brown_blight: "Nhận diện vùng lá có biểu hiện cháy nâu.",
  gray_blight: "Nhận diện biểu hiện cháy lá màu xám.",
  healthy: "Phân biệt lá khỏe mạnh không có dấu hiệu bệnh rõ ràng.",
  helopeltis: "Nhận diện tổn thương liên quan đến bọ xít muỗi.",
  red_spot: "Nhận diện các vùng đốm đỏ trên lá chè.",
};

const elements = {
  fileInput: document.getElementById("fileInput"),
  dropZone: document.getElementById("dropZone"),
  previewCard: document.getElementById("previewCard"),
  previewImage: document.getElementById("previewImage"),
  fileName: document.getElementById("fileName"),
  fileSize: document.getElementById("fileSize"),
  analyzeButton: document.getElementById("analyzeButton"),
  resetButton: document.getElementById("resetButton"),
  serverStatus: document.getElementById("serverStatus"),
  emptyResult: document.getElementById("emptyResult"),
  loadingResult: document.getElementById("loadingResult"),
  analysisResult: document.getElementById("analysisResult"),
  errorResult: document.getElementById("errorResult"),
  errorMessage: document.getElementById("errorMessage"),
  resultBadge: document.getElementById("resultBadge"),
  annotatedImage: document.getElementById("annotatedImage"),
  diseaseName: document.getElementById("diseaseName"),
  confidenceValue: document.getElementById("confidenceValue"),
  confidenceBar: document.getElementById("confidenceBar"),
  leafDetected: document.getElementById("leafDetected"),
  latencyValue: document.getElementById("latencyValue"),
  probabilityList: document.getElementById("probabilityList"),
  diseaseGrid: document.getElementById("diseaseGrid"),
  modelAccuracy: document.getElementById("modelAccuracy"),
  modelClasses: document.getElementById("modelClasses"),
  modelLatency: document.getElementById("modelLatency"),
};

let selectedFile = null;
let previewUrl = null;
let annotatedUrl = null;

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function displayName(className) {
  return diseaseLabels[className] || className.replaceAll("_", " ");
}

function renderDiseaseGuide(classNames) {
  elements.diseaseGrid.innerHTML = classNames
    .map(
      (className) => `
        <article class="disease-card">
          <strong>${displayName(className)}</strong>
          <span>${diseaseDescriptions[className] || "Nhóm bệnh được hỗ trợ bởi mô hình."}</span>
        </article>
      `,
    )
    .join("");
}

async function checkServer() {
  try {
    const [healthResponse, modelResponse] = await Promise.all([
      fetch(`${API_BASE}/health`),
      fetch(`${API_BASE}/model-info`),
    ]);
    if (!healthResponse.ok || !modelResponse.ok) throw new Error("API unavailable");
    const model = await modelResponse.json();
    elements.serverStatus.className = "server-status online";
    elements.serverStatus.lastElementChild.textContent = "Mô hình đã sẵn sàng";
    elements.modelClasses.textContent = model.class_names.length;
    elements.modelAccuracy.textContent = `${((model.metrics.accuracy || 0) * 100).toFixed(2)}%`;
    elements.modelLatency.textContent = `~${(model.metrics.inference_time_ms || 0).toFixed(1)} ms`;
    renderDiseaseGuide(model.class_names);
  } catch {
    elements.serverStatus.className = "server-status offline";
    elements.serverStatus.lastElementChild.textContent = "Không kết nối được API";
    renderDiseaseGuide(Object.keys(diseaseLabels));
  }
}

function setResultState(state) {
  elements.emptyResult.hidden = state !== "empty";
  elements.loadingResult.hidden = state !== "loading";
  elements.analysisResult.hidden = state !== "complete";
  elements.errorResult.hidden = state !== "error";
  elements.resultBadge.textContent =
    state === "complete" ? "Hoàn tất" : state === "loading" ? "Đang xử lý" : state === "error" ? "Có lỗi" : "Chờ ảnh";
  elements.resultBadge.classList.toggle("complete", state === "complete");
}

function selectFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    showError("Vui lòng chọn một tệp hình ảnh hợp lệ.");
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    showError("Ảnh vượt quá dung lượng tối đa 10 MB.");
    return;
  }

  selectedFile = file;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  elements.previewImage.src = previewUrl;
  elements.fileName.textContent = file.name;
  elements.fileSize.textContent = formatBytes(file.size);
  elements.dropZone.hidden = true;
  elements.previewCard.hidden = false;
  elements.resetButton.hidden = false;
  elements.analyzeButton.disabled = false;
  setResultState("empty");
}

function resetDemo() {
  selectedFile = null;
  elements.fileInput.value = "";
  elements.dropZone.hidden = false;
  elements.previewCard.hidden = true;
  elements.resetButton.hidden = true;
  elements.analyzeButton.disabled = true;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  if (annotatedUrl) URL.revokeObjectURL(annotatedUrl);
  previewUrl = null;
  annotatedUrl = null;
  setResultState("empty");
}

function showError(message) {
  elements.errorMessage.textContent = message;
  setResultState("error");
}

function renderProbabilities(probabilities) {
  const rows = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);
  elements.probabilityList.innerHTML = rows
    .map(
      ([className, probability]) => `
        <div class="probability-row">
          <span>${displayName(className)}</span>
          <div class="probability-track"><span style="width: ${probability * 100}%"></span></div>
          <strong>${(probability * 100).toFixed(1)}%</strong>
        </div>
      `,
    )
    .join("");
}

async function readError(response) {
  try {
    const data = await response.json();
    return data.detail || "API không thể xử lý ảnh.";
  } catch {
    return "API không thể xử lý ảnh.";
  }
}

async function analyzeImage() {
  if (!selectedFile) return;
  setResultState("loading");
  elements.analyzeButton.disabled = true;

  const predictionForm = new FormData();
  predictionForm.append("file", selectedFile);
  const annotatedForm = new FormData();
  annotatedForm.append("file", selectedFile);

  try {
    const [predictionResponse, annotatedResponse] = await Promise.all([
      fetch(`${API_BASE}/predict`, { method: "POST", body: predictionForm }),
      fetch(`${API_BASE}/predict/annotated`, { method: "POST", body: annotatedForm }),
    ]);

    if (!predictionResponse.ok) throw new Error(await readError(predictionResponse));
    if (!annotatedResponse.ok) throw new Error(await readError(annotatedResponse));

    const prediction = await predictionResponse.json();
    const annotatedBlob = await annotatedResponse.blob();
    if (annotatedUrl) URL.revokeObjectURL(annotatedUrl);
    annotatedUrl = URL.createObjectURL(annotatedBlob);
    elements.annotatedImage.src = annotatedUrl;

    const confidencePercent = prediction.confidence * 100;
    elements.diseaseName.textContent = displayName(prediction.class_name);
    elements.confidenceValue.textContent = `${confidencePercent.toFixed(1)}%`;
    elements.confidenceBar.style.width = `${confidencePercent}%`;
    elements.leafDetected.textContent = prediction.leaf_detected ? "Đã phát hiện" : "Dùng toàn ảnh";
    elements.latencyValue.textContent = `${prediction.latency_ms.toFixed(1)} ms`;
    renderProbabilities(prediction.probabilities);
    setResultState("complete");
  } catch (error) {
    showError(error.message || "Không thể kết nối tới API.");
  } finally {
    elements.analyzeButton.disabled = false;
  }
}

elements.fileInput.addEventListener("change", (event) => selectFile(event.target.files[0]));
elements.analyzeButton.addEventListener("click", analyzeImage);
elements.resetButton.addEventListener("click", resetDemo);

["dragenter", "dragover"].forEach((eventName) => {
  elements.dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.dropZone.classList.add("dragover");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  elements.dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.dropZone.classList.remove("dragover");
  });
});

elements.dropZone.addEventListener("drop", (event) => selectFile(event.dataTransfer.files[0]));

checkServer();
