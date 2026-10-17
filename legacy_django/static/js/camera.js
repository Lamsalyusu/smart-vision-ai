(function () {
  "use strict";
  const videoElement = document.getElementById("videoElement");
  const detectionCanvas = document.getElementById("detectionCanvas");
  const ctx = detectionCanvas.getContext("2d");
  const cameraPlaceholder = document.getElementById("cameraPlaceholder");
  const statusDot = document.getElementById("statusDot");
  const statusText = document.getElementById("statusText");
  const fpsBadge = document.getElementById("fpsBadge");
  const startCameraBtn = document.getElementById("startCameraBtn");
  const stopCameraBtn = document.getElementById("stopCameraBtn");
  const toggleDetectionBtn = document.getElementById("toggleDetectionBtn");
  const snapshotBtn = document.getElementById("snapshotBtn");
  const snapshotFlash = document.getElementById("snapshotFlash");
  const snapshotGallery = document.getElementById("snapshotGallery");
  const clearSnapshotsBtn = document.getElementById("clearSnapshotsBtn");
  const snapshotCountEl = document.getElementById("snapshotCount");
  const sessionObjectCount = document.getElementById("sessionObjectCount");
  const sessionFrameCount = document.getElementById("sessionFrameCount");
  const view3DBtn = document.getElementById("view3DBtn");
  const selectedObjectInfo = document.getElementById("selectedObjectInfo");
  const selectedObjectName = document.getElementById("selectedObjectName");
  const detectedObjectsList = document.getElementById("detectedObjectsList");
  const chatMessages = document.getElementById("chatMessages");
  const chatInput = document.getElementById("chatInput");
  const sendChatBtn = document.getElementById("sendChatBtn");
  const clearChatBtn = document.getElementById("clearChatBtn");
  const typingIndicator = document.getElementById("typingIndicator");
  let cameraStream = null;
  let detectionActive = false;
  let detectionInterval = null;
  let currentDetections = [];
  let snapshots = [];
  let selectedObject = null;
  let frameCount = 0;
  let totalObjectsEver = 0;
  let isProcessing = false; // prevent overlapping API calls

  const CLASS_COLORS = {
    sunglasses: "#3b82f6",
    knife: "#ef4444",
    water_bottle: "#06b6d4",
    pen: "#f59e0b",
    chair: "#10b981",
    human_face: "#ec4899",
    mobile_phone: "#8b5cf6",
    helmet: "#f97316",
    fire: "#dc2626",
    can: "#14b8a6",
  };

  function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== "") {
      const cookies = document.cookie.split(";");
      for (let i = 0; i < cookies.length; i++) {
        const cookie = cookies[i].trim();
        if (cookie.substring(0, name.length + 1) === name + "=") {
          cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
          break;
        }
      }
    }
    return cookieValue;
  }

  startCameraBtn.addEventListener("click", async () => {
    try {
      cameraStream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720, facingMode: "user" },
      });
      videoElement.srcObject = cameraStream;
      cameraPlaceholder.classList.add("hidden");

      stopCameraBtn.disabled = false;
      toggleDetectionBtn.disabled = false;
      snapshotBtn.disabled = false;

      statusDot.classList.add("active");
      statusText.textContent = "Camera Active";

      videoElement.onloadedmetadata = () => {
        detectionCanvas.width = videoElement.videoWidth;
        detectionCanvas.height = videoElement.videoHeight;
      };

      addChatMessage("system", "Camera activated successfully.");
    } catch (e) {
      alert("Could not access camera: " + e.message);
    }
  });

  stopCameraBtn.addEventListener("click", () => {
    if (cameraStream) {
      cameraStream.getTracks().forEach((t) => t.stop());
      videoElement.srcObject = null;
      cameraPlaceholder.classList.remove("hidden");
      stopCameraBtn.disabled = true;
      toggleDetectionBtn.disabled = true;
      snapshotBtn.disabled = true;

      if (detectionActive) toggleDetection();

      statusDot.classList.remove("active", "detecting");
      statusText.textContent = "Ready";
      fpsBadge.style.display = "none";
      addChatMessage("system", "Camera stopped.");
    }
  });

  toggleDetectionBtn.addEventListener("click", () => toggleDetection());

  function toggleDetection() {
    detectionActive = !detectionActive;
    const btnText = document.getElementById("detectionBtnText");

    if (detectionActive) {
      btnText.textContent = "Stop Detection";
      toggleDetectionBtn.className = "btn btn-yellow";
      statusDot.classList.add("detecting");
      statusDot.classList.remove("active");
      statusText.textContent = "Detecting...";
      fpsBadge.style.display = "inline-block";
      startDetectionLoop();
      addChatMessage("system", "Object detection started.");
    } else {
      btnText.textContent = "Start Detection";
      toggleDetectionBtn.className = "btn btn-green";
      statusDot.classList.remove("detecting");
      statusDot.classList.add("active");
      statusText.textContent = "Camera Active";
      fpsBadge.style.display = "none";
      stopDetectionLoop();
      addChatMessage("system", "Object detection stopped.");
    }
  }
  function startDetectionLoop() {
    detectionInterval = setInterval(() => {
      if (!isProcessing) processFrame();
    }, 1000);
  }

  function stopDetectionLoop() {
    if (detectionInterval) {
      clearInterval(detectionInterval);
      detectionInterval = null;
    }
    ctx.clearRect(0, 0, detectionCanvas.width, detectionCanvas.height);
  }

  async function processFrame() {
    if (!detectionActive || !cameraStream) return;
    isProcessing = true;

    try {
      // Capture the current video frame to a canvas
      const tempCanvas = document.createElement("canvas");
      tempCanvas.width = videoElement.videoWidth;
      tempCanvas.height = videoElement.videoHeight;
      const tempCtx = tempCanvas.getContext("2d");
      tempCtx.drawImage(videoElement, 0, 0);

      const blob = await new Promise((resolve) =>
        tempCanvas.toBlob(resolve, "image/jpeg", 0.6),
      );
      const formData = new FormData();
      formData.append("image", blob, "frame.jpg");

      const response = await fetch("/api/detect/", {
        method: "POST",
        headers: { "X-CSRFToken": getCookie("csrftoken") },
        body: formData,
      });

      const data = await response.json();
      if (data.success && data.detections) {
        currentDetections = data.detections;
        frameCount++;
        totalObjectsEver += currentDetections.length;

        sessionFrameCount.textContent = frameCount;
        sessionObjectCount.textContent = totalObjectsEver;
        fpsBadge.textContent = `${currentDetections.length} obj`;

        drawDetections(currentDetections);
        updateDetectedObjectsList();
        autoSaveCrops(currentDetections);
      } else {
        currentDetections = [];
        ctx.clearRect(0, 0, detectionCanvas.width, detectionCanvas.height);
        updateDetectedObjectsList();
        frameCount++;
        sessionFrameCount.textContent = frameCount;
      }
    } catch (err) {
      console.error("Detection error:", err);
    } finally {
      isProcessing = false;
    }
  }

  const lastSavedAt = {};
  const SAVE_THROTTLE_MS = 30000; // 30 seconds per class

  function autoSaveCrops(detections) {
    if (!detections || detections.length === 0) return;

    // Pick highest-confidence detection per class
    const bestPerClass = {};
    detections.forEach((d) => {
      if (
        !bestPerClass[d.class] ||
        d.confidence > bestPerClass[d.class].confidence
      ) {
        bestPerClass[d.class] = d;
      }
    });

    Object.values(bestPerClass).forEach((det) => {
      const now = Date.now();
      if (
        lastSavedAt[det.class] &&
        now - lastSavedAt[det.class] < SAVE_THROTTLE_MS
      ) {
        return; // throttled
      }
      // Only save high-confidence detections to avoid junk crops
      if (det.confidence < 0.4) return;

      const cropDataUrl = cropDetectionFromVideo(det);
      if (!cropDataUrl) return;

      lastSavedAt[det.class] = now;

      fetch("/api/save-crop/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-CSRFToken": getCookie("csrftoken"),
        },
        body: JSON.stringify({
          object_class: det.class,
          confidence: det.confidence,
          crop_image: cropDataUrl,
        }),
      }).catch((err) => console.warn("Crop save failed:", err));
    });
  }

  function cropDetectionFromVideo(det) {
    try {
      if (!videoElement.videoWidth) return null;

      const tempCanvas = document.createElement("canvas");
      tempCanvas.width = videoElement.videoWidth;
      tempCanvas.height = videoElement.videoHeight;
      const tempCtx = tempCanvas.getContext("2d");

      // Draw mirrored frame
      tempCtx.save();
      tempCtx.scale(-1, 1);
      tempCtx.drawImage(
        videoElement,
        -tempCanvas.width,
        0,
        tempCanvas.width,
        tempCanvas.height,
      );
      tempCtx.restore();

      if (!det.bbox) {
        return tempCanvas.toDataURL("image/jpeg", 0.85);
      }

      const bbox = det.bbox;
      const x1 = Array.isArray(bbox) ? bbox[0] : bbox.x1;
      const y1 = Array.isArray(bbox) ? bbox[1] : bbox.y1;
      const x2 = Array.isArray(bbox) ? bbox[2] : bbox.x2;
      const y2 = Array.isArray(bbox) ? bbox[3] : bbox.y2;

      // Mirror x for mirrored video
      const mx1 = tempCanvas.width - x2;
      const mx2 = tempCanvas.width - x1;

      const padX = (mx2 - mx1) * 0.15;
      const padY = (y2 - y1) * 0.15;
      const cropX = Math.max(0, Math.floor(mx1 - padX));
      const cropY = Math.max(0, Math.floor(y1 - padY));
      const cropW = Math.min(
        tempCanvas.width - cropX,
        Math.ceil(mx2 - mx1 + padX * 2),
      );
      const cropH = Math.min(
        tempCanvas.height - cropY,
        Math.ceil(y2 - y1 + padY * 2),
      );

      if (cropW < 20 || cropH < 20) return null;

      const cropCanvas = document.createElement("canvas");
      cropCanvas.width = cropW;
      cropCanvas.height = cropH;
      cropCanvas
        .getContext("2d")
        .drawImage(tempCanvas, cropX, cropY, cropW, cropH, 0, 0, cropW, cropH);

      return cropCanvas.toDataURL("image/jpeg", 0.85);
    } catch (e) {
      console.warn("Crop error:", e);
      return null;
    }
  }

  function drawDetections(detections) {
    ctx.clearRect(0, 0, detectionCanvas.width, detectionCanvas.height);

    detections.forEach((det) => {
      const color = CLASS_COLORS[det.class] || "#3b82f6";
      const [x1, y1, x2, y2] = det.bbox || [0, 0, 0, 0];

      // Mirror the x coordinates to match the mirrored video
      const mx1 = detectionCanvas.width - x2;
      const mx2 = detectionCanvas.width - x1;
      const w = mx2 - mx1;
      const h = y2 - y1;

      // Draw box
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.strokeRect(mx1, y1, w, h);

      // Label with track ID and optional distance
      let label = "";
      if (det.track_id) {
        label += `#${det.track_id} `;
      }
      label +=
        det.class.replace("_", " ") +
        " " +
        (det.confidence * 100).toFixed(0) +
        "%";
      if (det.spatial_3d && det.spatial_3d.distance_m) {
        label += ` · ${det.spatial_3d.distance_m}m`;
      }

      ctx.font = "bold 13px Inter, sans-serif";
      const textWidth = ctx.measureText(label).width;

      ctx.fillStyle = color;
      ctx.fillRect(mx1, y1 - 24, textWidth + 12, 24);

      ctx.fillStyle = "#fff";
      ctx.fillText(label, mx1 + 6, y1 - 7);

      // Draw motion velocity indicator if active
      if (det.velocity && (Math.abs(det.velocity[0]) > 1.5 || Math.abs(det.velocity[1]) > 1.5)) {
        const cx = mx1 + w / 2;
        const cy = y1 + h / 2;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(cx - det.velocity[0] * 3, cy + det.velocity[1] * 3);
        ctx.strokeStyle = "#fbbf24";
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    });
  }
  function updateDetectedObjectsList() {
    if (currentDetections.length === 0) {
      detectedObjectsList.innerHTML = `
                    <div style="grid-column: 1 / -1; text-align: center; padding: 2rem; color: #92adc9;">
                        <span class="material-symbols-outlined" style="font-size: 3rem; opacity: 0.5;">search_off</span>
                        <p style="font-size: 0.875rem;">No objects detected yet</p>
                    </div>`;
      view3DBtn.disabled = true;
      selectedObjectInfo.classList.add("hidden");
      return;
    }

    view3DBtn.disabled = false;

    const uniqueMap = {};
    currentDetections.forEach((d) => {
      if (!uniqueMap[d.class] || d.confidence > uniqueMap[d.class].confidence) {
        uniqueMap[d.class] = d;
      }
    });
    const uniqueDetections = Object.values(uniqueMap);

    detectedObjectsList.innerHTML = uniqueDetections
      .map(
        (o) => {
          const trackLabel = o.track_id
            ? `<span style="font-size: 0.7rem; background: rgba(255,255,255,0.15); padding: 1px 5px; border-radius: 4px; margin-right: 4px;">#${o.track_id}</span>`
            : "";
          const distLabel =
            o.spatial_3d && o.spatial_3d.distance_m
              ? ` · ${o.spatial_3d.distance_m}m`
              : "";
          return `
                <div class="object-card ${selectedObject === o.class ? "selected" : ""}" data-object="${o.class}">
                    <span class="material-symbols-outlined" style="font-size: 2rem; color: ${CLASS_COLORS[o.class] || "#137fec"}; margin-bottom: 0.5rem;">verified</span>
                    <p style="font-size: 0.875rem; font-weight: 600; text-transform: capitalize; margin-bottom: 0.25rem;">${trackLabel}${o.class.replace("_", " ")}</p>
                    <p style="font-size: 0.75rem; color: #92adc9;">${(o.confidence * 100).toFixed(0)}%${distLabel}</p>
                </div>`;
        },
      )
      .join("");

    // Click to select for 3D
    document.querySelectorAll(".object-card").forEach((card) => {
      card.addEventListener("click", () => {
        const obj = card.dataset.object;
        selectObjectFor3D(obj);
        document
          .querySelectorAll(".object-card")
          .forEach((c) => c.classList.remove("selected"));
        card.classList.add("selected");
      });
    });
  }

  function selectObjectFor3D(objectClass) {
    selectedObject = objectClass;
    selectedObjectInfo.classList.remove("hidden");
    selectedObjectName.textContent = objectClass.replace("_", " ");
  }

  view3DBtn.addEventListener("click", () => {
    if (!selectedObject && currentDetections.length > 0) {
      selectedObject = currentDetections[0].class;
    }
    if (!selectedObject) {
      alert("Please detect and select an object first.");
      return;
    }

    // detection data for selected object
    const det = currentDetections.find((d) => d.class === selectedObject);
    const conf = det ? (det.confidence * 100).toFixed(0) : "";

    try {
      const tempCanvas = document.createElement("canvas");
      tempCanvas.width = videoElement.videoWidth;
      tempCanvas.height = videoElement.videoHeight;
      const tempCtx = tempCanvas.getContext("2d");

      // Draw mirrored frame (matching what user sees)
      tempCtx.save();
      tempCtx.scale(-1, 1);
      tempCtx.drawImage(
        videoElement,
        -tempCanvas.width,
        0,
        tempCanvas.width,
        tempCanvas.height,
      );
      tempCtx.restore();

      if (det && det.bbox) {
        // Crop to bounding box with some padding
        const bbox = det.bbox;
        const x1 =
          typeof bbox === "object" && !Array.isArray(bbox) ? bbox.x1 : bbox[0];
        const y1 =
          typeof bbox === "object" && !Array.isArray(bbox) ? bbox.y1 : bbox[1];
        const x2 =
          typeof bbox === "object" && !Array.isArray(bbox) ? bbox.x2 : bbox[2];
        const y2 =
          typeof bbox === "object" && !Array.isArray(bbox) ? bbox.y2 : bbox[3];

        // Mirror x coordinates for the mirrored video
        const mirroredX1 = tempCanvas.width - x2;
        const mirroredX2 = tempCanvas.width - x1;

        // Add 15% padding around the crop
        const padX = (mirroredX2 - mirroredX1) * 0.15;
        const padY = (y2 - y1) * 0.15;
        const cropX = Math.max(0, Math.floor(mirroredX1 - padX));
        const cropY = Math.max(0, Math.floor(y1 - padY));
        const cropW = Math.min(
          tempCanvas.width - cropX,
          Math.ceil(mirroredX2 - mirroredX1 + padX * 2),
        );
        const cropH = Math.min(
          tempCanvas.height - cropY,
          Math.ceil(y2 - y1 + padY * 2),
        );

        // Create cropped canvas
        const cropCanvas = document.createElement("canvas");
        cropCanvas.width = cropW;
        cropCanvas.height = cropH;
        cropCanvas
          .getContext("2d")
          .drawImage(
            tempCanvas,
            cropX,
            cropY,
            cropW,
            cropH,
            0,
            0,
            cropW,
            cropH,
          );

        // Store cropped image in sessionStorage
        const croppedBase64 = cropCanvas.toDataURL("image/jpeg", 0.9);
        sessionStorage.setItem("smartvision_crop_image", croppedBase64);
        sessionStorage.setItem("smartvision_crop_class", selectedObject);
      } else {
        // No bbox, store full frame
        const fullBase64 = tempCanvas.toDataURL("image/jpeg", 0.9);
        sessionStorage.setItem("smartvision_crop_image", fullBase64);
        sessionStorage.setItem("smartvision_crop_class", selectedObject);
      }
    } catch (err) {
      console.error("Error cropping detection:", err);
      // Continue to navigate even if crop fails
    }

    // Navigate to model viewer with params
    // Save crop to database so it works from dashboard too
    var cropData = sessionStorage.getItem("smartvision_crop_image");
    if (cropData) {
      fetch("/api/save-crop/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-CSRFToken": getCookie("csrftoken"),
        },
        body: JSON.stringify({
          object_class: selectedObject,
          confidence: det ? det.confidence : 0,
          crop_image: cropData,
        }),
      }).catch(function (err) {
        console.error("Save crop error:", err);
      });
    }

    // Navigate to model viewer
    window.location.href = `/model-viewer/?object=${selectedObject}&confidence=${conf}`;
  });

  // SNAPSHOTS
  snapshotBtn.addEventListener("click", () => takeSnapshot());

  function takeSnapshot() {
    // Flash effect
    snapshotFlash.classList.remove("hidden");
    snapshotFlash.classList.add("snapshot-flash");
    setTimeout(() => {
      snapshotFlash.classList.add("hidden");
      snapshotFlash.classList.remove("snapshot-flash");
    }, 300);

    // Capture mirrored video
    const canvas = document.createElement("canvas");
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;
    const snapCtx = canvas.getContext("2d");
    snapCtx.save();
    snapCtx.scale(-1, 1);
    snapCtx.drawImage(
      videoElement,
      -canvas.width,
      0,
      canvas.width,
      canvas.height,
    );
    snapCtx.restore();

    // Overlay bounding boxes
    if (detectionActive && currentDetections.length > 0) {
      snapCtx.drawImage(detectionCanvas, 0, 0);
    }

    const snap = {
      id: Date.now(),
      image: canvas.toDataURL("image/png"),
      timestamp: new Date().toLocaleString(),
      detections: [...currentDetections],
    };
    snapshots.unshift(snap);
    updateSnapshotGallery();
    updateSnapshotCount();
    addChatMessage(
      "system",
      `Snapshot captured with ${currentDetections.length} object(s).`,
    );
  }

  function updateSnapshotGallery() {
    if (snapshots.length === 0) {
      snapshotGallery.innerHTML = `
                    <div style="width: 100%; text-align: center; padding: 2rem;">
                        <span class="material-symbols-outlined" style="font-size: 3rem; opacity: 0.5;">photo_library</span>
                        <p style="font-size: 0.875rem; color: #92adc9;">No snapshots yet</p>
                    </div>`;
      return;
    }
    snapshotGallery.innerHTML = snapshots
      .map(
        (s) => `
                <div class="snapshot-item">
                    <div class="snapshot-container">
                        <img src="${s.image}" style="width: 100%; height: 128px; object-fit: cover;" alt="Snapshot" />
                        <div class="snapshot-overlay">
                            <button onclick="window._downloadSnap(${s.id})" class="snap-action-btn" style="background: #137fec;">
                                <span class="material-symbols-outlined">download</span>
                            </button>
                            <button onclick="window._deleteSnap(${s.id})" class="snap-action-btn" style="background: #dc2626;">
                                <span class="material-symbols-outlined">delete</span>
                            </button>
                        </div>
                        <div style="padding: 0.5rem;">
                            <p style="font-size: 0.75rem; color: #92adc9;">${s.timestamp}</p>
                            <p style="font-size: 0.75rem;">${s.detections.length} object(s)</p>
                        </div>
                    </div>
                </div>`,
      )
      .join("");
  }

  function updateSnapshotCount() {
    snapshotCountEl.textContent = snapshots.length;
  }

  // Expose to inline onclick handlers
  window._downloadSnap = function (id) {
    const s = snapshots.find((x) => x.id === id);
    if (s) {
      const link = document.createElement("a");
      link.download = `smartvision-${s.id}.png`;
      link.href = s.image;
      link.click();
    }
  };
  window._deleteSnap = function (id) {
    snapshots = snapshots.filter((s) => s.id !== id);
    updateSnapshotGallery();
    updateSnapshotCount();
  };

  clearSnapshotsBtn.addEventListener("click", () => {
    if (
      snapshots.length > 0 &&
      confirm(`Delete all ${snapshots.length} snapshots?`)
    ) {
      snapshots = [];
      updateSnapshotGallery();
      updateSnapshotCount();
      addChatMessage("system", "All snapshots cleared.");
    }
  });

  //LLM Inegration
  function addChatMessage(sender, msg) {
    const time = new Date().toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
    });
    let html = "";

    if (sender === "user") {
      html = `
                    <div class="message user">
                        <div class="message-content">
                            <div class="message-bubble"><p>${escapeHtml(msg)}</p></div>
                            <p style="font-size: 0.75rem; color: #92adc9; margin-top: 0.25rem; text-align: right;">${time}</p>
                        </div>
                        <div class="message-avatar"><span class="material-symbols-outlined" style="font-size: 0.875rem;">person</span></div>
                    </div>`;
    } else if (sender === "ai") {
      html = `
                    <div class="message">
                        <div class="message-avatar"><span class="material-symbols-outlined" style="font-size: 0.875rem;">psychology</span></div>
                        <div class="message-content">
                            <div class="message-bubble"><p>${formatAIResponse(msg)}</p></div>
                            <p style="font-size: 0.75rem; color: #92adc9; margin-top: 0.25rem;">${time}</p>
                        </div>
                    </div>`;
    } else {
      html = `
                    <div class="message" style="justify-content: center;">
                        <p style="font-size: 0.75rem; color: #92adc9; background: #192633; display: inline-block; padding: 0.25rem 0.75rem; border-radius: 9999px;">${escapeHtml(msg)}</p>
                    </div>`;
    }

    chatMessages.insertAdjacentHTML("beforeend", html);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }

  function formatAIResponse(text) {
    return escapeHtml(text)
      .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
      .replace(/\n/g, "<br>");
  }

  async function sendChatMessage() {
    const msg = chatInput.value.trim();
    if (!msg) return;

    addChatMessage("user", msg);
    chatInput.value = "";
    chatInput.disabled = true;
    sendChatBtn.disabled = true;
    typingIndicator.classList.remove("hidden");
    chatMessages.scrollTop = chatMessages.scrollHeight;

    try {
      const response = await fetch("/api/chat/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-CSRFToken": getCookie("csrftoken"),
        },
        body: JSON.stringify({
          message: msg,
          detections: currentDetections,
        }),
      });

      const data = await response.json();
      typingIndicator.classList.add("hidden");

      if (data.success) {
        addChatMessage("ai", data.response);
      } else {
        addChatMessage(
          "ai",
          "Sorry, I encountered an error. Please try again.",
        );
      }
    } catch (err) {
      console.error("Chat error:", err);
      typingIndicator.classList.add("hidden");
      addChatMessage(
        "ai",
        "Connection error. Please check your network and try again.",
      );
    } finally {
      chatInput.disabled = false;
      sendChatBtn.disabled = false;
      chatInput.focus();
    }
  }

  sendChatBtn.addEventListener("click", sendChatMessage);
  chatInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") sendChatMessage();
  });

  // Quick action buttons
  document.querySelectorAll(".quick-action-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      chatInput.value = btn.dataset.msg;
      sendChatMessage();
    });
  });

  // Clear chat
  clearChatBtn.addEventListener("click", () => {
    if (confirm("Clear chat history?")) {
      chatMessages.innerHTML = `
                    <div class="message">
                        <div class="message-avatar"><span class="material-symbols-outlined" style="font-size: 0.875rem;">psychology</span></div>
                        <div class="message-content">
                            <div class="message-bubble"><p>Chat cleared. How can I help you?</p></div>
                            <p style="font-size: 0.75rem; color: #92adc9; margin-top: 0.25rem;">Just now</p>
                        </div>
                    </div>`;
    }
  });
})();
