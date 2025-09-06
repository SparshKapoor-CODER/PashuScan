// static/app.js
let lastResult = null;

document.addEventListener("DOMContentLoaded", () => {
  const imageInput = document.getElementById("imageInput");
  const previewImg = document.getElementById("previewImg");
  const noPreview = document.getElementById("noPreview");
  const predictBtn = document.getElementById("predictBtn");
  const predList = document.getElementById("predList");
  const measureText = document.getElementById("measureText");
  const topkInput = document.getElementById("topk");
  const calibInput = document.getElementById("calib");
  const saveBtn = document.getElementById("saveBtn");
  const savedMsg = document.getElementById("savedMsg");

  let currentFile = null;

  imageInput.addEventListener("change", (ev) => {
    const file = ev.target.files[0];
    if (!file) return;
    currentFile = file;
    const url = URL.createObjectURL(file);
    previewImg.src = url;
    previewImg.style.display = "block";
    noPreview.style.display = "none";
    // reset UI
    predList.innerHTML = "";
    measureText.textContent = "No measurements yet.";
    savedMsg.textContent = "";
    saveBtn.disabled = true;
  });

  predictBtn.addEventListener("click", async () => {
    if (!currentFile) {
      alert("Select an image first.");
      return;
    }
    predictBtn.disabled = true;
    predictBtn.textContent = "Processing...";
    savedMsg.textContent = "";

    const fd = new FormData();
    fd.append("image", currentFile);
    fd.append("topk", topkInput.value || "3");
    const calibVal = calibInput.value.trim();
    if (calibVal) fd.append("calib_cm_per_px", calibVal);

    try {
      const r = await fetch("/predict", { method: "POST", body: fd });
      if (!r.ok) {
        const txt = await r.text();
        throw new Error(txt || r.statusText);
      }
      const data = await r.json();
      lastResult = data;

      // show predictions
      predList.innerHTML = "";
      if (data.predictions && data.predictions.length) {
        for (const p of data.predictions) {
          const li = document.createElement("li");
          li.innerHTML = `<strong>${p.breed}</strong> — ${(p.confidence*100).toFixed(2)}%`;
          predList.appendChild(li);
        }
      } else {
        predList.innerHTML = "<li>No prediction (model not loaded)</li>";
      }

      // show measurements
      if (data.measurements) {
        let txt = "";
        for (const key of Object.keys(data.measurements)) {
          txt += `${key}: ${JSON.stringify(data.measurements[key])}\n`;
        }
        measureText.textContent = txt;
      } else {
        measureText.textContent = "Measurements not available.";
      }

      // update preview to served URL (so saves & reloads use same file)
      if (data.image_url) {
        previewImg.src = data.image_url + "?_=" + Date.now();
      }
      saveBtn.disabled = false;
    } catch (err) {
      alert("Error: " + err.message);
    } finally {
      predictBtn.disabled = false;
      predictBtn.textContent = "Predict & Measure";
    }
  });

  saveBtn.addEventListener("click", async () => {
    if (!lastResult) return;
    saveBtn.disabled = true;
    savedMsg.textContent = "Saving...";
    // build classification field from top prediction
    let classification = {};
    if (lastResult.predictions && lastResult.predictions.length) {
      classification.breed = lastResult.predictions[0].breed;
      classification.is_purebred = false;
    }

    const payload = {
      image_url: lastResult.image_url,
      classification: classification,
      measurements: lastResult.measurements || {},
      metadata: { saved_from: "web_app" }
    };

    try {
      const r = await fetch("/save_evaluation", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await r.json();
      if (r.ok) {
        savedMsg.textContent = "Saved: " + data.saved_path;
      } else {
        savedMsg.textContent = "Save failed: " + (data.error || "unknown");
      }
    } catch (err) {
      savedMsg.textContent = "Save error: " + err.message;
    } finally {
      saveBtn.disabled = false;
    }
  });
});
