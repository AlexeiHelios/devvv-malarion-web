/* ═══════════════════════════════════════════════════════════
   MALARION — Frontend application
   ═══════════════════════════════════════════════════════════ */

// ── State ────────────────────────────────────────────────────────────
let selectedFile  = null;
let lastResponse  = null;
let currentMode   = 'standard';   // 'standard' | 'wsi'

// ── Mode switcher ─────────────────────────────────────────────────────
function setMode(mode) {
  currentMode = mode;
  document.getElementById('modeStandard').classList.toggle('active', mode === 'standard');
  document.getElementById('modeWsi').classList.toggle('active', mode === 'wsi');

  const infoEl  = document.getElementById('modeInfoText');
  const infoBox = document.getElementById('modeInfo');
  const iconEl  = document.getElementById('dropIcon');
  const textEl  = document.getElementById('dropText');

  // Reset info banner styling
  infoBox.style.borderColor = '';
  infoBox.style.color       = '';

  if (mode === 'standard') {
    infoEl.textContent = 'Single 960×960 field-of-view image. Full GradCAM + BV panels.';
    iconEl.textContent = '🔬';
    textEl.innerHTML   = 'Drag & drop a slide image<br/><small>.jpg · .png · .tiff</small>';
  } else {
    infoEl.textContent = 'Large or stitched slide image. Will be tiled into 960×960 patches automatically. Minimum 1920px on longest side.';
    iconEl.textContent = '🗂️';
    textEl.innerHTML   = 'Drag & drop a WSI image<br/><small>.jpg · .png · .tiff (large OK)</small>';
  }
}

// ── Species colours (hex, for charts + dots) ─────────────────────────
const SPECIES_HEX = {
  falciparum: "#E63946",
  vivax:      "#2A9D8F",
  ovale:      "#E9C46A",
  malariae:   "#6A4C93",
};
const STAGE_HEX = {
  R: "#FF6B6B",   // Ring
  T: "#4ECDC4",   // Trophozoite
  S: "#45B7D1",   // Schizont
  G: "#96CEB4",   // Gametocyte
};
const STAGE_LABELS = { R: "Ring", T: "Trophozoite", S: "Schizont", G: "Gametocyte" };

// ── Model selector ───────────────────────────────────────────────────
document.querySelectorAll(".model-option").forEach(opt => {
  opt.addEventListener("click", () => {
    document.querySelectorAll(".model-option").forEach(o => o.classList.remove("selected"));
    opt.classList.add("selected");
    opt.querySelector("input").checked = true;
    // Hide BV threshold slider for models that don't use BV
    const modelId = parseInt(opt.dataset.id);
    document.getElementById("bvThreshRow").style.display =
      [2, 3, 5].includes(modelId) ? "" : "none";
  });
});

// ── File handling ────────────────────────────────────────────────────
const dropzone   = document.getElementById("dropzone");
const fileInput  = document.getElementById("fileInput");
const previewImg = document.getElementById("previewImg");
const dropInner  = document.getElementById("dropInner");
const analyseBtn = document.getElementById("analyseBtn");
const clearBtn   = document.getElementById("clearBtn");
fileInput.addEventListener("change", e => handleFile(e.target.files[0]));

dropzone.addEventListener("dragover", e => {
  e.preventDefault();
  dropzone.classList.add("drag-over");
});
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag-over"));
dropzone.addEventListener("drop", e => {
  e.preventDefault();
  dropzone.classList.remove("drag-over");
  const f = e.dataTransfer.files[0];
  if (f) handleFile(f);
});

function handleFile(file) {
  if (!file) return;
  if (!file.type.match(/image\/(jpeg|png|tiff?)/)) {
    alert("Please upload a JPEG, PNG or TIFF image.");
    return;
  }
  selectedFile = file;
  const url = URL.createObjectURL(file);
  previewImg.src = url;
  previewImg.style.display = "block";
  dropInner.style.display  = "none";
  analyseBtn.disabled      = false;
  clearBtn.disabled        = false;

  // Warn if WSI mode selected but image looks too small
  if (currentMode === 'wsi') {
    const img = new Image();
    img.onload = () => {
      if (Math.max(img.width, img.height) < 1920) {
        document.getElementById('modeInfoText').textContent =
          `⚠️ This image (${img.width}×${img.height}px) is too small for Whole Slide mode. ` +
          `Switch to Standard FOV mode instead.`;
        document.getElementById('modeInfo').style.borderColor = 'var(--red)';
        document.getElementById('modeInfo').style.color = 'var(--red2)';
      } else {
        document.getElementById('modeInfoText').textContent =
          `✓ Image size ${img.width}×${img.height}px — suitable for WSI tiling.`;
        document.getElementById('modeInfo').style.borderColor = 'var(--teal)';
        document.getElementById('modeInfo').style.color = 'var(--teal2)';
      }
      URL.revokeObjectURL(url);
    };
    img.src = url;
  }
}

function clearImage() {
  selectedFile = null;
  previewImg.src = '';
  previewImg.style.display = 'none';
  analyseBtn.disabled = true;
  clearBtn.disabled = true;
  fileInput.value = '';
  
  // Restore dropzone inner content
  dropInner.style.display = 'block';
  dropInner.innerHTML = `
    <span class="drop-icon" id="dropIcon">🔬</span>
    <p id="dropText">Drag &amp; drop a slide image<br/><small>.jpg · .png · .tiff</small></p>
    <button class="btn btn-outline" id="chooseFileBtn">
      Choose file
    </button>
  `;
  
  // Attach click handler to new button
  document.getElementById('chooseFileBtn').addEventListener('click', () => {
    fileInput.click();
  });
  
  // Update dropzone to current mode
  const iconEl = dropInner.querySelector('.drop-icon');
  const textEl = dropInner.querySelector('#dropText');
  
  if (currentMode === 'standard') {
    iconEl.textContent = '🔬';
    textEl.innerHTML = 'Drag &amp; drop a slide image<br/><small>.jpg · .png · .tiff</small>';
  } else {
    iconEl.textContent = '🗂️';
    textEl.innerHTML = 'Drag &amp; drop a WSI image<br/><small>.jpg · .png · .tiff (large OK)</small>';
  }
  
  // Reset mode info banner
  const infoBox = document.getElementById('modeInfo');
  infoBox.style.borderColor = '';
  infoBox.style.color = '';
}

// ── Tabs ─────────────────────────────────────────────────────────────
document.querySelectorAll(".tab").forEach(tab => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
    tab.classList.add("active");
    document.getElementById("tabAnnotated").style.display = "none";
    document.getElementById("tabGradcam").style.display   = "none";
    document.getElementById("tabBvpanels").style.display  = "none";
    const map = { annotated: "tabAnnotated", gradcam: "tabGradcam", bvpanels: "tabBvpanels" };
    document.getElementById(map[tab.dataset.tab]).style.display = "";
  });
});

// ── Main analysis ─────────────────────────────────────────────────────
async function runAnalysis() {
  if (!selectedFile) return;

  const modelId    = document.querySelector("input[name=model_id]:checked")?.value || "5";
  const confThresh = document.getElementById("conf_thresh").value;
  const iouThresh  = document.getElementById("iou_thresh").value;
  const bvThresh   = document.getElementById("bv_thresh").value;

  setView("loading");
  document.getElementById("loadingMsg").textContent =
    currentMode === 'wsi'
      ? "Tiling slide and running pipeline on each patch…"
      : "Running pipeline…";

  const fd = new FormData();
  fd.append("image",       selectedFile);
  fd.append("model_id",    modelId);
  fd.append("conf_thresh", confThresh);
  fd.append("iou_thresh",  iouThresh);
  fd.append("bv_thresh",   bvThresh);

  const endpoint = currentMode === 'wsi' ? '/api/predict_wsi' : '/api/predict';

  let data;
  try {
    const res = await fetch(endpoint, { method: "POST", body: fd });
    data = await res.json();
    if (!res.ok) {
      setView("empty");
      // Show friendly message for WSI size error
      if (res.status === 400 && data.required_min) {
        alert(
          `Image too small for Whole Slide mode.\n\n` +
          `Your image: ${data.image_size}px\n` +
          `Required minimum: ${data.required_min}px on longest side\n\n` +
          `Switch to Standard FOV mode to analyze this image.`
        );
      } else {
        alert("Prediction failed: " + (data.error || `HTTP ${res.status}`));
      }
      return;
    }
  } catch (err) {
    setView("empty");
    alert("Prediction failed: " + err.message);
    return;
  }

  lastResponse = data;
  renderResults(data);
  setView("results");

  // Hide Gemini card for WSI mode; show prompt for standard mode
  if (currentMode === 'standard') {
    document.getElementById("geminiCard").style.display = "";
    // Hide the report and show the prompt/buttons instead
    document.getElementById("geminiPrompt").style.display = "";
    document.getElementById("geminiLoading").style.display = "none";
    document.getElementById("geminiResult").style.display = "none";
    document.getElementById("geminiError").style.display = "none";
    document.getElementById("generateReportBtn").disabled = false;
    document.getElementById("downloadReportBtn").disabled = true;
  } else {
    // Hide Gemini card for WSI mode
    document.getElementById("geminiCard").style.display = "none";
  }
}

// ── Generate XAI Report (on-demand) ──────────────────────────────────
async function generateXaiReport() {
  if (!lastResponse) {
    alert("No analysis results available. Please run analysis first.");
    return;
  }

  const generateBtn = document.getElementById("generateReportBtn");
  const geminiLoading = document.getElementById("geminiLoading");
  const geminiResult  = document.getElementById("geminiResult");
  const geminiError   = document.getElementById("geminiError");
  const geminiPrompt  = document.getElementById("geminiPrompt");

  generateBtn.disabled = true;
  geminiPrompt.style.display = "none";
  geminiLoading.style.display = "";
  geminiResult.style.display  = "none";
  geminiError.style.display   = "none";

  try {
    const body = {
      yolo_result: lastResponse.yolo_result || {
        det_boxes_xyxy: [],
        det_cls: [],
        det_conf: []
      },
      slide_report: lastResponse.slide_report || {},
      detections: lastResponse.detections || [],
      pipeline_name: lastResponse.pipeline_name || "Unknown",
      image_b64: lastResponse.images?.annotated || null,
    };

    const res = await fetch("/api/generate_xai_report", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(body),
    });

    const result = await res.json();
    if (!res.ok) throw new Error(result.error || `HTTP ${res.status}`);

    // Store report for download
    window.currentReportText = result.report_text;
    
    const xai = result.gemini_xai;
    const sections = xai.sections || xai;

    const sectionDefs = [
      { id: "gs1", key: "slide_assessment",  label: "1. Slide Assessment" },
      { id: "gs2", key: "detection_quality", label: "2. Detection Quality" },
      { id: "gs3", key: "bv_filter_effect",  label: "3. BV Filter Effect" },
      { id: "gs4", key: "clinical_verdict",  label: "4. Clinical Verdict" },
    ];

    sectionDefs.forEach(({ id, key, label }) => {
      const el = document.getElementById(id);
      el.innerHTML = `
        <div class="gemini-section-label">${label}</div>
        <div class="gemini-section-body">${escHtml(sections[key] || "[No content]")}</div>
      `;
    });

    // Show buttons and report
    geminiPrompt.style.display = "";
    geminiLoading.style.display = "none";
    geminiResult.style.display  = "";
    document.getElementById("downloadReportBtn").disabled = false;
    generateBtn.disabled = false;

    if (xai.status === "no_key") {
      geminiError.textContent    = "Gemini API key not set (GEMINI_API_KEY env var). XAI narrative disabled.";
      geminiError.style.display  = "";
      geminiResult.style.display = "none";
    }
  } catch (err) {
    console.error("XAI Report error:", err);
    geminiPrompt.style.display = "";
    geminiLoading.style.display = "none";
    geminiError.textContent     = "Report generation failed: " + err.message;
    geminiError.style.display   = "";
    generateBtn.disabled = false;
  }
}

// ── Download XAI Report ──────────────────────────────────────────────
function downloadXaiReport() {
  if (!window.currentReportText) {
    alert("No report available. Please generate a report first.");
    return;
  }

  try {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
    const filename = `malarion_report_${timestamp}.txt`;
    
    // Create blob and download
    const blob = new Blob([window.currentReportText], { type: 'text/plain; charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    link.click();
    URL.revokeObjectURL(url);
  } catch (err) {
    alert("Download failed: " + err.message);
  }
}

// ── Render results ────────────────────────────────────────────────────
function renderResults(data) {
  const sr     = data.slide_report;
  const isWsi  = data.mode === 'wsi';
  const verdict = sr.slide_verdict;

  // Show/hide WSI block
  document.getElementById('wsiBlock').style.display    = isWsi ? '' : 'none';
  document.getElementById('geminiCard').style.display  = isWsi ? 'none' : '';

  // Show/hide image tabs that don't apply to WSI
  document.querySelector('.tab[data-tab="gradcam"]').style.display  = isWsi ? 'none' : '';
  document.querySelector('.tab[data-tab="bvpanels"]').style.display = isWsi ? 'none' : '';
  if (isWsi) {
    // Switch active tab to annotated
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelector('.tab[data-tab="annotated"]').classList.add('active');
    document.getElementById('tabAnnotated').style.display = '';
    document.getElementById('tabGradcam').style.display   = 'none';
    document.getElementById('tabBvpanels').style.display  = 'none';
  }

  // Verdict banner
  const banner = document.getElementById("verdictBanner");
  banner.className = "verdict-banner " + verdict;
  document.getElementById("verdictIcon").textContent  = verdict === "infected" ? "🦠" : "✅";
  document.getElementById("verdictTitle").textContent = verdict === "infected" ? "INFECTED" : "HEALTHY";
  document.getElementById("verdictSub").textContent   =
    (isWsi ? "WSI · " : "") + `Pipeline: ${data.pipeline_name}`;
  document.getElementById("statRaw").textContent    = sr.raw_count;
  document.getElementById("statKept").textContent   = sr.validated_count;
  document.getElementById("statSpecies").textContent =
    Object.keys(sr.species_summary || {}).length || 0;
  
  // Update kept label based on whether BV filtering is used
  const keptLabel = document.getElementById("keptLabel");
  keptLabel.textContent = data.uses_bv ? "Kept (BV)" : "Detections";

  // Threshold chips
  const threshRow = document.getElementById("threshRow");
  threshRow.innerHTML = "";
  Object.entries(sr.threshold_predictions || {}).forEach(([key, val]) => {
    const n    = key.replace("thresh_", "≥ ");
    const chip = document.createElement("span");
    chip.className   = "thresh-chip " + (val === "infected" ? "thresh-infected" : "thresh-healthy");
    chip.textContent = `${n} detections → ${val.toUpperCase()}`;
    threshRow.appendChild(chip);
  });

  // ── WSI-specific block ────────────────────────────────────────────
  if (isWsi) {
    const wsiStats = document.getElementById('wsiStats');
    wsiStats.innerHTML = `
      <div class="wsi-stat"><span>${sr.total_tiles || 0}</span><small>Total tiles</small></div>
      <div class="wsi-stat"><span>${sr.infected_tiles || 0}</span><small>Infected tiles</small></div>
      <div class="wsi-stat"><span>${sr.healthy_tiles || 0}</span><small>Clean tiles</small></div>
      <div class="wsi-stat"><span>${sr.validated_count}</span><small>Confirmed parasites</small></div>
    `;
    if (data.images?.wsi_density_overlay)
      document.getElementById('imgWsiDensity').src =
        'data:image/jpeg;base64,' + data.images.wsi_density_overlay;
    if (data.images?.wsi_verdict_grid)
      document.getElementById('imgWsiGrid').src =
        'data:image/jpeg;base64,' + data.images.wsi_verdict_grid;
  }

  // Annotated image (standard) or thumbnail (WSI — show density overlay in annotated tab)
  if (isWsi && data.images?.wsi_density_overlay) {
    document.getElementById("imgAnnotated").src =
      'data:image/jpeg;base64,' + data.images.wsi_density_overlay;
  } else if (data.images?.annotated) {
    document.getElementById("imgAnnotated").src =
      "data:image/jpeg;base64," + data.images.annotated;
  }

  // YOLO GradCAM (standard only)
  if (!isWsi && data.images?.yolo_gradcam) {
    document.getElementById("imgGradcam").src =
      "data:image/jpeg;base64," + data.images.yolo_gradcam;
  }

  // BV GradCAM panels (standard only)
  if (!isWsi) {
    const panels     = data.images?.bv_gradcam_panels || [];
    const bvGrid     = document.getElementById("bvPanelGrid");
    const bvTab      = document.querySelector(".tab[data-tab='bvpanels']");
    bvGrid.innerHTML = "";
    if (panels.length === 0) {
      bvTab.style.opacity = "0.4";
      bvGrid.innerHTML = "<p style='color:var(--text2);padding:1rem'>No BV GradCAM panels.</p>";
    } else {
      bvTab.style.opacity = "1";
      panels.forEach(p => {
        const div = document.createElement("div");
        div.className = "bv-panel";
        div.innerHTML = `
          <div class="bv-panel-header">
            <span>${escHtml(p.class_name)} · det ${p.detection_index}</span>
            <span>BV conf: ${p.bv_conf.toFixed(3)}</span>
          </div>
          <div class="bv-panel-imgs">
            <div><img src="data:image/jpeg;base64,${p.crop_original}" alt="crop"/>
              <span>Original crop</span></div>
            <div><img src="data:image/jpeg;base64,${p.bv_gradcam}" alt="bv cam"/>
              <span>BV GradCAM</span></div>
          </div>`;
        bvGrid.appendChild(div);
      });
    }
  }

  // Per-class table
  const classNames = sr.class_names || [];
  const rawPc      = sr.raw_count_per_class || [];
  const valPc      = sr.validated_count_per_class || [];
  const usesBv     = data.uses_bv === true;
  const thead      = document.getElementById("classTableHead");
  const tbody      = document.getElementById("classTableBody");
  
  // Update header based on uses_bv flag
  const headerCells = usesBv 
    ? '<th>Class</th><th>Species</th><th>Stage</th><th>Raw dets</th><th>BV-kept</th><th>BV-kept %</th>'
    : '<th>Class</th><th>Species</th><th>Stage</th><th>Detections</th>';
  
  thead.innerHTML = `<tr>${headerCells}</tr>`;
  
  tbody.innerHTML  = "";
  classNames.forEach((cls, i) => {
    const raw = rawPc[i] || 0;
    const val = valPc[i] || 0;
    if (raw === 0 && val === 0) return;
    const species = cls.split("_")[0];
    const stage   = cls.split("_")[1] || "?";
    const color   = SPECIES_HEX[species] || "#888";
    const tr = document.createElement("tr");
    
    let rowHtml = `
      <td><span class="species-dot" style="background:${color}"></span>${escHtml(cls)}</td>
      <td style="color:${color}">${capitalize(species)}</td>
      <td>${STAGE_LABELS[stage] || stage}</td>`;
    
    if (usesBv) {
      const pct = raw > 0 ? Math.round(100 * val / raw) : 0;
      rowHtml += `
      <td>${raw}</td>
      <td>${val}</td>
      <td>${pct}%<span class="pct-bar" style="width:${pct*0.6}px;background:${color}"></span></td>`;
    } else {
      rowHtml += `<td>${val}</td>`;
    }
    
    tr.innerHTML = rowHtml;
    tbody.appendChild(tr);
  });

  // Species / stage charts
  renderBarChart("speciesChart", sr.species_summary, SPECIES_HEX, s => capitalize(s));
  const stageByLabel = {};
  const stageColors  = {};
  Object.entries(STAGE_LABELS).forEach(([k, v]) => { stageColors[v] = STAGE_HEX[k] || "#888"; });
  Object.entries(sr.stage_summary || {}).forEach(([k, v]) => {
    stageByLabel[STAGE_LABELS[k] || k] = v;
  });
  renderBarChart("stageChart", stageByLabel, stageColors, s => s);

  // Detection list
  const detList = document.getElementById("detectionList");
  detList.innerHTML = "";
  const dets = data.detections || [];
  if (dets.length === 0) {
    detList.innerHTML = "<p style='color:var(--text2);padding:0.5rem 0'>No detections.</p>";
  } else {
    // For WSI show max 50 to avoid DOM overload
    const toShow = isWsi ? dets.slice(0, 50) : dets;
    toShow.forEach(det => {
      const div     = document.createElement("div");
      const kept    = det.bv_kept;
      const species = det.species || det.class_name?.split("_")[0] || "unknown";
      const color   = SPECIES_HEX[species] || "#888";
      const usesBv  = data.uses_bv === true;
      div.className = "det-item " + (kept ? "det-kept" : "det-filter");
      div.innerHTML = `
        <span class="det-idx">#${det.index !== undefined ? det.index : det.tile_idx ?? ""}</span>
        <span class="det-cls" style="color:${color}">${escHtml(det.class_name)}</span>
        <span class="det-conf">YOLO ${det.yolo_conf?.toFixed(3)}</span>
        ${usesBv && det.bv_conf > 0 ? `<span class="det-conf">BV ${det.bv_conf.toFixed(3)}</span>` : ""}
        ${det.xai_category ? `<span class="det-cat cat-${det.xai_category}">${det.xai_category}</span>` : ""}
        ${isWsi && det.tile_idx !== undefined ? `<span style="color:var(--text3);font-size:0.7rem">tile ${det.tile_idx}</span>` : ""}
        <span style="color:var(--text3);font-size:0.7rem">${kept ? "✓ kept" : "✗ filtered"}</span>`;
      detList.appendChild(div);
    });
    if (isWsi && dets.length > 50) {
      const more = document.createElement("p");
      more.style.cssText = "color:var(--text3);font-size:0.75rem;padding:0.4rem 0";
      more.textContent   = `… and ${dets.length - 50} more detections across all tiles`;
      detList.appendChild(more);
    }
  }
}

// ── Bar chart helper ─────────────────────────────────────────────────
function renderBarChart(containerId, dataObj, colorMap, labelFn) {
  const container = document.getElementById(containerId);
  container.innerHTML = "";
  const entries = Object.entries(dataObj || {}).sort((a, b) => b[1] - a[1]);
  const max     = Math.max(...entries.map(e => e[1]), 1);
  entries.forEach(([key, val]) => {
    const color = colorMap[key] || "#6c5ce7";
    const pct   = Math.round(100 * val / max);
    const row   = document.createElement("div");
    row.className = "bc-row";
    row.innerHTML = `
      <span class="bc-label">${escHtml(labelFn(key))}</span>
      <div class="bc-bar-wrap">
        <div class="bc-bar" style="width:${pct}%;background:${color}"></div>
      </div>
      <span class="bc-count">${val}</span>`;
    container.appendChild(row);
  });
  if (entries.length === 0) {
    container.innerHTML = "<p style='color:var(--text3);font-size:0.78rem'>No data</p>";
  }
}

// ── View state manager ───────────────────────────────────────────────
function setView(state) {
  document.getElementById("emptyState").style.display   = state === "empty"   ? "" : "none";
  document.getElementById("loadingState").style.display = state === "loading" ? "" : "none";
  document.getElementById("resultContent").style.display = state === "results" ? "" : "none";
}

// ── Utilities ────────────────────────────────────────────────────────
function escHtml(str) {
  return String(str)
    .replace(/&/g,"&amp;").replace(/</g,"&lt;")
    .replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}
function capitalize(s) { return s ? s[0].toUpperCase() + s.slice(1) : ""; }
