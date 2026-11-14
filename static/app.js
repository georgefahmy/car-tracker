const videoEl = document.getElementById("videoStream");
const zoneCanvas = document.getElementById("zoneCanvas");
const ctx = zoneCanvas.getContext("2d");

let savedZones = [];
let savedLines = [];
let drawingMode = null; // null | "zone" | "line"
let draftPoints = [];
let heatmapEnabled = false;

function syncCanvasSize() {
  const rect = videoEl.getBoundingClientRect();
  if (rect.width > 0 && rect.height > 0) {
    zoneCanvas.width = rect.width;
    zoneCanvas.height = rect.height;
    renderZoneCanvas();
  }
}

window.addEventListener("resize", syncCanvasSize);
videoEl.addEventListener("load", syncCanvasSize);

function renderZoneCanvas() {
  const w = zoneCanvas.width;
  const h = zoneCanvas.height;
  ctx.clearRect(0, 0, w, h);

  if (!drawingMode) return;

  // Draw existing zones lightly while editing
  for (const z of savedZones) {
    if (z.points.length < 3) continue;
    ctx.beginPath();
    z.points.forEach(([nx, ny], idx) => {
      if (idx === 0) ctx.moveTo(nx * w, ny * h);
      else ctx.lineTo(nx * w, ny * h);
    });
    ctx.closePath();
    ctx.strokeStyle = z.enabled ? "rgba(239, 68, 68, 0.7)" : "rgba(148, 163, 184, 0.5)";
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  // Draw existing lines lightly while editing
  for (const ln of savedLines) {
    ctx.beginPath();
    ctx.moveTo(ln.p1[0] * w, ln.p1[1] * h);
    ctx.lineTo(ln.p2[0] * w, ln.p2[1] * h);
    ctx.strokeStyle = ln.enabled ? "rgba(56, 189, 248, 0.75)" : "rgba(148, 163, 184, 0.5)";
    ctx.lineWidth = 3;
    ctx.stroke();
  }

  // Draw active draft polygon or line
  if (draftPoints.length > 0) {
    ctx.beginPath();
    draftPoints.forEach(([nx, ny], idx) => {
      if (idx === 0) ctx.moveTo(nx * w, ny * h);
      else ctx.lineTo(nx * w, ny * h);
    });
    ctx.strokeStyle = drawingMode === "line" ? "#38bdf8" : "#facc15";
    ctx.lineWidth = 2.5;
    ctx.stroke();

    for (const [nx, ny] of draftPoints) {
      ctx.beginPath();
      ctx.arc(nx * w, ny * h, 5, 0, Math.PI * 2);
      ctx.fillStyle = drawingMode === "line" ? "#38bdf8" : "#ef4444";
      ctx.fill();
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  }
}

function setDrawingMode(mode) {
  drawingMode = mode;
  draftPoints = [];
  const active = Boolean(mode);
  zoneCanvas.classList.toggle("drawing", active);
  const hint = document.getElementById("drawHint");
  hint.classList.toggle("hidden", !active);
  if (mode === "line") {
    hint.textContent = "Click 2 points on the video to place a traffic counting line (start and end).";
  } else {
    hint.textContent = 'Click on the video to place polygon corners around the red zone. Double-click or press "Finish & Save" when done.';
  }
  document.getElementById("btnDrawZone").classList.toggle("hidden", active);
  document.getElementById("btnDrawLine").classList.toggle("hidden", active);
  document.getElementById("btnUndoPoint").classList.toggle("hidden", !active);
  document.getElementById("btnFinishZone").classList.toggle("hidden", mode !== "zone");
  document.getElementById("btnCancelDraw").classList.toggle("hidden", !active);
  syncCanvasSize();
}

zoneCanvas.addEventListener("click", (e) => {
  if (!drawingMode) return;
  const rect = zoneCanvas.getBoundingClientRect();
  const nx = Math.min(Math.max((e.clientX - rect.left) / rect.width, 0), 1);
  const ny = Math.min(Math.max((e.clientY - rect.top) / rect.height, 0), 1);
  draftPoints.push([Number(nx.toFixed(4)), Number(ny.toFixed(4))]);
  renderZoneCanvas();
  if (drawingMode === "line" && draftPoints.length === 2) {
    finishLine();
  }
});

zoneCanvas.addEventListener("dblclick", () => {
  if (drawingMode === "zone" && draftPoints.length >= 3) {
    finishZone();
  }
});

document.getElementById("btnDrawZone").addEventListener("click", () => setDrawingMode("zone"));
document.getElementById("btnDrawLine").addEventListener("click", () => setDrawingMode("line"));
document.getElementById("btnCancelDraw").addEventListener("click", () => setDrawingMode(null));
document.getElementById("btnUndoPoint").addEventListener("click", () => {
  draftPoints.pop();
  renderZoneCanvas();
});
document.getElementById("btnFinishZone").addEventListener("click", finishZone);

async function finishLine() {
  if (draftPoints.length !== 2) return;
  const name = prompt("Name this traffic line:", `Traffic Line ${savedLines.length + 1}`);
  if (!name) {
    setDrawingMode(null);
    return;
  }
  const updated = [
    ...savedLines,
    {
      id: `line_${Date.now()}`,
      name: name.trim(),
      p1: draftPoints[0],
      p2: draftPoints[1],
      enabled: true,
      color: "#38bdf8",
      count_a_to_b: 0,
      count_b_to_a: 0,
    },
  ];
  await saveLines(updated);
  setDrawingMode(null);
}

async function saveLines(lines) {
  const res = await fetch("/api/lines", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(lines),
  });
  savedLines = await res.json();
  renderLinesList();
}

async function loadLines() {
  const res = await fetch("/api/lines");
  savedLines = await res.json();
  renderLinesList();
}

function renderLinesList() {
  const el = document.getElementById("linesList");
  if (!savedLines.length) {
    el.innerHTML = `<span class="muted">No traffic lines defined yet. Click "+ Draw Traffic Line" to place a 2-point counting line.</span>`;
    return;
  }
  el.innerHTML = "";
  for (const ln of savedLines) {
    const total = (ln.count_a_to_b || 0) + (ln.count_b_to_a || 0);
    const chip = document.createElement("div");
    chip.className = "zone-chip";
    chip.innerHTML = `
      <span>${ln.enabled ? "🔵" : "⚪"} ${escapeHtml(ln.name)}: <strong>${total}</strong> (${ln.count_a_to_b || 0} / ${ln.count_b_to_a || 0})</span>
      <button title="Toggle">${ln.enabled ? "Disable" : "Enable"}</button>
      <button title="Delete">✕</button>
    `;
    const [toggleBtn, delBtn] = chip.querySelectorAll("button");
    toggleBtn.addEventListener("click", () => {
      const next = savedLines.map((item) =>
        item.id === ln.id ? { ...item, enabled: !item.enabled } : item
      );
      saveLines(next);
    });
    delBtn.addEventListener("click", () => {
      saveLines(savedLines.filter((item) => item.id !== ln.id));
    });
    el.appendChild(chip);
  }
}

async function finishZone() {
  if (draftPoints.length < 3) {
    alert("Please place at least 3 points to form a zone.");
    return;
  }
  const name = prompt("Name this red zone:", `Red Zone ${savedZones.length + 1}`);
  if (!name) return;

  const updated = [
    ...savedZones,
    {
      id: `zone_${Date.now()}`,
      name: name.trim(),
      points: draftPoints,
      enabled: true,
      color: "#ef4444",
    },
  ];
  await saveZones(updated);
  setDrawingMode(false);
}

async function saveZones(zones) {
  const res = await fetch("/api/zones", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(zones),
  });
  savedZones = await res.json();
  renderZonesList();
}

async function loadZones() {
  const res = await fetch("/api/zones");
  savedZones = await res.json();
  renderZonesList();
}

function renderZonesList() {
  const el = document.getElementById("zonesList");
  if (!savedZones.length) {
    el.innerHTML = `<span class="muted">No red zones defined yet. Click "+ Draw Red Zone" to outline one.</span>`;
    return;
  }
  el.innerHTML = "";
  for (const z of savedZones) {
    const chip = document.createElement("div");
    chip.className = "zone-chip";
    chip.innerHTML = `
      <span>${z.enabled ? "🔴" : "⚪"} ${escapeHtml(z.name)}</span>
      <button title="Toggle">${z.enabled ? "Disable" : "Enable"}</button>
      <button title="Delete">✕</button>
    `;
    const [toggleBtn, delBtn] = chip.querySelectorAll("button");
    toggleBtn.addEventListener("click", () => {
      const next = savedZones.map((item) =>
        item.id === z.id ? { ...item, enabled: !item.enabled } : item
      );
      saveZones(next);
    });
    delBtn.addEventListener("click", () => {
      saveZones(savedZones.filter((item) => item.id !== z.id));
    });
    el.appendChild(chip);
  }
}

async function refreshStats() {
  try {
    const res = await fetch("/api/stats");
    const data = await res.json();

    document.getElementById("kpiActiveCars").textContent = data.active_cars;
    document.getElementById("kpiUniqueCars").textContent = data.unique_cars;
    document.getElementById("kpiTotalCars").textContent = data.total_cars;
    document.getElementById("kpiPeople").textContent = `${data.active_people} / ${data.unique_people}`;
    document.getElementById("kpiViolations").textContent = data.violations_total;
    if (Array.isArray(data.counting_lines)) {
      savedLines = data.counting_lines;
      renderLinesList();
      const totalCrossings = savedLines.reduce(
        (acc, ln) => acc + (ln.count_a_to_b || 0) + (ln.count_b_to_a || 0),
        0
      );
      document.getElementById("kpiLineCrossings").textContent = totalCrossings;
    }

    const dot = document.getElementById("statusDot");
    dot.className = "dot";
    if (data.stream_status === "connected") dot.classList.add("connected");
    else if (data.stream_status === "error") dot.classList.add("error");

    const resStr = data.resolution?.width ? `${data.resolution.width}x${data.resolution.height}` : "—";
    document.getElementById("streamInfo").textContent =
      `${data.stream_status.toUpperCase()} • ${resStr} • ${data.fps} FPS • ${data.source}`;

    heatmapEnabled = Boolean(data.heatmap_enabled);
    document.getElementById("btnHeatmapToggle").textContent = `Heatmap: ${heatmapEnabled ? "ON" : "OFF"}`;

    drawTrafficChart(data.hourly_traffic || []);
  } catch (err) {
    console.error("Stats poll error:", err);
  }
}

function drawTrafficChart(rows) {
  const canvas = document.getElementById("trafficChart");
  const c = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  c.clearRect(0, 0, w, h);

  if (!rows.length) {
    c.fillStyle = "#8b9bb4";
    c.font = "13px sans-serif";
    c.fillText("Hourly traffic data will appear as objects are tracked.", 16, h / 2);
    return;
  }

  const maxVal = Math.max(1, ...rows.map((r) => Math.max(r.vehicles_count, r.people_count)));
  const padL = 28, padB = 24, padT = 12, padR = 10;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;
  const groupW = plotW / rows.length;
  const barW = Math.max(4, Math.min(16, groupW * 0.35));

  rows.forEach((r, idx) => {
    const x0 = padL + idx * groupW + groupW * 0.15;
    const vh = (r.vehicles_count / maxVal) * plotH;
    const ph = (r.people_count / maxVal) * plotH;

    c.fillStyle = "#22c55e";
    c.fillRect(x0, padT + plotH - vh, barW, vh);

    c.fillStyle = "#38bdf8";
    c.fillRect(x0 + barW + 2, padT + plotH - ph, barW, ph);

    if (rows.length <= 12 || idx % 2 === 0) {
      c.fillStyle = "#8b9bb4";
      c.font = "10px sans-serif";
      const label = r.hour_bucket.slice(11) + "h";
      c.fillText(label, x0, h - 6);
    }
  });
}

async function refreshViolations() {
  try {
    const res = await fetch("/api/violations?limit=30");
    const items = await res.json();
    const list = document.getElementById("violationsList");
    document.getElementById("violationCountLabel").textContent = `${items.length} shown`;

    if (!items.length) {
      list.innerHTML = `<p class="muted">No illegal parking violations recorded yet.</p>`;
      return;
    }

    list.innerHTML = "";
    for (const v of items) {
      const card = document.createElement("div");
      card.className = "violation-card";
      const timeStr = new Date(v.flagged_at).toLocaleString();
      card.innerHTML = `
        <img src="/snapshots/${encodeURIComponent(v.snapshot_filename)}" alt="Snapshot" />
        <div class="violation-meta">
          <div class="violation-title">${escapeHtml(v.vehicle_class.toUpperCase())} #${v.track_id} • ${escapeHtml(v.zone_name)}</div>
          <div class="muted">Dwell: ${v.dwell_seconds.toFixed(0)}s</div>
          <div class="muted">${escapeHtml(timeStr)}</div>
        </div>
        <button class="btn btn-ghost" title="Delete">✕</button>
      `;
      card.querySelector("img").addEventListener("click", () => {
        document.getElementById("snapshotTitle").textContent = `${v.vehicle_class.toUpperCase()} #${v.track_id} — ${v.zone_name}`;
        document.getElementById("snapshotFullImg").src = `/snapshots/${encodeURIComponent(v.snapshot_filename)}`;
        document.getElementById("snapshotModal").classList.remove("hidden");
      });
      card.querySelector("button").addEventListener("click", async () => {
        await fetch(`/api/violations/${v.id}`, { method: "DELETE" });
        refreshViolations();
        refreshStats();
      });
      list.appendChild(card);
    }
  } catch (err) {
    console.error("Violations poll error:", err);
  }
}

// Heatmap & Reset controls
document.getElementById("btnHeatmapToggle").addEventListener("click", async () => {
  const res = await fetch("/api/heatmap/toggle", { method: "POST" });
  const data = await res.json();
  heatmapEnabled = data.heatmap_enabled;
  document.getElementById("btnHeatmapToggle").textContent = `Heatmap: ${heatmapEnabled ? "ON" : "OFF"}`;
});

document.getElementById("btnHeatmapReset").addEventListener("click", async () => {
  await fetch("/api/heatmap/reset", { method: "POST" });
});

document.getElementById("btnResetStats").addEventListener("click", async () => {
  if (!confirm("Reset unique/total vehicle & people counters?")) return;
  await fetch("/api/stats/reset", { method: "POST" });
  refreshStats();
});

// Settings modal
const settingsModal = document.getElementById("settingsModal");
const settingsForm = document.getElementById("settingsForm");

document.getElementById("btnSettings").addEventListener("click", async () => {
  const res = await fetch("/api/config");
  const cfg = await res.json();
  for (const [key, val] of Object.entries(cfg)) {
    const input = settingsForm.elements[key];
    if (!input) continue;
    if (input.type === "checkbox") input.checked = Boolean(val);
    else input.value = val ?? "";
  }
  document.getElementById("testStreamResult").textContent = "";
  settingsModal.classList.remove("hidden");
});

document.getElementById("btnCloseSettings").addEventListener("click", () => settingsModal.classList.add("hidden"));
document.getElementById("btnCancelSettings").addEventListener("click", () => settingsModal.classList.add("hidden"));
document.getElementById("btnCloseSnapshot").addEventListener("click", () => document.getElementById("snapshotModal").classList.add("hidden"));

document.getElementById("btnTestStream").addEventListener("click", async () => {
  const out = document.getElementById("testStreamResult");
  out.textContent = "Testing connection...";
  const payload = formToPayload();
  const res = await fetch("/api/stream/test", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (data.ok) {
    out.textContent = `Connected (${data.width}x${data.height})`;
    out.style.color = "#22c55e";
  } else {
    out.textContent = data.error || "Connection failed";
    out.style.color = "#ef4444";
  }
});

settingsForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const payload = formToPayload();
  payload.heatmap_enabled = heatmapEnabled;
  await fetch("/api/config", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  settingsModal.classList.add("hidden");
  refreshStats();
});

function formToPayload() {
  const fd = new FormData(settingsForm);
  return {
    nvr_host: fd.get("nvr_host"),
    nvr_port: Number(fd.get("nvr_port")),
    nvr_channel: Number(fd.get("nvr_channel")),
    nvr_stream_type: fd.get("nvr_stream_type"),
    nvr_user: fd.get("nvr_user"),
    nvr_pass: fd.get("nvr_pass"),
    test_video_path: fd.get("test_video_path"),
    target_fps: Number(fd.get("target_fps")),
    yolo_imgsz: Number(fd.get("yolo_imgsz")),
    confidence_threshold: Number(fd.get("confidence_threshold")),
    preview_max_width: Number(fd.get("preview_max_width")),
    dwell_seconds: Number(fd.get("dwell_seconds")),
    movement_threshold: Number(fd.get("movement_threshold")),
    annotation_scale: Number(fd.get("annotation_scale")),
    exempt_emergency_vehicles: Boolean(settingsForm.elements["exempt_emergency_vehicles"]?.checked),
    emergency_min_saturation: Number(fd.get("emergency_min_saturation")),
    emergency_min_value: Number(fd.get("emergency_min_value")),
    emergency_strobe_swing: Number(fd.get("emergency_strobe_swing")),
    emergency_fire_red_ratio: Number(fd.get("emergency_fire_red_ratio")),
  };
}

function escapeHtml(str) {
  return String(str)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

loadZones();
loadLines();
refreshStats();
refreshViolations();
setInterval(refreshStats, 1500);
setInterval(refreshViolations, 5000);
