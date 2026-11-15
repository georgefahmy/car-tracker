const videoEl = document.getElementById("videoStream");
const zoneCanvas = document.getElementById("zoneCanvas");
const ctx = zoneCanvas.getContext("2d");

let savedZones = [];
let savedLines = [];
let drawingMode = null; // null | "zone" | "exclusion" | "line" | "edit" | "calibrate"
let draftPoints = [];
let editingTarget = null; // null | { kind: "zone" | "line", id: string, name: string }
let dragVertexIndex = -1;
let heatmapEnabled = false;
let groundCalib = null;
let groundToImgMatrix = null;

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

let currentObjectUrl = null;
async function pollFrameLoop() {
  while (true) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 4000);
    try {
      const res = await fetch(`api/stream/frame.jpg?t=${Date.now()}`, {
        cache: "no-store",
        signal: controller.signal,
      });
      if (res.ok) {
        const blob = await res.blob();
        const nextUrl = URL.createObjectURL(blob);
        if (currentObjectUrl) URL.revokeObjectURL(currentObjectUrl);
        currentObjectUrl = nextUrl;
        videoEl.src = nextUrl;
      }
    } catch (_) {
      // Transient timeout or server restart; retry cleanly
    } finally {
      clearTimeout(timer);
    }
    await new Promise((r) => setTimeout(r, 40));
  }
}

function projectGroundToCanvas(xFt, yFt, w, h) {
  if (!groundToImgMatrix) return null;
  const m = groundToImgMatrix;
  const denom = m[2][0] * xFt + m[2][1] * yFt + m[2][2];
  if (Math.abs(denom) < 1e-6 || denom <= 0) return null;
  const nx = (m[0][0] * xFt + m[0][1] * yFt + m[0][2]) / denom;
  const ny = (m[1][0] * xFt + m[1][1] * yFt + m[1][2]) / denom;
  if (!Number.isFinite(nx) || !Number.isFinite(ny)) return null;
  return [nx * w, ny * h];
}

function drawPerspectiveGroundGrid(w, h) {
  if (!groundCalib || !groundToImgMatrix) return;
  const wFt = groundCalib.width_ft;
  const lFt = groundCalib.length_ft;
  const step = Math.max(groundCalib.grid_step_ft || 5, 1);
  const minX = -wFt * 2;
  const maxX = wFt * 3;
  const minY = -lFt * 2;
  const maxY = lFt * 4;

  ctx.save();
  ctx.strokeStyle = "rgba(34, 211, 238, 0.35)";
  ctx.lineWidth = 1;
  for (let x = minX; x <= maxX + 1e-3; x += step) {
    ctx.beginPath();
    let started = false;
    for (let y = minY; y <= maxY + 1e-3; y += step * 0.5) {
      const pt = projectGroundToCanvas(x, y, w, h);
      if (!pt || pt[0] < -w || pt[0] > w * 2 || pt[1] < -h || pt[1] > h * 2) {
        started = false;
        continue;
      }
      if (!started) {
        ctx.moveTo(pt[0], pt[1]);
        started = true;
      } else {
        ctx.lineTo(pt[0], pt[1]);
      }
    }
    ctx.stroke();
  }
  for (let y = minY; y <= maxY + 1e-3; y += step) {
    ctx.beginPath();
    let started = false;
    for (let x = minX; x <= maxX + 1e-3; x += step * 0.5) {
      const pt = projectGroundToCanvas(x, y, w, h);
      if (!pt || pt[0] < -w || pt[0] > w * 2 || pt[1] < -h || pt[1] > h * 2) {
        started = false;
        continue;
      }
      if (!started) {
        ctx.moveTo(pt[0], pt[1]);
        started = true;
      } else {
        ctx.lineTo(pt[0], pt[1]);
      }
    }
    ctx.stroke();
  }

  // Highlight calibrated reference rectangle
  if (groundCalib.points?.length === 4) {
    ctx.beginPath();
    groundCalib.points.forEach(([nx, ny], idx) => {
      if (idx === 0) ctx.moveTo(nx * w, ny * h);
      else ctx.lineTo(nx * w, ny * h);
    });
    ctx.closePath();
    ctx.strokeStyle = "rgba(34, 211, 238, 0.85)";
    ctx.lineWidth = 2;
    ctx.stroke();
  }
  ctx.restore();
}

function renderZoneCanvas() {
  const w = zoneCanvas.width;
  const h = zoneCanvas.height;
  ctx.clearRect(0, 0, w, h);

  if (groundCalib && (groundCalib.show_grid || drawingMode)) {
    drawPerspectiveGroundGrid(w, h);
  }

  if (!drawingMode) return;

  // Draw existing zones lightly while editing
  for (const z of savedZones) {
    if (editingTarget && editingTarget.kind === "zone" && editingTarget.id === z.id) continue;
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
    if (editingTarget && editingTarget.kind === "line" && editingTarget.id === ln.id) continue;
    ctx.beginPath();
    ctx.moveTo(ln.p1[0] * w, ln.p1[1] * h);
    ctx.lineTo(ln.p2[0] * w, ln.p2[1] * h);
    ctx.strokeStyle = ln.enabled ? "rgba(56, 189, 248, 0.75)" : "rgba(148, 163, 184, 0.5)";
    ctx.lineWidth = 3;
    ctx.stroke();
  }

  // Draw active draft polygon or line
  if (draftPoints.length > 0) {
    const isLine = drawingMode === "line" || (drawingMode === "edit" && editingTarget?.kind === "line");
    ctx.beginPath();
    draftPoints.forEach(([nx, ny], idx) => {
      if (idx === 0) ctx.moveTo(nx * w, ny * h);
      else ctx.lineTo(nx * w, ny * h);
    });
    if (drawingMode === "edit" && !isLine && draftPoints.length >= 3) {
      ctx.closePath();
      ctx.fillStyle = "rgba(250, 204, 21, 0.12)";
      ctx.fill();
    }
    ctx.strokeStyle = isLine ? "#38bdf8" : "#facc15";
    ctx.lineWidth = 2.5;
    ctx.stroke();

    const handleRadius =
      drawingMode === "edit"
        ? (window.matchMedia("(pointer: coarse)").matches ? 9 : 7)
        : 5;
    draftPoints.forEach(([nx, ny], idx) => {
      const px = nx * w;
      const py = ny * h;
      ctx.beginPath();
      ctx.arc(px, py, handleRadius, 0, Math.PI * 2);
      ctx.fillStyle = isLine ? "#38bdf8" : "#ef4444";
      ctx.fill();
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 1.8;
      ctx.stroke();
      if (drawingMode === "edit" && isLine) {
        ctx.fillStyle = "#fff";
        ctx.font = "bold 11px sans-serif";
        ctx.fillText(idx === 0 ? "A" : "B", px + 10, py - 8);
      }
    });
  }
}

function setDrawingMode(mode, target = null) {
  drawingMode = mode;
  editingTarget = target;
  dragVertexIndex = -1;
  if (!target) {
    draftPoints = [];
  }
  const active = Boolean(mode);
  const isEdit = mode === "edit";
  zoneCanvas.classList.toggle("drawing", active);
  zoneCanvas.classList.toggle("editing", isEdit);
  const hint = document.getElementById("drawHint");
  hint.classList.toggle("hidden", !active);
  if (isEdit && target?.kind === "line") {
    hint.textContent = `Editing "${target.name}": Drag endpoint A or B to reposition, click Rename to rename, then Finish & Save.`;
  } else if (isEdit) {
    hint.textContent = `Editing "${target?.name || "Zone"}": Drag corners to move, click an edge to add a corner, right-click a corner to remove, then Finish & Save.`;
  } else if (mode === "calibrate") {
    hint.textContent = "Click 4 ground corners of a known rectangle (Top-Left → Top-Right → Bottom-Right → Bottom-Left) to calibrate 3D perspective.";
  } else if (mode === "line") {
    hint.textContent = "Click 2 points on the video to place a traffic counting line (start and end).";
  } else if (mode === "exclusion") {
    hint.textContent = 'Click on the video to outline an exclusion mask (e.g. balcony/plants to ignore). Double-click or press "Finish & Save".';
  } else {
    hint.textContent = 'Click on the video to place polygon corners around the red zone. Double-click or press "Finish & Save" when done.';
  }
  document.getElementById("btnDrawZone").classList.toggle("hidden", active);
  document.getElementById("btnAutoRedZone").classList.toggle("hidden", active);
  document.getElementById("btnDrawExclusion").classList.toggle("hidden", active);
  document.getElementById("btnDrawLine").classList.toggle("hidden", active);
  document.getElementById("btnCalibrateGround").classList.toggle("hidden", active);
  document.getElementById("btnUndoPoint").classList.toggle("hidden", !active || isEdit);
  document.getElementById("btnRenameEdit").classList.toggle("hidden", !isEdit);
  document.getElementById("btnFinishZone").classList.toggle("hidden", mode !== "zone" && mode !== "exclusion" && !isEdit);
  document.getElementById("btnCancelDraw").classList.toggle("hidden", !active);
  syncCanvasSize();
}

function getNormalizedPointer(e) {
  const rect = zoneCanvas.getBoundingClientRect();
  const nx = Math.min(Math.max((e.clientX - rect.left) / rect.width, 0), 1);
  const ny = Math.min(Math.max((e.clientY - rect.top) / rect.height, 0), 1);
  return {
    nx: Number(nx.toFixed(4)),
    ny: Number(ny.toFixed(4)),
    px: nx * zoneCanvas.width,
    py: ny * zoneCanvas.height,
  };
}

function findNearestVertex(px, py, maxDist = 15) {
  const w = zoneCanvas.width;
  const h = zoneCanvas.height;
  let bestIdx = -1;
  let bestDist = maxDist;
  draftPoints.forEach(([nx, ny], idx) => {
    const d = Math.hypot(nx * w - px, ny * h - py);
    if (d <= bestDist) {
      bestDist = d;
      bestIdx = idx;
    }
  });
  return bestIdx;
}

function findNearestEdgeIndex(px, py, maxDist = 14) {
  if (draftPoints.length < 2) return -1;
  const w = zoneCanvas.width;
  const h = zoneCanvas.height;
  let bestEdge = -1;
  let bestDist = maxDist;
  for (let i = 0; i < draftPoints.length; i++) {
    const j = (i + 1) % draftPoints.length;
    const ax = draftPoints[i][0] * w;
    const ay = draftPoints[i][1] * h;
    const bx = draftPoints[j][0] * w;
    const by = draftPoints[j][1] * h;
    const dx = bx - ax;
    const dy = by - ay;
    const lenSq = dx * dx + dy * dy;
    const t = lenSq > 0 ? Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / lenSq)) : 0;
    const d = Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
    if (d <= bestDist) {
      bestDist = d;
      bestEdge = i;
    }
  }
  return bestEdge;
}

zoneCanvas.addEventListener("pointerdown", (e) => {
  if (drawingMode !== "edit" || e.button !== 0) return;
  const { nx, ny, px, py } = getNormalizedPointer(e);
  const isTouch = e.pointerType === "touch";
  const vIdx = findNearestVertex(px, py, isTouch ? 28 : 15);
  if (vIdx >= 0) {
    dragVertexIndex = vIdx;
    zoneCanvas.setPointerCapture(e.pointerId);
    return;
  }
  if (editingTarget?.kind === "zone") {
    const edgeIdx = findNearestEdgeIndex(px, py, isTouch ? 24 : 14);
    if (edgeIdx >= 0) {
      draftPoints.splice(edgeIdx + 1, 0, [nx, ny]);
      dragVertexIndex = edgeIdx + 1;
      zoneCanvas.setPointerCapture(e.pointerId);
      renderZoneCanvas();
    }
  }
});

zoneCanvas.addEventListener("pointermove", (e) => {
  if (drawingMode !== "edit" || dragVertexIndex < 0) return;
  const { nx, ny } = getNormalizedPointer(e);
  draftPoints[dragVertexIndex] = [nx, ny];
  renderZoneCanvas();
});

zoneCanvas.addEventListener("pointerup", (e) => {
  if (dragVertexIndex >= 0) {
    dragVertexIndex = -1;
    if (zoneCanvas.hasPointerCapture(e.pointerId)) {
      zoneCanvas.releasePointerCapture(e.pointerId);
    }
  }
});

zoneCanvas.addEventListener("pointercancel", () => {
  dragVertexIndex = -1;
});

zoneCanvas.addEventListener("contextmenu", (e) => {
  if (drawingMode !== "edit" || editingTarget?.kind !== "zone") return;
  e.preventDefault();
  if (draftPoints.length <= 3) return;
  const { px, py } = getNormalizedPointer(e);
  const vIdx = findNearestVertex(px, py);
  if (vIdx >= 0) {
    draftPoints.splice(vIdx, 1);
    renderZoneCanvas();
  }
});

zoneCanvas.addEventListener("click", (e) => {
  if (!drawingMode || drawingMode === "edit") return;
  const { nx, ny } = getNormalizedPointer(e);
  draftPoints.push([nx, ny]);
  renderZoneCanvas();
  if (drawingMode === "line" && draftPoints.length === 2) {
    finishLine();
  } else if (drawingMode === "calibrate" && draftPoints.length === 4) {
    finishCalibration();
  }
});

zoneCanvas.addEventListener("dblclick", () => {
  if ((drawingMode === "zone" || drawingMode === "exclusion") && draftPoints.length >= 3) {
    finishZone();
  }
});

document.getElementById("btnDrawZone").addEventListener("click", () => setDrawingMode("zone"));
document.getElementById("btnAutoRedZone").addEventListener("click", autoDetectRedCurb);
document.getElementById("btnDrawExclusion").addEventListener("click", () => setDrawingMode("exclusion"));
document.getElementById("btnDrawLine").addEventListener("click", () => setDrawingMode("line"));
document.getElementById("btnCalibrateGround").addEventListener("click", () => setDrawingMode("calibrate"));
document.getElementById("btnCancelDraw").addEventListener("click", () => setDrawingMode(null));
document.getElementById("btnUndoPoint").addEventListener("click", () => {
  draftPoints.pop();
  renderZoneCanvas();
});
document.getElementById("btnRenameEdit").addEventListener("click", () => {
  if (drawingMode !== "edit" || !editingTarget) return;
  const nextName = prompt("Rename:", editingTarget.name);
  if (nextName && nextName.trim()) {
    editingTarget.name = nextName.trim();
    setDrawingMode("edit", editingTarget);
  }
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
  const res = await fetch("api/lines", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(lines),
  });
  savedLines = await res.json();
  renderLinesList();
}

async function loadLines() {
  const res = await fetch("api/lines");
  savedLines = await res.json();
  renderLinesList();
}

function startEditLine(ln) {
  draftPoints = [[...ln.p1], [...ln.p2]];
  setDrawingMode("edit", { kind: "line", id: ln.id, name: ln.name });
}

function startEditZone(z) {
  draftPoints = z.points.map((pt) => [...pt]);
  setDrawingMode("edit", { kind: "zone", id: z.id, name: z.name });
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
      <button title="Edit">Edit</button>
      <button title="Toggle">${ln.enabled ? "Disable" : "Enable"}</button>
      <button title="Delete">✕</button>
    `;
    const [editBtn, toggleBtn, delBtn] = chip.querySelectorAll("button");
    editBtn.addEventListener("click", () => startEditLine(ln));
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
  if (drawingMode === "edit" && editingTarget) {
    if (editingTarget.kind === "line") {
      if (draftPoints.length !== 2) return;
      const updated = savedLines.map((item) =>
        item.id === editingTarget.id
          ? { ...item, name: editingTarget.name, p1: draftPoints[0], p2: draftPoints[1] }
          : item
      );
      await saveLines(updated);
      setDrawingMode(null);
      return;
    }
    if (draftPoints.length < 3) {
      alert("Please keep at least 3 points for a zone.");
      return;
    }
    const updated = editingTarget.id
      ? savedZones.map((item) =>
          item.id === editingTarget.id
            ? { ...item, name: editingTarget.name, points: draftPoints }
            : item
        )
      : [
          ...savedZones,
          {
            id: `zone_${Date.now()}`,
            name: editingTarget.name,
            points: draftPoints,
            enabled: true,
            color: "#ef4444",
            zone_type: "red_zone",
          },
        ];
    await saveZones(updated);
    setDrawingMode(null);
    return;
  }
  if (draftPoints.length < 3) {
    alert("Please place at least 3 points to form a zone.");
    return;
  }
  const isExclusion = drawingMode === "exclusion";
  const defaultName = isExclusion ? `Exclusion Mask ${savedZones.length + 1}` : `Red Zone ${savedZones.length + 1}`;
  const name = prompt(isExclusion ? "Name this exclusion mask:" : "Name this red zone:", defaultName);
  if (!name) return;

  const updated = [
    ...savedZones,
    {
      id: `zone_${Date.now()}`,
      name: name.trim(),
      points: draftPoints,
      enabled: true,
      color: isExclusion ? "#94a3b8" : "#ef4444",
      zone_type: isExclusion ? "exclusion" : "red_zone",
    },
  ];
  await saveZones(updated);
  setDrawingMode(null);
}

async function autoDetectRedCurb() {
  const btn = document.getElementById("btnAutoRedZone");
  const prevText = btn.textContent;
  btn.disabled = true;
  btn.textContent = "Scanning curb...";
  try {
    const res = await fetch("api/zones/auto-detect", { method: "POST" });
    const data = await res.json();
    if (!res.ok) {
      alert(data.detail || "Auto-detect failed.");
      return;
    }
    if (!data.detected || !data.zones?.length) {
      alert(data.message || "No red curb detected in current frame.");
      return;
    }
    const first = data.zones[0];
    draftPoints = first.points.map((pt) => [...pt]);
    setDrawingMode("edit", {
      kind: "zone",
      id: null,
      name: `${first.name || "Auto Red Zone"} ${savedZones.length + 1}`,
    });
    const hint = document.getElementById("drawHint");
    hint.textContent = `${data.message} Drag any corner to adjust, then click Finish & Save.`;
  } catch (err) {
    alert(`Auto-detect error: ${err.message}`);
  } finally {
    btn.disabled = false;
    btn.textContent = prevText;
  }
}

async function saveZones(zones) {
  const res = await fetch("api/zones", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(zones),
  });
  savedZones = await res.json();
  renderZonesList();
}

async function loadZones() {
  const res = await fetch("api/zones");
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
    const icon = z.zone_type === "exclusion" ? "🩶" : "🔴";
    chip.innerHTML = `
      <span>${z.enabled ? icon : "⚪"} ${escapeHtml(z.name)}</span>
      <button title="Edit">Edit</button>
      <button title="Toggle">${z.enabled ? "Disable" : "Enable"}</button>
      <button title="Delete">✕</button>
    `;
    const [editBtn, toggleBtn, delBtn] = chip.querySelectorAll("button");
    editBtn.addEventListener("click", () => startEditZone(z));
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
    const res = await fetch("api/stats");
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
    if (data.stream_status.startsWith("connected")) dot.classList.add("connected");
    else if (data.stream_status === "error") dot.classList.add("error");

    const resStr = data.resolution?.width ? `${data.resolution.width}x${data.resolution.height}` : "—";
    const sys = data.system || {};
    const sysParts = [];
    if (sys.cpu_percent != null) sysParts.push(`CPU ${sys.cpu_percent}%`);
    if (sys.memory_percent != null) sysParts.push(`RAM ${sys.memory_percent}%`);
    if (sys.cpu_temp_c != null) sysParts.push(`${sys.cpu_temp_c}°C`);
    const sysStr = sysParts.length ? ` • ${sysParts.join(" • ")}` : "";
    const backendStr = data.inference_backend ? ` • ${String(data.inference_backend).toUpperCase()}` : "";
    document.getElementById("streamInfo").textContent =
      `${data.stream_status.toUpperCase()} • ${resStr} • ${data.fps} FPS${backendStr}${sysStr} • ${data.source}`;

    heatmapEnabled = Boolean(data.heatmap_enabled);
    document.getElementById("btnHeatmapToggle").textContent = `Heatmap: ${heatmapEnabled ? "ON" : "OFF"}`;

    const speedEl = document.getElementById("speedSummary");
    if (speedEl) {
      const sp = data.speed || {};
      if (!sp.calibrated) {
        speedEl.textContent = "Speed (2-line): Calibrate 3D Ground Grid and enable 2 traffic lines to measure mph.";
      } else if (sp.samples_count > 0) {
        const recentText = (sp.recent || [])
          .slice(0, 4)
          .map((r) => `#${r.track_id} ${r.vehicle_class} ${r.speed_mph} mph`)
          .join(" • ");
        speedEl.textContent = `Speed (2-line): Last ${sp.last_speed_mph} mph • Avg ${sp.avg_speed_mph} mph • Max ${sp.max_speed_mph} mph` + (recentText ? ` — ${recentText}` : "");
      } else {
        speedEl.textContent = "Speed (2-line): Waiting for a vehicle to cross 2 enabled traffic lines.";
      }
    }

    drawTrafficChart(data.hourly_traffic || []);
  } catch (err) {
    console.error("Stats poll error:", err);
  }
}

let lastTrafficRows = [];
let hoveredTrafficIdx = -1;

function drawTrafficChart(rows) {
  lastTrafficRows = rows;
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
  const padL = 28, padB = 24, padT = 16, padR = 10;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;
  const groupW = plotW / rows.length;
  const barW = Math.max(4, Math.min(16, groupW * 0.35));

  rows.forEach((r, idx) => {
    const gx = padL + idx * groupW;
    if (idx === hoveredTrafficIdx) {
      c.fillStyle = "rgba(148, 163, 184, 0.14)";
      c.fillRect(gx + 2, padT, Math.max(groupW - 4, 4), plotH);
    }

    const x0 = gx + groupW * 0.15;
    const vh = (r.vehicles_count / maxVal) * plotH;
    const ph = (r.people_count / maxVal) * plotH;

    c.fillStyle = idx === hoveredTrafficIdx ? "#4ade80" : "#22c55e";
    c.fillRect(x0, padT + plotH - vh, barW, vh);

    c.fillStyle = idx === hoveredTrafficIdx ? "#7dd3fc" : "#38bdf8";
    c.fillRect(x0 + barW + 2, padT + plotH - ph, barW, ph);

    if (rows.length <= 12 || idx % 2 === 0) {
      c.fillStyle = idx === hoveredTrafficIdx ? "#f8fafc" : "#8b9bb4";
      c.font = "10px sans-serif";
      const label = r.hour_bucket.slice(11) + "h";
      c.fillText(label, x0, h - 6);
    }
  });

  if (hoveredTrafficIdx >= 0 && hoveredTrafficIdx < rows.length) {
    const r = rows[hoveredTrafficIdx];
    const hourText = r.hour_bucket.replace("T", " ") + ":00";
    const line1 = hourText;
    const line2 = `Vehicles: ${r.vehicles_count}   People: ${r.people_count}`;
    c.font = "11px sans-serif";
    const boxW = Math.max(c.measureText(line1).width, c.measureText(line2).width) + 18;
    const boxH = 40;
    const rawX = padL + (hoveredTrafficIdx + 0.5) * groupW - boxW / 2;
    const boxX = Math.max(6, Math.min(w - boxW - 6, rawX));
    const boxY = 6;

    c.fillStyle = "rgba(15, 23, 42, 0.94)";
    c.strokeStyle = "#334155";
    c.lineWidth = 1;
    c.beginPath();
    c.roundRect(boxX, boxY, boxW, boxH, 6);
    c.fill();
    c.stroke();

    c.fillStyle = "#f8fafc";
    c.font = "bold 11px sans-serif";
    c.fillText(line1, boxX + 9, boxY + 16);
    c.fillStyle = "#cbd5e1";
    c.font = "11px sans-serif";
    c.fillText(line2, boxX + 9, boxY + 31);
  }
}

const trafficChartEl = document.getElementById("trafficChart");
function updateChartHoverFromEvent(e) {
  if (!lastTrafficRows.length) return;
  const rect = trafficChartEl.getBoundingClientRect();
  const x = (e.clientX - rect.left) * (trafficChartEl.width / Math.max(rect.width, 1));
  const padL = 28, padR = 10;
  const plotW = trafficChartEl.width - padL - padR;
  const idx = Math.floor(((x - padL) / plotW) * lastTrafficRows.length);
  const nextIdx = idx >= 0 && idx < lastTrafficRows.length ? idx : -1;
  if (nextIdx !== hoveredTrafficIdx) {
    hoveredTrafficIdx = nextIdx;
    drawTrafficChart(lastTrafficRows);
  }
}
trafficChartEl.addEventListener("pointermove", updateChartHoverFromEvent);
trafficChartEl.addEventListener("pointerdown", updateChartHoverFromEvent);
trafficChartEl.addEventListener("pointerleave", () => {
  if (hoveredTrafficIdx !== -1) {
    hoveredTrafficIdx = -1;
    drawTrafficChart(lastTrafficRows);
  }
});

async function refreshViolations() {
  try {
    const res = await fetch("api/violations?limit=30");
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
      const detailLine = v.zone_id === "speeding"
        ? `Speed: ${v.dwell_seconds.toFixed(1)} mph`
        : `Dwell: ${v.dwell_seconds.toFixed(0)}s`;
      card.innerHTML = `
        <img src="snapshots/${encodeURIComponent(v.snapshot_filename)}" alt="Snapshot" />
        <div class="violation-meta">
          <div class="violation-title">${escapeHtml(v.vehicle_class.toUpperCase())} #${v.track_id} • ${escapeHtml(v.zone_name)}</div>
          <div class="muted">${detailLine}</div>
          <div class="muted">${escapeHtml(timeStr)}</div>
        </div>
        <button class="btn btn-ghost" title="Delete">✕</button>
      `;
      card.querySelector("img").addEventListener("click", () => {
        document.getElementById("snapshotTitle").textContent = `${v.vehicle_class.toUpperCase()} #${v.track_id} — ${v.zone_name}`;
        document.getElementById("snapshotFullImg").src = `snapshots/${encodeURIComponent(v.snapshot_filename)}`;
        document.getElementById("snapshotModal").classList.remove("hidden");
      });
      card.querySelector("button").addEventListener("click", async () => {
        await fetch(`api/violations/${v.id}`, { method: "DELETE" });
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
  const res = await fetch("api/heatmap/toggle", { method: "POST" });
  const data = await res.json();
  heatmapEnabled = data.heatmap_enabled;
  document.getElementById("btnHeatmapToggle").textContent = `Heatmap: ${heatmapEnabled ? "ON" : "OFF"}`;
});

document.getElementById("btnHeatmapReset").addEventListener("click", async () => {
  await fetch("api/heatmap/reset", { method: "POST" });
});

document.getElementById("btnResetStats").addEventListener("click", async () => {
  if (!confirm("Reset unique/total vehicle & people counters?")) return;
  await fetch("api/stats/reset", { method: "POST" });
  refreshStats();
});

// Settings modal
const settingsModal = document.getElementById("settingsModal");
const settingsForm = document.getElementById("settingsForm");

document.getElementById("btnSettings").addEventListener("click", async () => {
  const res = await fetch("api/config");
  const cfg = await res.json();
  for (const [key, val] of Object.entries(cfg)) {
    const input = settingsForm.elements[key];
    if (!input) continue;
    if (input.type === "checkbox") input.checked = Boolean(val);
    else input.value = val ?? "";
  }
  document.getElementById("testStreamResult").textContent = "";
  document.getElementById("testWebhookResult").textContent = "";
  settingsModal.classList.remove("hidden");
});

document.getElementById("btnCloseSettings").addEventListener("click", () => settingsModal.classList.add("hidden"));
document.getElementById("btnCancelSettings").addEventListener("click", () => settingsModal.classList.add("hidden"));
document.getElementById("btnCloseSnapshot").addEventListener("click", () => document.getElementById("snapshotModal").classList.add("hidden"));

document.getElementById("btnTestStream").addEventListener("click", async () => {
  const out = document.getElementById("testStreamResult");
  out.textContent = "Testing connection...";
  const payload = formToPayload();
  const res = await fetch("api/stream/test", {
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

document.getElementById("btnTestWebhook").addEventListener("click", async () => {
  const out = document.getElementById("testWebhookResult");
  const url = settingsForm.elements["webhook_url"]?.value?.trim() || "";
  if (!url) {
    out.textContent = "Enter a webhook URL first.";
    out.style.color = "#f59e0b";
    return;
  }
  out.textContent = "Sending test alert...";
  out.style.color = "#8b9bb4";
  const res = await fetch("api/webhook/test", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ webhook_url: url }),
  });
  const data = await res.json();
  if (data.ok) {
    out.textContent = `Test alert sent (HTTP ${data.status_code})`;
    out.style.color = "#22c55e";
  } else {
    out.textContent = data.error || "Webhook test failed";
    out.style.color = "#ef4444";
  }
});

document.getElementById("btnSendDailyReport").addEventListener("click", async () => {
  const out = document.getElementById("testWebhookResult");
  const url = settingsForm.elements["webhook_url"]?.value?.trim() || "";
  if (!url) {
    out.textContent = "Enter a webhook URL first.";
    out.style.color = "#f59e0b";
    return;
  }
  out.textContent = "Sending daily report...";
  out.style.color = "#8b9bb4";
  const res = await fetch("api/report/daily/send", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ webhook_url: url }),
  });
  const data = await res.json();
  if (data.ok) {
    out.textContent = `Daily report sent (HTTP ${data.status_code})`;
    out.style.color = "#22c55e";
  } else {
    out.textContent = data.error || "Failed to send daily report";
    out.style.color = "#ef4444";
  }
});

settingsForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const payload = formToPayload();
  payload.heatmap_enabled = heatmapEnabled;
  await fetch("api/config", {
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
    stream_mode: fd.get("stream_mode"),
    nvr_user: fd.get("nvr_user"),
    nvr_pass: fd.get("nvr_pass"),
    test_video_path: fd.get("test_video_path"),
    yolo_model: fd.get("yolo_model") || "yolov8n.pt",
    inference_backend: fd.get("inference_backend") || "auto",
    target_fps: Number(fd.get("target_fps")),
    yolo_imgsz: Number(fd.get("yolo_imgsz")),
    confidence_threshold: Number(fd.get("confidence_threshold")),
    preview_max_width: Number(fd.get("preview_max_width")),
    dwell_seconds: Number(fd.get("dwell_seconds")),
    movement_threshold: Number(fd.get("movement_threshold")),
    annotation_scale: Number(fd.get("annotation_scale")),
    speed_capture_enabled: Boolean(settingsForm.elements["speed_capture_enabled"]?.checked),
    speed_limit_mph: Number(fd.get("speed_limit_mph")) || 25,
    exempt_emergency_vehicles: Boolean(settingsForm.elements["exempt_emergency_vehicles"]?.checked),
    emergency_min_saturation: Number(fd.get("emergency_min_saturation")),
    emergency_min_value: Number(fd.get("emergency_min_value")),
    emergency_strobe_swing: Number(fd.get("emergency_strobe_swing")),
    emergency_fire_red_ratio: Number(fd.get("emergency_fire_red_ratio")),
    webhook_url: fd.get("webhook_url") || "",
    daily_report_enabled: Boolean(settingsForm.elements["daily_report_enabled"]?.checked),
    daily_report_time: fd.get("daily_report_time") || "23:55",
  };
}

// --- 3D Ground Calibration & Bird's-Eye View ---

function updateCalibrationUI() {
  const toggleBtn = document.getElementById("btnToggleGrid");
  const bevBtn = document.getElementById("btnBirdseye");
  const hasCalib = Boolean(groundCalib);
  toggleBtn.classList.toggle("hidden", !hasCalib);
  bevBtn.classList.toggle("hidden", !hasCalib);
  if (hasCalib) {
    toggleBtn.textContent = `Grid: ${groundCalib.show_grid ? "ON" : "OFF"}`;
  }
  syncCanvasSize();
}

async function loadCalibration() {
  try {
    const res = await fetch("api/calibration");
    const data = await res.json();
    groundCalib = data.calibration || null;
    groundToImgMatrix = data.ground_to_img_norm || null;
    updateCalibrationUI();
  } catch (_) {}
}

async function finishCalibration() {
  if (draftPoints.length !== 4) return;
  const wStr = prompt("Real-world width (P1 → P2) in feet:", groundCalib?.width_ft || "20");
  if (!wStr) {
    setDrawingMode(null);
    return;
  }
  const lStr = prompt("Real-world length/depth (P1 → P4) in feet:", groundCalib?.length_ft || "10");
  if (!lStr) {
    setDrawingMode(null);
    return;
  }
  const payload = {
    points: draftPoints,
    width_ft: Math.max(1, Number(wStr) || 20),
    length_ft: Math.max(1, Number(lStr) || 10),
    grid_step_ft: 5.0,
    show_grid: true,
  };
  const res = await fetch("api/calibration", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  groundCalib = data.calibration;
  groundToImgMatrix = data.ground_to_img_norm;
  setDrawingMode(null);
  updateCalibrationUI();
}

document.getElementById("btnToggleGrid").addEventListener("click", async () => {
  if (!groundCalib) return;
  const updated = { ...groundCalib, show_grid: !groundCalib.show_grid };
  const res = await fetch("api/calibration", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(updated),
  });
  const data = await res.json();
  groundCalib = data.calibration;
  groundToImgMatrix = data.ground_to_img_norm;
  updateCalibrationUI();
});

const birdseyeModal = document.getElementById("birdseyeModal");
const birdseyeImg = document.getElementById("birdseyeImg");
const birdseyeCanvas = document.getElementById("birdseyeCanvas");
const bevCtx = birdseyeCanvas.getContext("2d");
let bevDrawing = false;
let bevDraftPoints = [];

function renderBevCanvas() {
  const rect = birdseyeImg.getBoundingClientRect();
  if (rect.width > 0 && rect.height > 0) {
    birdseyeCanvas.width = rect.width;
    birdseyeCanvas.height = rect.height;
  }
  const w = birdseyeCanvas.width;
  const h = birdseyeCanvas.height;
  bevCtx.clearRect(0, 0, w, h);
  if (!bevDraftPoints.length) return;

  bevCtx.beginPath();
  bevDraftPoints.forEach(([nx, ny], idx) => {
    if (idx === 0) bevCtx.moveTo(nx * w, ny * h);
    else bevCtx.lineTo(nx * w, ny * h);
  });
  if (bevDraftPoints.length >= 3) {
    bevCtx.closePath();
    bevCtx.fillStyle = "rgba(239, 68, 68, 0.25)";
    bevCtx.fill();
  }
  bevCtx.strokeStyle = "#facc15";
  bevCtx.lineWidth = 2.5;
  bevCtx.stroke();
  for (const [nx, ny] of bevDraftPoints) {
    bevCtx.beginPath();
    bevCtx.arc(nx * w, ny * h, 5, 0, Math.PI * 2);
    bevCtx.fillStyle = "#ef4444";
    bevCtx.fill();
    bevCtx.strokeStyle = "#fff";
    bevCtx.lineWidth = 1.5;
    bevCtx.stroke();
  }
}

function setBevDrawing(active) {
  bevDrawing = active;
  bevDraftPoints = [];
  birdseyeCanvas.classList.toggle("drawing", active);
  birdseyeCanvas.style.pointerEvents = active ? "auto" : "none";
  birdseyeCanvas.style.cursor = active ? "crosshair" : "default";
  document.getElementById("btnBevDrawZone").classList.toggle("hidden", active);
  document.getElementById("btnBevSaveZone").classList.toggle("hidden", !active);
  document.getElementById("btnBevCancelZone").classList.toggle("hidden", !active);
  document.getElementById("bevHint").textContent = active
    ? "Click corners on the top-down map, then click Save Map Zone."
    : "";
  renderBevCanvas();
}

function refreshBirdseyeImage() {
  birdseyeImg.src = `api/calibration/birdseye.jpg?t=${Date.now()}`;
}

birdseyeImg.addEventListener("load", renderBevCanvas);

birdseyeCanvas.addEventListener("click", (e) => {
  if (!bevDrawing) return;
  const rect = birdseyeCanvas.getBoundingClientRect();
  const nx = Math.min(Math.max((e.clientX - rect.left) / rect.width, 0), 1);
  const ny = Math.min(Math.max((e.clientY - rect.top) / rect.height, 0), 1);
  bevDraftPoints.push([Number(nx.toFixed(4)), Number(ny.toFixed(4))]);
  renderBevCanvas();
});

document.getElementById("btnBirdseye").addEventListener("click", () => {
  setBevDrawing(false);
  refreshBirdseyeImage();
  birdseyeModal.classList.remove("hidden");
});
document.getElementById("btnCloseBirdseye").addEventListener("click", () => {
  setBevDrawing(false);
  birdseyeModal.classList.add("hidden");
});
document.getElementById("btnBevRefresh").addEventListener("click", refreshBirdseyeImage);
document.getElementById("btnBevDrawZone").addEventListener("click", () => setBevDrawing(true));
document.getElementById("btnBevCancelZone").addEventListener("click", () => setBevDrawing(false));

document.getElementById("btnBevSaveZone").addEventListener("click", async () => {
  if (bevDraftPoints.length < 3) {
    alert("Click at least 3 corners on the top-down map.");
    return;
  }
  const name = prompt("Name this Red Zone:", `Red Zone ${savedZones.length + 1}`);
  if (!name) return;
  const projRes = await fetch("api/calibration/project", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ points: bevDraftPoints }),
  });
  const proj = await projRes.json();
  const clamped = (proj.points || []).map(([x, y]) => [
    Number(Math.min(Math.max(x, 0), 1).toFixed(4)),
    Number(Math.min(Math.max(y, 0), 1).toFixed(4)),
  ]);
  const updated = [
    ...savedZones,
    {
      id: `zone_${Date.now()}`,
      name: name.trim(),
      points: clamped,
      enabled: true,
      color: "#ef4444",
      zone_type: "red_zone",
    },
  ];
  await saveZones(updated);
  setBevDrawing(false);
  refreshBirdseyeImage();
  syncCanvasSize();
});

document.getElementById("btnBevRecalibrate").addEventListener("click", () => {
  setBevDrawing(false);
  birdseyeModal.classList.add("hidden");
  setDrawingMode("calibrate");
});

document.getElementById("btnBevClear").addEventListener("click", async () => {
  await fetch("api/calibration", { method: "DELETE" });
  groundCalib = null;
  groundToImgMatrix = null;
  setBevDrawing(false);
  birdseyeModal.classList.add("hidden");
  updateCalibrationUI();
});

function escapeHtml(str) {
  return String(str)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

loadZones();
loadLines();
loadCalibration();
refreshStats();
refreshViolations();
pollFrameLoop();
setInterval(refreshStats, 1500);
setInterval(refreshViolations, 5000);
