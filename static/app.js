const videoEl = document.getElementById("videoStream");
const zoneCanvas = document.getElementById("zoneCanvas");
const ctx = zoneCanvas.getContext("2d");
const btnZonesMenu = document.getElementById("btnZonesMenu");
const zonesDropdown = document.getElementById("zonesDropdown");
const drawToolbar = document.getElementById("drawToolbar");
const zonesMenuBadge = document.getElementById("zonesMenuBadge");
const settingsModal = document.getElementById("settingsModal");
const settingsForm = document.getElementById("settingsForm");
const snapshotModal = document.getElementById("snapshotModal");
const birdseyeModal = document.getElementById("birdseyeModal");
const supportsPopover = Object.hasOwn(HTMLElement.prototype, "popover");
const supportsAnchorPositioning =
  typeof CSS !== "undefined" &&
  typeof CSS.supports === "function" &&
  CSS.supports("position-area: block-start");

let savedZones = [];
let savedLines = [];
let lastLinesSignature = "";
let drawingMode = null; // null | "zone" | "exclusion" | "line" | "edit" | "calibrate"
let draftPoints = [];
let editingTarget = null; // null | { kind: "zone" | "line", id: string, name: string }
let dragVertexIndex = -1;
let heatmapEnabled = false;
let groundCalib = null;
let groundToImgMatrix = null;
let latestLiveStats = null;
let latestHardwareDiag = null;

const MODEL_CATALOG = {
  "yolo26s.pt": { name: "YOLO26s", trait: "High Accuracy, NMS-Free", family: "yolo" },
  "yolo26n.pt": { name: "YOLO26n", trait: "Fast, NMS-Free", family: "yolo" },
  "yolo11s.pt": { name: "YOLO11s", trait: "High Accuracy", family: "yolo" },
  "yolo11n.pt": { name: "YOLO11n", trait: "Fast + Balanced", family: "yolo" },
  "yolov8s.pt": { name: "YOLOv8s", trait: "High Accuracy Baseline", family: "yolo" },
  "yolov8n.pt": { name: "YOLOv8n", trait: "Fast Baseline", family: "yolo" },
  "dfine_s.onnx": { name: "D-FINE Small", trait: "Max Accuracy Transformer", family: "dfine" },
  "dfine_n.onnx": { name: "D-FINE Nano", trait: "Fast Transformer", family: "dfine" },
};

function formatModelShortName(rawModel) {
  const key = String(rawModel || "").trim();
  return MODEL_CATALOG[key]?.name || key || "—";
}

function formatRuntimeBackendBadge(rawBackend) {
  const b = String(rawBackend || "").trim().toLowerCase();
  const map = {
    openvino_gpu: "GPU · OpenVINO",
    mps: "GPU · Apple MPS",
    cuda: "GPU · CUDA",
    onnx: "CPU · ONNX",
    openvino: "CPU · OpenVINO",
    pytorch: "CPU · PyTorch",
  };
  return map[b] || (b ? b.toUpperCase() : "—");
}

function resolveModelRuntimeBadge(family, selectedBackend, hwDiag, liveStats) {
  const req = String(selectedBackend || "auto").trim().toLowerCase();
  const hasProbe = Boolean(hwDiag || liveStats);
  const hasOvGpu = hwDiag
    ? Boolean(hwDiag.openvino?.gpu_available)
    : liveStats?.inference_backend === "openvino_gpu";
  const hasMps = hwDiag
    ? Boolean(hwDiag.pytorch?.mps_available)
    : liveStats?.inference_backend === "mps";
  const hasCuda = hwDiag
    ? Boolean(hwDiag.pytorch?.cuda_available)
    : liveStats?.inference_backend === "cuda";

  if (req === "onnx") return "CPU · ONNX Runtime";
  if (req === "openvino") return "CPU · OpenVINO";
  if (req === "openvino_gpu") {
    return !hasProbe || hasOvGpu ? "GPU · OpenVINO FP16" : "CPU · ONNX Fallback";
  }
  if (req === "pytorch") {
    if (family === "dfine") return "CPU · ONNX Runtime";
    if (hasMps) return "GPU · Apple MPS";
    if (hasCuda) return "GPU · CUDA";
    return "CPU · PyTorch";
  }
  if (!hasProbe) {
    return family === "dfine"
      ? "CPU/GPU · OpenVINO, ONNX"
      : "CPU/GPU · OpenVINO, ONNX, PyTorch";
  }
  if (family === "dfine") {
    return hasOvGpu ? "GPU · OpenVINO FP16" : "CPU · ONNX Runtime";
  }
  if (hasMps) return "GPU · Apple MPS";
  if (hasCuda) return "GPU · CUDA";
  if (hasOvGpu) return "GPU · OpenVINO FP16";
  return "CPU · PyTorch";
}

function syncModelOptionLabels() {
  const modelSelect = document.getElementById("cfg-yolo_model");
  if (!modelSelect) return;
  const backendSelect = document.getElementById("cfg-inference_backend");
  const selectedBackend = backendSelect?.value || "auto";
  for (const opt of modelSelect.options) {
    const meta = MODEL_CATALOG[opt.value];
    if (!meta) continue;
    const badge = resolveModelRuntimeBadge(
      meta.family,
      selectedBackend,
      latestHardwareDiag,
      latestLiveStats
    );
    opt.textContent = `${meta.name} — ${meta.trait} [${badge}]`;
  }
}

function isModalOpen(dialog) {
  return Boolean(dialog && (dialog.open || dialog.hasAttribute("open")));
}

function openModalDialog(dialog) {
  if (!dialog) return;
  closeZonesMenu({ restoreFocus: false });
  if (!isModalOpen(dialog)) {
    if (typeof dialog.showModal === "function") {
      dialog.showModal();
    } else {
      dialog.setAttribute("open", "");
    }
  }
}

function closeModalDialog(dialog) {
  if (!dialog) return;
  if (isModalOpen(dialog)) {
    if (typeof dialog.close === "function") {
      dialog.close();
    } else {
      dialog.removeAttribute("open");
    }
  }
}

// Fallback light-dismiss for dialog[closedby="any"] in browsers lacking native closedBy (e.g. Safari)
if (typeof HTMLDialogElement !== "undefined" && !("closedBy" in HTMLDialogElement.prototype)) {
  const outsideDialog = (e, dialog) => {
    if (e.target instanceof Element && e.target.closest(".setting-tooltip")) return false;
    const r = dialog.getBoundingClientRect();
    return e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom;
  };
  for (const dialog of document.querySelectorAll('dialog[closedby="any"]')) {
    let pressedOutside = false;
    dialog.addEventListener("pointerdown", (e) => {
      pressedOutside = isModalOpen(dialog) && outsideDialog(e, dialog);
    });
    dialog.addEventListener("pointerup", (e) => {
      if (pressedOutside && isModalOpen(dialog) && outsideDialog(e, dialog)) {
        closeModalDialog(dialog);
      }
      pressedOutside = false;
    });
    dialog.addEventListener("pointercancel", () => {
      pressedOutside = false;
    });
  }
}

// Accessible setting tooltips / toggletips (popover="hint" + interestfor + cross-browser fallbacks)
const managedTooltipTriggers = new Set();
let activeTipTrigger = null;
let activeOpenTip = null;
let lastTooltipClosedAt = 0;
let suppressDialogCancelThisTick = false;

function isSettingTooltipOpen(tip) {
  if (!tip) return false;
  if (supportsPopover) {
    return tip.matches(":popover-open");
  }
  return tip.classList.contains("is-open");
}

function positionSettingTooltipFallback(trigger, tip) {
  if (!trigger || !tip || supportsAnchorPositioning) return;
  const tRect = trigger.getBoundingClientRect();
  const tipRect = tip.getBoundingClientRect();
  const tipW = tipRect.width || Math.min(310, window.innerWidth - 24);
  const tipH = tipRect.height || 88;
  const margin = 12;
  const left = Math.max(
    margin,
    Math.min(tRect.left + tRect.width / 2 - tipW / 2, window.innerWidth - tipW - margin)
  );
  const fitsAbove = tRect.top - tipH - 8 >= margin;
  const top = fitsAbove
    ? tRect.top - tipH - 8
    : Math.min(window.innerHeight - tipH - margin, tRect.bottom + 8);
  tip.style.left = `${Math.round(left)}px`;
  tip.style.top = `${Math.round(Math.max(margin, top))}px`;
}

function clearActiveTipAnchor() {
  if (activeTipTrigger) {
    activeTipTrigger.style.removeProperty("anchor-name");
  }
  if (activeOpenTip) {
    activeOpenTip.style.removeProperty("position-anchor");
  }
  activeTipTrigger = null;
  activeOpenTip = null;
}

function hideSettingTooltip(tip) {
  if (!tip || !isSettingTooltipOpen(tip)) return false;
  if (supportsPopover) {
    try {
      tip.hidePopover();
    } catch (_) {}
  } else {
    tip.classList.remove("is-open");
    const btn = document.querySelector(`button[interestfor="${CSS.escape(tip.id)}"]`);
    if (btn && managedTooltipTriggers.has(btn)) {
      btn.setAttribute("aria-expanded", "false");
    }
    if (activeOpenTip === tip) {
      clearActiveTipAnchor();
    }
    lastTooltipClosedAt = performance.now();
  }
  return true;
}

function hideAnyOpenSettingTooltip(exceptTip = null) {
  let closed = false;
  for (const tip of document.querySelectorAll(".setting-tooltip")) {
    if (tip !== exceptTip && hideSettingTooltip(tip)) {
      closed = true;
    }
  }
  return closed;
}

function showSettingTooltip(tip, trigger) {
  if (!tip || !trigger) return;
  if (isSettingTooltipOpen(tip)) return;
  hideAnyOpenSettingTooltip(tip);
  if (supportsPopover) {
    try {
      tip.showPopover({ source: trigger });
    } catch (_) {
      try {
        tip.showPopover();
      } catch (_2) {}
    }
  } else {
    clearActiveTipAnchor();
    activeTipTrigger = trigger;
    activeOpenTip = tip;
    tip.classList.add("is-open");
    if (managedTooltipTriggers.has(trigger)) {
      trigger.setAttribute("aria-expanded", "true");
    }
    positionSettingTooltipFallback(trigger, tip);
  }
}

function repositionOpenSettingTooltip() {
  if (!activeOpenTip || !activeTipTrigger || !isSettingTooltipOpen(activeOpenTip)) return;
  const scrollBody = settingsModal?.querySelector(".settings-body");
  if (scrollBody) {
    const bRect = scrollBody.getBoundingClientRect();
    const tRect = activeTipTrigger.getBoundingClientRect();
    if (tRect.bottom < bRect.top || tRect.top > bRect.bottom) {
      hideSettingTooltip(activeOpenTip);
      return;
    }
  }
  positionSettingTooltipFallback(activeTipTrigger, activeOpenTip);
}

function syncTooltipAccessibility(root = document) {
  const getAccessibleName = (el) => {
    const labelledBy = el.getAttribute("aria-labelledby");
    if (labelledBy) {
      const text = labelledBy
        .split(/\s+/)
        .map((id) => document.getElementById(id)?.textContent || "")
        .join(" ")
        .trim();
      if (text) return text;
    }
    const ariaLabel = el.getAttribute("aria-label");
    if (ariaLabel && ariaLabel.trim()) return ariaLabel.trim();
    const textContent = Array.from(el.childNodes)
      .map((node) => (node.nodeType === 1 && node.tagName === "IMG" ? node.getAttribute("alt") || "" : node.textContent || ""))
      .join(" ")
      .trim();
    return textContent || el.getAttribute("title") || "";
  };

  for (const trigger of root.querySelectorAll("[interestfor]")) {
    const targetId = trigger.getAttribute("interestfor");
    const target = targetId ? document.getElementById(targetId) : null;
    if (!target || target.getAttribute("popover") !== "hint") continue;
    const hintText = (target.textContent || "").replace(/\s+/g, " ").trim();
    const triggerName = getAccessibleName(trigger).replace(/\s+/g, " ").trim();
    if (hintText && hintText === triggerName) continue;
    if (!target.hasAttribute("role")) {
      target.setAttribute("role", "tooltip");
    }
    if (!triggerName) {
      if (!trigger.hasAttribute("aria-labelledby")) {
        trigger.setAttribute("aria-labelledby", targetId);
      }
      continue;
    }
    const activationTarget = trigger.getAttribute("popovertarget") || trigger.getAttribute("commandfor");
    if (activationTarget === targetId) {
      if (!trigger.hasAttribute("aria-expanded")) {
        trigger.setAttribute("aria-expanded", String(isSettingTooltipOpen(target)));
        managedTooltipTriggers.add(trigger);
      }
    } else if (!trigger.hasAttribute("aria-describedby")) {
      trigger.setAttribute("aria-describedby", targetId);
    }
  }
}

function initSettingTooltips() {
  syncTooltipAccessibility(document);

  const hintProbe = document.createElement("div");
  if (supportsPopover) {
    hintProbe.popover = "hint";
  }
  const supportsHintPopover = supportsPopover && hintProbe.popover === "hint";

  if (supportsPopover) {
    document.addEventListener(
      "beforetoggle",
      (e) => {
        const target = e.target;
        if (!(target instanceof HTMLElement) || !target.classList.contains("setting-tooltip")) return;
        if (e.newState !== "open") return;
        hideAnyOpenSettingTooltip(target);
        if (!supportsHintPopover) {
          // Switch hint -> auto right before open on Safari/Firefox so Esc & light-dismiss work natively
          target.popover = "auto";
        }
        const trigger = document.querySelector(`button[interestfor="${CSS.escape(target.id)}"]`);
        if (trigger) {
          clearActiveTipAnchor();
          activeTipTrigger = trigger;
          activeOpenTip = target;
          if (supportsAnchorPositioning) {
            trigger.style.setProperty("anchor-name", "--active-setting-tip");
            target.style.setProperty("position-anchor", "--active-setting-tip");
          }
        }
      },
      true
    );

    document.addEventListener(
      "toggle",
      (e) => {
        const target = e.target;
        if (!(target instanceof HTMLElement) || !target.classList.contains("setting-tooltip")) return;
        const isOpen = e.newState === "open";
        for (const btn of document.querySelectorAll(`button[interestfor="${CSS.escape(target.id)}"]`)) {
          if (managedTooltipTriggers.has(btn)) {
            btn.setAttribute("aria-expanded", String(isOpen));
          }
        }
        if (isOpen) {
          positionSettingTooltipFallback(activeTipTrigger, target);
        } else {
          if (activeOpenTip === target) {
            clearActiveTipAnchor();
          }
          lastTooltipClosedAt = performance.now();
          suppressDialogCancelThisTick = true;
          queueMicrotask(() => {
            suppressDialogCancelThisTick = false;
          });
        }
      },
      true
    );
  }

  // Esc inside Settings closes an active tooltip first instead of dismissing the entire modal
  settingsModal?.addEventListener("cancel", (e) => {
    if (
      hideAnyOpenSettingTooltip() ||
      suppressDialogCancelThisTick ||
      performance.now() - lastTooltipClosedAt < 80
    ) {
      e.preventDefault();
    }
  });
  settingsModal?.addEventListener("close", () => {
    hideAnyOpenSettingTooltip();
  });
  settingsModal?.querySelector(".settings-body")?.addEventListener("scroll", repositionOpenSettingTooltip, {
    passive: true,
  });

  // Hover + keyboard focus fallback when native interestfor is unavailable
  if (!("interestForElement" in HTMLButtonElement.prototype)) {
    const fallbackAbort = new AbortController();
    const { signal } = fallbackAbort;
    let hoverOpenTimer = null;
    let hoverCloseTimer = null;
    let pointerInsideTip = false;

    const clearTimers = () => {
      clearTimeout(hoverOpenTimer);
      clearTimeout(hoverCloseTimer);
    };

    const scheduleClose = (tip) => {
      clearTimeout(hoverCloseTimer);
      hoverCloseTimer = setTimeout(() => {
        const trigger = document.querySelector(`button[interestfor="${CSS.escape(tip.id)}"]`);
        if (pointerInsideTip || (trigger && document.activeElement === trigger)) return;
        hideSettingTooltip(tip);
      }, 140);
    };

    for (const trigger of document.querySelectorAll(".help-tip-btn[interestfor]")) {
      const tipId = trigger.getAttribute("interestfor");
      const tip = tipId ? document.getElementById(tipId) : null;
      if (!tip) continue;

      trigger.addEventListener(
        "pointerenter",
        () => {
          clearTimers();
          hoverOpenTimer = setTimeout(() => showSettingTooltip(tip, trigger), 180);
        },
        { signal }
      );
      trigger.addEventListener(
        "pointerleave",
        () => {
          clearTimeout(hoverOpenTimer);
          scheduleClose(tip);
        },
        { signal }
      );
      trigger.addEventListener(
        "focusin",
        () => {
          clearTimers();
          showSettingTooltip(tip, trigger);
        },
        { signal }
      );
      trigger.addEventListener(
        "focusout",
        () => {
          scheduleClose(tip);
        },
        { signal }
      );
      if (!supportsPopover) {
        trigger.addEventListener(
          "click",
          (e) => {
            e.preventDefault();
            showSettingTooltip(tip, trigger);
          },
          { signal }
        );
      }
    }

    for (const tip of document.querySelectorAll(".setting-tooltip")) {
      tip.addEventListener(
        "pointerenter",
        () => {
          pointerInsideTip = true;
          clearTimeout(hoverCloseTimer);
        },
        { signal }
      );
      tip.addEventListener(
        "pointerleave",
        () => {
          pointerInsideTip = false;
          scheduleClose(tip);
        },
        { signal }
      );
    }

    if (!supportsPopover) {
      document.addEventListener(
        "pointerdown",
        (e) => {
          if (
            activeOpenTip &&
            !activeOpenTip.contains(e.target) &&
            !activeTipTrigger?.contains(e.target)
          ) {
            hideAnyOpenSettingTooltip();
          }
        },
        { signal }
      );
    }

    import("https://unpkg.com/interestfor@latest")
      .then(() => {
        fallbackAbort.abort();
      })
      .catch(() => {
        // Offline / Home Assistant air-gapped LAN keeps built-in hover & focus handler active
      });
  }
}

initSettingTooltips();

function updateZonesMenuBadge() {
  if (zonesMenuBadge) {
    zonesMenuBadge.textContent = String(savedZones.length + savedLines.length);
  }
}

function isZonesMenuOpen() {
  if (!zonesDropdown) return false;
  if (supportsPopover) {
    return zonesDropdown.matches(":popover-open");
  }
  return zonesDropdown.classList.contains("is-open");
}

function positionZonesDropdownFallback() {
  if (!zonesDropdown || !btnZonesMenu) return;
  const usesAnchor =
    typeof CSS !== "undefined" &&
    typeof CSS.supports === "function" &&
    CSS.supports("position-area: block-end");
  if ((usesAnchor && window.innerWidth > 640) || window.innerWidth <= 640) {
    zonesDropdown.style.top = "";
    zonesDropdown.style.left = "";
    return;
  }
  const rect = btnZonesMenu.getBoundingClientRect();
  const menuRect = zonesDropdown.getBoundingClientRect();
  const menuW = menuRect.width || Math.min(420, window.innerWidth - 24);
  const menuH = menuRect.height || 360;
  const margin = 12;
  const left = Math.max(margin, Math.min(rect.right - menuW, window.innerWidth - menuW - margin));
  const fitsBelow = rect.bottom + 6 + menuH <= window.innerHeight - margin;
  const top = fitsBelow ? rect.bottom + 6 : Math.max(margin, rect.top - menuH - 6);
  zonesDropdown.style.left = `${Math.round(left)}px`;
  zonesDropdown.style.top = `${Math.round(top)}px`;
}

function closeZonesMenu({ restoreFocus = false } = {}) {
  if (!zonesDropdown) return;
  if (supportsPopover) {
    if (zonesDropdown.matches(":popover-open")) {
      zonesDropdown.hidePopover();
    }
  } else {
    zonesDropdown.classList.remove("is-open");
    btnZonesMenu?.setAttribute("aria-expanded", "false");
  }
  if (restoreFocus && btnZonesMenu && !btnZonesMenu.classList.contains("hidden")) {
    btnZonesMenu.focus();
  }
}

if (zonesDropdown) {
  if (supportsPopover) {
    zonesDropdown.addEventListener("toggle", (e) => {
      const open = e.newState === "open";
      btnZonesMenu?.setAttribute("aria-expanded", String(open));
      if (open) {
        positionZonesDropdownFallback();
      }
    });
  } else {
    btnZonesMenu?.addEventListener("click", (e) => {
      e.preventDefault();
      const willOpen = !zonesDropdown.classList.contains("is-open");
      zonesDropdown.classList.toggle("is-open", willOpen);
      btnZonesMenu.setAttribute("aria-expanded", String(willOpen));
      if (willOpen) {
        positionZonesDropdownFallback();
      }
    });
    document.addEventListener("pointerdown", (e) => {
      if (
        isZonesMenuOpen() &&
        !zonesDropdown.contains(e.target) &&
        !btnZonesMenu?.contains(e.target)
      ) {
        closeZonesMenu({ restoreFocus: false });
      }
    });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && isZonesMenuOpen()) {
        closeZonesMenu({ restoreFocus: true });
      }
    });
  }
}

function syncCanvasSize() {
  const rect = videoEl.getBoundingClientRect();
  if (rect.width > 0 && rect.height > 0) {
    zoneCanvas.width = rect.width;
    zoneCanvas.height = rect.height;
    renderZoneCanvas();
  }
}

window.addEventListener("resize", () => {
  syncCanvasSize();
  if (isZonesMenuOpen()) {
    positionZonesDropdownFallback();
  }
  repositionOpenSettingTooltip();
});
window.addEventListener(
  "scroll",
  () => {
    if (isZonesMenuOpen()) {
      positionZonesDropdownFallback();
    }
  },
  { passive: true }
);
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
    if (!z.enabled) {
      ctx.strokeStyle = "rgba(148, 163, 184, 0.45)";
    } else if (z.zone_type === "exclusion") {
      ctx.strokeStyle = "rgba(148, 163, 184, 0.78)";
    } else if (z.zone_type === "person_zone") {
      ctx.strokeStyle = "rgba(168, 85, 247, 0.82)";
    } else {
      ctx.strokeStyle = "rgba(239, 68, 68, 0.72)";
    }
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  // Draw existing lines lightly while editing
  for (const ln of savedLines) {
    if (editingTarget && editingTarget.kind === "line" && editingTarget.id === ln.id) continue;
    const isSpeedOnly = (ln.line_role || "both") === "speed";
    ctx.beginPath();
    ctx.setLineDash(isSpeedOnly ? [8, 6] : []);
    ctx.moveTo(ln.p1[0] * w, ln.p1[1] * h);
    ctx.lineTo(ln.p2[0] * w, ln.p2[1] * h);
    ctx.strokeStyle = !ln.enabled
      ? "rgba(148, 163, 184, 0.5)"
      : isSpeedOnly
        ? "rgba(245, 158, 11, 0.85)"
        : "rgba(56, 189, 248, 0.75)";
    ctx.lineWidth = 3;
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // Draw active draft polygon or line
  if (draftPoints.length > 0) {
    const isLine =
      drawingMode === "line" ||
      drawingMode === "speed_line" ||
      (drawingMode === "edit" && editingTarget?.kind === "line");
    const isSpeedDraft =
      drawingMode === "speed_line" ||
      (drawingMode === "edit" && editingTarget?.kind === "line" && editingTarget?.line_role === "speed");
    ctx.beginPath();
    ctx.setLineDash(isSpeedDraft ? [8, 6] : []);
    draftPoints.forEach(([nx, ny], idx) => {
      if (idx === 0) ctx.moveTo(nx * w, ny * h);
      else ctx.lineTo(nx * w, ny * h);
    });
    if (drawingMode === "edit" && !isLine && draftPoints.length >= 3) {
      ctx.closePath();
      ctx.fillStyle = "rgba(250, 204, 21, 0.12)";
      ctx.fill();
    }
    ctx.strokeStyle = isSpeedDraft ? "#f59e0b" : isLine ? "#38bdf8" : "#facc15";
    ctx.lineWidth = 2.5;
    ctx.stroke();
    ctx.setLineDash([]);

    const handleRadius =
      drawingMode === "edit"
        ? (window.matchMedia("(pointer: coarse)").matches ? 9 : 7)
        : 5;
    draftPoints.forEach(([nx, ny], idx) => {
      const px = nx * w;
      const py = ny * h;
      ctx.beginPath();
      ctx.arc(px, py, handleRadius, 0, Math.PI * 2);
      let handleColor = "#ef4444";
      if (isSpeedDraft) handleColor = "#f59e0b";
      else if (isLine) handleColor = "#38bdf8";
      else if (drawingMode === "exclusion" || (drawingMode === "edit" && editingTarget?.zone_type === "exclusion")) {
        handleColor = "#94a3b8";
      } else if (drawingMode === "person_zone" || (drawingMode === "edit" && editingTarget?.zone_type === "person_zone")) {
        handleColor = "#a855f7";
      }
      ctx.fillStyle = handleColor;
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
  const focusWasInDropdown = Boolean(zonesDropdown?.contains(document.activeElement));
  const focusWasInDrawToolbar = Boolean(drawToolbar?.contains(document.activeElement));
  drawingMode = mode;
  editingTarget = target;
  dragVertexIndex = -1;
  if (!target) {
    draftPoints = [];
  }
  const active = Boolean(mode);
  const isEdit = mode === "edit";
  if (active) {
    closeZonesMenu({ restoreFocus: false });
  }
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
  } else if (mode === "speed_line") {
    hint.textContent = "Click 2 points across the lane to place a speed-measurement line (requires 2 speed lines + 3D Ground Grid).";
  } else if (mode === "line") {
    hint.textContent = "Click 2 points on the video to place a directional traffic counting line (A → B / B → A).";
  } else if (mode === "exclusion") {
    hint.textContent = 'Click on the video to outline an exclusion mask (e.g. balcony/plants to ignore). Double-click or press "Finish & Save".';
  } else if (mode === "person_zone") {
    hint.textContent = 'Click on the video to outline an after-hours Person Zone. Double-click or press "Finish & Save" when done.';
  } else {
    hint.textContent = 'Click on the video to place polygon corners around the red zone. Double-click or press "Finish & Save" when done.';
  }
  btnZonesMenu?.classList.toggle("hidden", active);
  drawToolbar?.classList.toggle("hidden", !active);
  document.getElementById("btnUndoPoint").classList.toggle("hidden", !active || isEdit);
  document.getElementById("btnRenameEdit").classList.toggle("hidden", !isEdit);
  const finishBtn = document.getElementById("btnFinishZone");
  const showFinish = mode === "zone" || mode === "exclusion" || mode === "person_zone" || isEdit;
  finishBtn.classList.toggle("hidden", !showFinish);
  const cancelBtn = document.getElementById("btnCancelDraw");
  cancelBtn.classList.toggle("hidden", !active);
  if (active && focusWasInDropdown) {
    (showFinish ? finishBtn : cancelBtn)?.focus();
  } else if (!active && focusWasInDrawToolbar) {
    btnZonesMenu?.focus();
  }
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
  if ((drawingMode === "line" || drawingMode === "speed_line") && draftPoints.length === 2) {
    finishLine();
  } else if (drawingMode === "calibrate" && draftPoints.length === 4) {
    finishCalibration();
  }
});

zoneCanvas.addEventListener("dblclick", () => {
  if ((drawingMode === "zone" || drawingMode === "exclusion" || drawingMode === "person_zone") && draftPoints.length >= 3) {
    finishZone();
  }
});

document.getElementById("btnDrawZone").addEventListener("click", () => setDrawingMode("zone"));
document.getElementById("btnAutoRedZone").addEventListener("click", autoDetectRedCurb);
document.getElementById("btnDrawPersonZone")?.addEventListener("click", () => setDrawingMode("person_zone"));
document.getElementById("btnDrawExclusion").addEventListener("click", () => setDrawingMode("exclusion"));
document.getElementById("btnDrawLine").addEventListener("click", () => setDrawingMode("line"));
document.getElementById("btnDrawSpeedLine")?.addEventListener("click", () => setDrawingMode("speed_line"));
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
  const isSpeedOnly = drawingMode === "speed_line";
  const defaultName = isSpeedOnly
    ? `Speed Line ${savedLines.length + 1}`
    : `Traffic Line ${savedLines.length + 1}`;
  const promptTitle = isSpeedOnly
    ? "Name this speed measurement line:"
    : "Name this traffic counting line:";
  const name = prompt(promptTitle, defaultName);
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
      color: isSpeedOnly ? "#f59e0b" : "#38bdf8",
      count_a_to_b: 0,
      count_b_to_a: 0,
      line_role: isSpeedOnly ? "speed" : "count",
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
  setDrawingMode("edit", {
    kind: "line",
    id: ln.id,
    name: ln.name,
    line_role: ln.line_role || "both",
  });
}

function startEditZone(z) {
  draftPoints = z.points.map((pt) => [...pt]);
  setDrawingMode("edit", { kind: "zone", id: z.id, name: z.name, zone_type: z.zone_type || "red_zone" });
}

function getFocusedRowMeta(containerEl) {
  const activeEl = document.activeElement;
  if (!containerEl?.contains(activeEl)) return null;
  const id = activeEl.dataset?.id;
  const action = activeEl.dataset?.action;
  return id && action ? { id, action } : null;
}

function restoreRowFocus(containerEl, focusMeta, fallbackEl) {
  if (!focusMeta) return;
  const target = containerEl.querySelector(
    `button[data-id="${CSS.escape(focusMeta.id)}"][data-action="${CSS.escape(focusMeta.action)}"]`
  );
  if (target) {
    target.focus();
  } else {
    (containerEl.querySelector("button") || fallbackEl || btnZonesMenu)?.focus();
  }
}

function renderLinesList() {
  const el = document.getElementById("linesList");
  const focusMeta = getFocusedRowMeta(el);
  lastLinesSignature = JSON.stringify(savedLines);
  updateZonesMenuBadge();
  if (!savedLines.length) {
    el.innerHTML = `<span class="muted small">No traffic lines defined yet.</span>`;
    if (focusMeta) {
      document.getElementById("btnDrawLine")?.focus();
    }
    return;
  }
  el.innerHTML = "";
  const nextRoleMap = { both: "count", count: "speed", speed: "both" };
  for (const ln of savedLines) {
    const role = ln.line_role || "both";
    const roleLabel =
      role === "count" ? "Count Only" : role === "speed" ? "Speed Only" : "Count + Speed";
    const total = (ln.count_a_to_b || 0) + (ln.count_b_to_a || 0);
    const labelHtml =
      role === "speed"
        ? `${ln.enabled ? "⚡" : "⚪"} ${escapeHtml(ln.name)} <span class="muted small">· Speed trap</span>`
        : `${ln.enabled ? (role === "both" ? "🔵⚡" : "🔵") : "⚪"} ${escapeHtml(ln.name)}: <strong>${total}</strong> (${ln.count_a_to_b || 0} / ${ln.count_b_to_a || 0})`;
    const row = document.createElement("div");
    row.className = "zone-menu-row";
    row.innerHTML = `
      <span class="zone-menu-label">${labelHtml}</span>
      <div class="zone-menu-actions">
        <button type="button" class="line-role-btn role-${escapeHtml(role)}" data-id="${escapeHtml(ln.id)}" data-action="role" aria-label="${roleLabel} — ${escapeHtml(ln.name)}" title="Click to switch mode: Count + Speed → Count Only → Speed Only">${roleLabel}</button>
        <button type="button" data-id="${escapeHtml(ln.id)}" data-action="edit" aria-label="Edit ${escapeHtml(ln.name)}" title="Edit">Edit</button>
        <button type="button" data-id="${escapeHtml(ln.id)}" data-action="toggle" aria-label="${ln.enabled ? "Disable" : "Enable"} ${escapeHtml(ln.name)}" title="Toggle">${ln.enabled ? "Disable" : "Enable"}</button>
        <button type="button" data-id="${escapeHtml(ln.id)}" data-action="delete" aria-label="Delete ${escapeHtml(ln.name)}" title="Delete">✕</button>
      </div>
    `;
    const [roleBtn, editBtn, toggleBtn, delBtn] = row.querySelectorAll("button");
    roleBtn.addEventListener("click", () => {
      const nextRole = nextRoleMap[role] || "count";
      const next = savedLines.map((item) => {
        if (item.id !== ln.id) return item;
        const curColor = String(item.color || "#38bdf8").toLowerCase();
        let nextColor = item.color || "#38bdf8";
        if (nextRole === "speed" && curColor === "#38bdf8") {
          nextColor = "#f59e0b";
        } else if (nextRole !== "speed" && curColor === "#f59e0b") {
          nextColor = "#38bdf8";
        }
        return { ...item, line_role: nextRole, color: nextColor };
      });
      saveLines(next);
    });
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
    el.appendChild(row);
  }
  restoreRowFocus(el, focusMeta, document.getElementById("btnDrawLine"));
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
  const isPersonZone = drawingMode === "person_zone";
  let defaultName = `Red Zone ${savedZones.length + 1}`;
  let promptLabel = "Name this red zone:";
  let color = "#ef4444";
  let zoneType = "red_zone";
  if (isExclusion) {
    defaultName = `Exclusion Mask ${savedZones.length + 1}`;
    promptLabel = "Name this exclusion mask:";
    color = "#94a3b8";
    zoneType = "exclusion";
  } else if (isPersonZone) {
    defaultName = `Person Zone ${savedZones.length + 1}`;
    promptLabel = "Name this person watch zone:";
    color = "#a855f7";
    zoneType = "person_zone";
  }
  const name = prompt(promptLabel, defaultName);
  if (!name) return;

  const updated = [
    ...savedZones,
    {
      id: `zone_${Date.now()}`,
      name: name.trim(),
      points: draftPoints,
      enabled: true,
      color,
      zone_type: zoneType,
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
  const focusMeta = getFocusedRowMeta(el);
  updateZonesMenuBadge();
  if (!savedZones.length) {
    el.innerHTML = `<span class="muted small">No zones defined yet.</span>`;
    if (focusMeta) {
      document.getElementById("btnDrawZone")?.focus();
    }
    return;
  }
  el.innerHTML = "";
  for (const z of savedZones) {
    const row = document.createElement("div");
    row.className = "zone-menu-row";
    let icon = "🔴";
    if (z.zone_type === "exclusion") icon = "🩶";
    else if (z.zone_type === "person_zone") icon = "🟣";
    row.innerHTML = `
      <span class="zone-menu-label">${z.enabled ? icon : "⚪"} ${escapeHtml(z.name)}</span>
      <div class="zone-menu-actions">
        <button type="button" data-id="${escapeHtml(z.id)}" data-action="edit" title="Edit">Edit</button>
        <button type="button" data-id="${escapeHtml(z.id)}" data-action="toggle" title="Toggle">${z.enabled ? "Disable" : "Enable"}</button>
        <button type="button" data-id="${escapeHtml(z.id)}" data-action="delete" aria-label="Delete ${escapeHtml(z.name)}" title="Delete">✕</button>
      </div>
    `;
    const [editBtn, toggleBtn, delBtn] = row.querySelectorAll("button");
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
    el.appendChild(row);
  }
  restoreRowFocus(el, focusMeta, document.getElementById("btnDrawZone"));
}

function formatSourceSummary(data) {
  if (data.source_label) return String(data.source_label);
  const raw = String(data.source || "").trim();
  if (!raw) return "";
  const lower = raw.toLowerCase();
  if (lower.startsWith("rtmp://") || lower.startsWith("rtsp://")) {
    const proto = lower.startsWith("rtmp://") ? "RTMP" : "RTSP";
    if (/([_=/])sub\b|stream=1\b/i.test(raw)) return `${proto} • Sub Stream`;
    if (/([_=/])main\b|stream=0\b/i.test(raw)) return `${proto} • Main Stream`;
    return proto;
  }
  if (lower.startsWith("http://") || lower.startsWith("https://")) {
    return "HTTP Snap";
  }
  const fname = raw.split(/[\\/]/).pop() || "video";
  return `File (${fname})`;
}

function formatStreamStatus(rawStatus) {
  const s = String(rawStatus || "").trim().toLowerCase();
  const map = {
    connected: "CONNECTED",
    connected_http: "HTTP SNAP",
    connecting: "CONNECTING",
    reconnecting: "RECONNECTING",
    initializing: "STARTING",
    loading_model: "LOAD MODEL",
    waiting_for_credentials: "NEEDS LOGIN",
    error: "ERROR",
  };
  return map[s] || String(rawStatus || "UNKNOWN").toUpperCase();
}

function renderStreamTelemetryHtml(data) {
  const statusLabel = escapeHtml(formatStreamStatus(data.stream_status));
  const resStr = data.resolution?.width
    ? `${Number(data.resolution.width)}x${Number(data.resolution.height)}`
    : "—";
  const fpsVal = Number(data.fps ?? 0).toFixed(2);
  const modelLabel = escapeHtml(formatModelShortName(data.yolo_model));
  const backendLabel = formatRuntimeBackendBadge(data.inference_backend);
  const sizeLabel = data.input_resolution
    ? `@${data.input_resolution}`
    : (data.yolo_imgsz ? `@${data.yolo_imgsz}` : "");
  const backendFull = escapeHtml(`${backendLabel}${sizeLabel}`);

  let motionLabel = "Gate: Off";
  let motionActive = false;
  if (data.motion_gate_enabled && data.motion_state !== "off") {
    if (data.motion_state === "idle") {
      motionLabel = "Gate: Idle";
    } else {
      motionLabel = "Gate: Active";
      motionActive = true;
    }
  }

  const sys = data.system || {};
  const cpuHtml = sys.cpu_percent != null
    ? `CPU<span class="t-num t-num-pct">${Number(sys.cpu_percent).toFixed(1)}</span>%`
    : "CPU —";
  const ramHtml = sys.memory_percent != null
    ? `RAM<span class="t-num t-num-pct">${Number(sys.memory_percent).toFixed(1)}</span>%`
    : "RAM —";

  const slots = [
    `<span class="t-slot t-status">${statusLabel}</span>`,
    `<span class="t-slot t-res">${escapeHtml(resStr)}</span>`,
    `<span class="t-slot t-fps"><span class="t-num t-num-fps">${fpsVal}</span>FPS</span>`,
    `<span class="t-slot t-model">${modelLabel}</span>`,
    `<span class="t-slot t-backend">${backendFull}</span>`,
    `<span class="t-slot t-motion${motionActive ? " is-active" : ""}">${motionLabel}</span>`,
    `<span class="t-slot t-cpu">${cpuHtml}</span>`,
    `<span class="t-slot t-ram">${ramHtml}</span>`,
  ];

  if (sys.cpu_temp_c != null) {
    slots.push(
      `<span class="t-slot t-temp"><span class="t-num t-num-temp">${Number(sys.cpu_temp_c).toFixed(1)}</span>°C</span>`
    );
  }

  const sourceLabel = formatSourceSummary(data);
  if (sourceLabel) {
    slots.push(`<span class="t-slot t-source">${escapeHtml(sourceLabel)}</span>`);
  }

  let noteText = "";
  if (data.model_loading && data.model_loading_target) {
    noteText = `LOADING ${data.model_loading_target}`;
  } else if (data.model_error) {
    noteText = `MODEL ERR: ${data.model_error}`;
  } else if (data.backend_fallback) {
    noteText = String(data.backend_fallback);
  }
  if (noteText) {
    slots.push(`<span class="t-slot t-note">${escapeHtml(noteText)}</span>`);
  }

  return slots.join("");
}

function updateCvRuntimeHint() {
  const hintEl = document.getElementById("cvRuntimeHint");
  if (!hintEl) return;
  if (!latestLiveStats) {
    hintEl.textContent = "Runtime status will appear once connected.";
    return;
  }
  const activeModel = formatModelShortName(latestLiveStats.yolo_model);
  const activeBackend = formatRuntimeBackendBadge(latestLiveStats.inference_backend);
  const activeSize = latestLiveStats.input_resolution
    ? ` @ ${latestLiveStats.input_resolution}`
    : (latestLiveStats.yolo_imgsz ? ` @ ${latestLiveStats.yolo_imgsz}px` : "");
  const parts = [`Active: ${activeModel} [${activeBackend}${activeSize}]`];
  if (latestLiveStats.video_decode === "vaapi") {
    parts.push("Decode: VA-API");
  }
  if (latestLiveStats.motion_gate_enabled) {
    parts.push(
      latestLiveStats.motion_state === "idle"
        ? "Motion Gate: Idle (0.5 FPS)"
        : "Motion Gate: Active (Full)"
    );
  }
  if (latestLiveStats.backend_fallback) {
    parts.push(`⚠️ ${latestLiveStats.backend_fallback}`);
  }
  if (latestLiveStats.model_loading && latestLiveStats.model_loading_target) {
    parts.push(`⏳ Loading ${latestLiveStats.model_loading_target}…`);
  }
  if (latestLiveStats.model_error) {
    parts.push(`❌ ${latestLiveStats.model_error}`);
  }
  hintEl.textContent = parts.join(" • ");
}

async function refreshStats() {
  try {
    const res = await fetch("api/stats");
    const data = await res.json();
    latestLiveStats = data;

    document.getElementById("kpiActiveCars").textContent = data.active_cars;
    document.getElementById("kpiUniqueCars").textContent = data.unique_cars;
    document.getElementById("kpiTotalCars").textContent = data.total_cars;
    document.getElementById("kpiPeople").textContent = `${data.active_people} / ${data.unique_people}`;
    document.getElementById("kpiViolations").textContent = data.violations_total;
    const personAlertsEl = document.getElementById("kpiPersonAlerts");
    if (personAlertsEl) {
      personAlertsEl.textContent = data.person_alerts ?? 0;
    }
    const personSchedEl = document.getElementById("kpiPersonSchedule");
    if (personSchedEl) {
      if (!data.person_zone_enabled) {
        personSchedEl.textContent = "Disabled";
        personSchedEl.style.color = "";
      } else if (data.person_zone_active) {
        personSchedEl.textContent = `ACTIVE (${data.person_zone_window || "00:00–05:00"})`;
        personSchedEl.style.color = "#c084fc";
      } else {
        personSchedEl.textContent = `Scheduled ${data.person_zone_window || "00:00–05:00"}`;
        personSchedEl.style.color = "";
      }
    }
    if (Array.isArray(data.counting_lines)) {
      savedLines = data.counting_lines;
      const sig = JSON.stringify(savedLines);
      if (sig !== lastLinesSignature) {
        renderLinesList();
      }
      const totalCrossings = savedLines
        .filter((ln) => (ln.line_role || "both") !== "speed")
        .reduce((acc, ln) => acc + (ln.count_a_to_b || 0) + (ln.count_b_to_a || 0), 0);
      document.getElementById("kpiLineCrossings").textContent = totalCrossings;
    }

    const dot = document.getElementById("statusDot");
    dot.className = "dot";
    if (data.stream_status.startsWith("connected")) dot.classList.add("connected");
    else if (data.stream_status === "error") dot.classList.add("error");

    const verBadge = document.getElementById("appVersionBadge");
    if (verBadge && data.version) {
      const verLabel = `v${String(data.version).replace(/^v/i, "")}`;
      if (verBadge.textContent !== verLabel) {
        verBadge.textContent = verLabel;
      }
      verBadge.classList.remove("hidden");
    }

    const streamInfoEl = document.getElementById("streamInfo");
    if (streamInfoEl) {
      const nextTelemetryHtml = renderStreamTelemetryHtml(data);
      if (streamInfoEl.innerHTML !== nextTelemetryHtml) {
        streamInfoEl.innerHTML = nextTelemetryHtml;
      }
    }
    if (isModalOpen(settingsModal)) {
      updateCvRuntimeHint();
    }

    heatmapEnabled = Boolean(data.heatmap_enabled);
    document.getElementById("btnHeatmapToggle").textContent = `Heatmap: ${heatmapEnabled ? "ON" : "OFF"}`;

    const speedEl = document.getElementById("speedSummary");
    if (speedEl) {
      const sp = data.speed || {};
      const speedLineCount = savedLines.filter(
        (ln) => ln.enabled && (ln.line_role || "both") !== "count"
      ).length;
      if (!sp.calibrated) {
        speedEl.textContent = "Speed (2-line): Calibrate 3D Ground Grid and enable 2 Speed or Count+Speed lines to measure mph.";
      } else if (sp.samples_count > 0) {
        const recentText = (sp.recent || [])
          .slice(0, 4)
          .map((r) => `#${r.track_id} ${r.vehicle_class} ${Number(r.speed_mph).toFixed(1)} mph`)
          .join(" • ");
        speedEl.textContent = `Speed (2-line): Last ${Number(sp.last_speed_mph).toFixed(1)} mph • Avg ${Number(sp.avg_speed_mph).toFixed(1)} mph • Max ${Number(sp.max_speed_mph).toFixed(1)} mph` + (recentText ? ` — ${recentText}` : "");
      } else if (speedLineCount < 2) {
        speedEl.textContent = `Speed (2-line): Enable at least 2 Speed or Count+Speed lines (${speedLineCount} active) to measure mph.`;
      } else {
        speedEl.textContent = "Speed (2-line): Waiting for a vehicle to cross 2 Speed / Count+Speed lines.";
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

let lastViolationItems = [];
let lastViolationsSignature = "";
let openSnapshotViolationId = null;

const durationFmt =
  typeof Intl !== "undefined" && typeof Intl.DurationFormat === "function"
    ? new Intl.DurationFormat(undefined, { style: "narrow" })
    : null;

function formatDurationSeconds(seconds) {
  const total = Math.max(0, Math.round(Number(seconds) || 0));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  if (durationFmt) {
    const dur = {};
    if (hours > 0) dur.hours = hours;
    if (minutes > 0 || hours > 0) dur.minutes = minutes;
    dur.seconds = secs;
    return durationFmt.format(dur);
  }
  if (hours > 0) return `${hours}h ${minutes}m ${secs}s`;
  if (minutes > 0) return `${minutes}m ${secs}s`;
  return `${secs}s`;
}

function violationImageUrl(v) {
  const rev = v.zone_id === "speeding"
    ? Math.round((Number(v.dwell_seconds) || 0) * 10)
    : Math.round(Number(v.dwell_seconds) || 0);
  return `snapshots/${encodeURIComponent(v.snapshot_filename)}?t=${rev}`;
}

function formatSnapshotTitle(v) {
  const plateSuffix = v.license_plate ? ` [${v.license_plate}]` : "";
  const base = `${v.vehicle_class.toUpperCase()} #${v.track_id}${plateSuffix} — ${v.zone_name}`;
  return v.zone_id === "speeding" ? base : `${base} (${formatDurationSeconds(v.dwell_seconds)})`;
}

function renderViolationsList() {
  const list = document.getElementById("violationsList");
  const filterInput = document.getElementById("violationFilter");
  const query = (filterInput?.value || "").trim().toLowerCase();

  const filtered = query
    ? lastViolationItems.filter((v) => {
        const hay = `${v.license_plate || ""} ${v.zone_name || ""} ${v.vehicle_class || ""} #${v.track_id}`.toLowerCase();
        return hay.includes(query);
      })
    : lastViolationItems;

  const labelEl = document.getElementById("violationCountLabel");
  labelEl.textContent = query
    ? `${filtered.length} / ${lastViolationItems.length} shown`
    : `${lastViolationItems.length} shown`;

  if (!filtered.length) {
    list.innerHTML = query
      ? `<p class="muted">No violations or alerts match "${escapeHtml(query)}".</p>`
      : `<p class="muted">No violations or alerts recorded yet.</p>`;
    return;
  }

  list.innerHTML = "";
  for (const v of filtered) {
    const card = document.createElement("div");
    card.className = "violation-card";
    const timeStr = new Date(v.flagged_at).toLocaleString();
    const isPersonAlert = v.vehicle_class === "person" || String(v.zone_id || "").startsWith("person_zone");
    let detailLine;
    if (v.zone_id === "speeding") {
      detailLine = `Speed: ${v.dwell_seconds.toFixed(1)} mph`;
    } else if (isPersonAlert) {
      detailLine = `Detected dwell: ${escapeHtml(formatDurationSeconds(v.dwell_seconds))}`;
    } else {
      detailLine = `Parked: ${escapeHtml(formatDurationSeconds(v.dwell_seconds))}`;
    }
    const plateBadge = v.license_plate
      ? `<div class="plate-badge">🪪 ${escapeHtml(v.license_plate)}</div>`
      : "";
    const imgUrl = violationImageUrl(v);
    card.innerHTML = `
      <img src="${imgUrl}" alt="Snapshot" width="96" height="60" loading="lazy" />
      <div class="violation-meta">
        <div class="violation-title${isPersonAlert ? " is-person" : ""}">${escapeHtml(v.vehicle_class.toUpperCase())} #${v.track_id} • ${escapeHtml(v.zone_name)}</div>
        ${plateBadge}
        <div class="muted">${detailLine}</div>
        <div class="muted">${escapeHtml(timeStr)}</div>
      </div>
      <button class="btn btn-ghost" type="button" title="Delete" aria-label="Delete violation #${v.id}">✕</button>
    `;
    card.querySelector("img").addEventListener("click", () => {
      openSnapshotViolationId = v.id;
      document.getElementById("snapshotTitle").textContent = formatSnapshotTitle(v);
      document.getElementById("snapshotFullImg").src = violationImageUrl(v);
      openModalDialog(snapshotModal);
    });
    card.querySelector("button").addEventListener("click", async () => {
      if (openSnapshotViolationId === v.id) {
        openSnapshotViolationId = null;
        closeModalDialog(snapshotModal);
      }
      await fetch(`api/violations/${v.id}`, { method: "DELETE" });
      refreshViolations();
      refreshStats();
    });
    list.appendChild(card);
  }
}

document.getElementById("violationFilter")?.addEventListener("input", renderViolationsList);

async function refreshViolations() {
  try {
    const res = await fetch("api/violations?limit=30");
    const items = await res.json();
    const sig = JSON.stringify(
      items.map((v) => [v.id, v.dwell_seconds, v.license_plate, v.snapshot_filename])
    );
    if (sig === lastViolationsSignature) return;
    lastViolationsSignature = sig;
    lastViolationItems = items;
    renderViolationsList();

    if (openSnapshotViolationId !== null && isModalOpen(snapshotModal)) {
      const openItem = items.find((v) => v.id === openSnapshotViolationId);
      if (openItem) {
        document.getElementById("snapshotTitle").textContent = formatSnapshotTitle(openItem);
        const fullImg = document.getElementById("snapshotFullImg");
        const nextUrl = violationImageUrl(openItem);
        if (fullImg.getAttribute("src") !== nextUrl) {
          fullImg.src = nextUrl;
        }
      }
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

async function refreshHardwareDiagnostics(force = false) {
  const hwEl = document.getElementById("hwDiagnosticHint");
  const btn = document.getElementById("btnRefreshHardware");
  if (!hwEl || !isModalOpen(settingsModal)) return;
  if (btn) btn.disabled = true;
  try {
    const url = force ? "api/system/hardware?refresh=true" : "api/system/hardware";
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    latestHardwareDiag = data;
    syncModelOptionLabels();
    if (!isModalOpen(settingsModal)) return;
    const cpuLabel = data.cpu_model
      ? `${data.cpu_model} (${data.cpu_cores || "?"}c)`
      : "Unknown CPU";
    const renderNodes = (data.dri_devices || [])
      .filter((d) => String(d.name || "").startsWith("renderD"))
      .map((d) => d.name);
    const driLabel = renderNodes.length ? `DRI: ${renderNodes.join(",")}` : "DRI: none";
    const pciLabel = data.pci_gpus?.length
      ? `GPU: ${data.pci_gpus.map((g) => g.description || g.vendor).join(", ")}`
      : "";
    const summaryParts = [cpuLabel, driLabel];
    if (pciLabel) summaryParts.push(pciLabel);
    if (data.detector_runtime?.video_decode === "vaapi") summaryParts.push("Decode: VA-API");
    if (data.diagnosis) summaryParts.push(data.diagnosis);
    const summaryText = summaryParts.join(" • ");
    if (data.gpu_available) {
      hwEl.textContent = `✓ ${summaryText}`;
      hwEl.style.color = "#22c55e";
    } else if (settingsForm.elements["inference_backend"]?.value === "openvino_gpu") {
      hwEl.textContent = `! ${summaryText}`;
      hwEl.style.color = "#f59e0b";
    } else {
      hwEl.textContent = summaryText;
      hwEl.style.color = "";
    }
  } catch (err) {
    if (isModalOpen(settingsModal)) {
      hwEl.textContent = `! Hardware check failed: ${err.message || err}`;
      hwEl.style.color = "#ef4444";
    }
  } finally {
    if (btn) btn.disabled = false;
  }
}

document.getElementById("btnRefreshHardware")?.addEventListener("click", () => {
  refreshHardwareDiagnostics(true);
});

document.getElementById("cfg-inference_backend")?.addEventListener("change", () => {
  syncModelOptionLabels();
  if (latestHardwareDiag) {
    refreshHardwareDiagnostics(false);
  }
});

// Settings & Snapshot dialogs
document.getElementById("btnSettings").addEventListener("click", async () => {
  const res = await fetch("api/config");
  const cfg = await res.json();
  for (const [key, val] of Object.entries(cfg)) {
    const input = settingsForm.elements[key];
    if (!input) continue;
    if (input.type === "checkbox") input.checked = Boolean(val);
    else input.value = val ?? "";
  }
  const haHint = document.getElementById("haSupervisorHint");
  if (haHint) {
    haHint.textContent = cfg.ha_supervisor_detected
      ? "✅ Home Assistant Supervisor API detected automatically — leave URL & Token blank."
      : "Optional: Enter Home Assistant URL & Long-Lived Access Token if running outside HAOS.";
  }
  document.getElementById("testStreamResult").textContent = "";
  document.getElementById("testWebhookResult").textContent = "";
  const haOut = document.getElementById("testHaResult");
  if (haOut) haOut.textContent = "";
  syncModelOptionLabels();
  updateCvRuntimeHint();
  openModalDialog(settingsModal);
  refreshHardwareDiagnostics(false);
});

document.getElementById("btnCloseSettings").addEventListener("click", () => closeModalDialog(settingsModal));
document.getElementById("btnCancelSettings").addEventListener("click", () => closeModalDialog(settingsModal));
document.getElementById("btnCloseSnapshot").addEventListener("click", () => closeModalDialog(snapshotModal));
snapshotModal?.addEventListener("close", () => {
  openSnapshotViolationId = null;
});

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

document.getElementById("btnTestHa")?.addEventListener("click", async () => {
  const out = document.getElementById("testHaResult");
  const ha_url = settingsForm.elements["ha_url"]?.value?.trim() || "";
  const ha_token = settingsForm.elements["ha_token"]?.value?.trim() || "";
  out.textContent = "Syncing Home Assistant entities...";
  out.style.color = "#8b9bb4";
  try {
    const res = await fetch("api/ha/test", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ha_url, ha_token }),
    });
    const data = await res.json();
    if (data.ok) {
      out.textContent = `Synced ${data.entities} entities (${data.mode})`;
      out.style.color = "#22c55e";
    } else {
      out.textContent = data.error || "HA sync failed";
      out.style.color = "#ef4444";
    }
  } catch (err) {
    out.textContent = err.message || "HA sync failed";
    out.style.color = "#ef4444";
  }
});

settingsForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const submitBtn = settingsForm.querySelector('button[type="submit"]');
  const prevLabel = submitBtn ? submitBtn.textContent : "";
  if (submitBtn) {
    submitBtn.disabled = true;
    submitBtn.textContent = "Applying...";
  }
  try {
    const payload = formToPayload();
    payload.heatmap_enabled = heatmapEnabled;
    const res = await fetch("api/config", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      let detail = `HTTP ${res.status}`;
      try {
        const body = await res.json();
        if (body && body.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
      } catch (_) {}
      throw new Error(detail);
    }
    closeModalDialog(settingsModal);
    refreshStats();
  } catch (err) {
    alert(`Failed to save settings: ${err.message || err}`);
  } finally {
    if (submitBtn) {
      submitBtn.disabled = false;
      submitBtn.textContent = prevLabel || "Save & Apply";
    }
  }
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
    yolo_model: fd.get("yolo_model") || "yolo26s.pt",
    inference_backend: fd.get("inference_backend") || "auto",
    target_fps: Number(fd.get("target_fps")),
    yolo_imgsz: Number(fd.get("yolo_imgsz")),
    confidence_threshold: Number(fd.get("confidence_threshold")),
    motion_gate_enabled: Boolean(settingsForm.elements["motion_gate_enabled"]?.checked),
    preview_max_width: Number(fd.get("preview_max_width")),
    dwell_seconds: Number(fd.get("dwell_seconds")),
    movement_threshold: Number(fd.get("movement_threshold")),
    annotation_scale: Number(fd.get("annotation_scale")),
    speed_capture_enabled: Boolean(settingsForm.elements["speed_capture_enabled"]?.checked),
    speed_limit_mph: Number(fd.get("speed_limit_mph")) || 25,
    alpr_enabled: Boolean(settingsForm.elements["alpr_enabled"]?.checked),
    person_zone_enabled: Boolean(settingsForm.elements["person_zone_enabled"]?.checked),
    person_zone_start_time: fd.get("person_zone_start_time") || "00:00",
    person_zone_end_time: fd.get("person_zone_end_time") || "05:00",
    person_zone_dwell_seconds: Number(fd.get("person_zone_dwell_seconds") ?? 2),
    person_zone_scope: fd.get("person_zone_scope") || "person_zones_or_frame",
    exempt_emergency_vehicles: Boolean(settingsForm.elements["exempt_emergency_vehicles"]?.checked),
    emergency_min_saturation: Number(fd.get("emergency_min_saturation")),
    emergency_min_value: Number(fd.get("emergency_min_value")),
    emergency_strobe_swing: Number(fd.get("emergency_strobe_swing")),
    emergency_fire_red_ratio: Number(fd.get("emergency_fire_red_ratio")),
    webhook_url: fd.get("webhook_url") || "",
    daily_report_enabled: Boolean(settingsForm.elements["daily_report_enabled"]?.checked),
    daily_report_time: fd.get("daily_report_time") || "23:55",
    ha_enabled: Boolean(settingsForm.elements["ha_enabled"]?.checked),
    ha_url: fd.get("ha_url") || "",
    ha_token: fd.get("ha_token") || "",
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
  closeZonesMenu({ restoreFocus: false });
  setBevDrawing(false);
  openModalDialog(birdseyeModal);
  refreshBirdseyeImage();
});
document.getElementById("btnCloseBirdseye").addEventListener("click", () => closeModalDialog(birdseyeModal));
birdseyeModal?.addEventListener("close", () => {
  setBevDrawing(false);
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
  closeModalDialog(birdseyeModal);
  setDrawingMode("calibrate");
});

document.getElementById("btnBevClear").addEventListener("click", async () => {
  await fetch("api/calibration", { method: "DELETE" });
  groundCalib = null;
  groundToImgMatrix = null;
  setBevDrawing(false);
  closeModalDialog(birdseyeModal);
  updateCalibrationUI();
});

document.addEventListener("keydown", (e) => {
  if (e.key !== "Escape") return;
  if (!supportsPopover && hideAnyOpenSettingTooltip()) {
    e.preventDefault();
    return;
  }
  if (
    !e.defaultPrevented &&
    !isModalOpen(settingsModal) &&
    !isModalOpen(snapshotModal) &&
    !isModalOpen(birdseyeModal) &&
    !isZonesMenuOpen() &&
    drawingMode
  ) {
    e.preventDefault();
    setDrawingMode(null);
  }
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
