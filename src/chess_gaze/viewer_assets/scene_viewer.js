const elements = {
  canvas: document.querySelector('[data-testid="scene-canvas"]'),
  fallbackStatus: document.querySelector(".fallback-status"),
  frameSlider: document.querySelector('[data-testid="frame-slider"]'),
  frameStatus: document.querySelector('[data-testid="frame-status"]'),
  hitCount: document.querySelector('[data-testid="hit-count"]'),
  modeInstant: document.querySelector('[data-testid="mode-instant"]'),
  modeAccumulated: document.querySelector('[data-testid="mode-accumulated"]'),
  playPause: document.querySelector('[data-testid="play-pause"]'),
  stepPrev: document.querySelector('[data-testid="step-prev"]'),
  stepNext: document.querySelector('[data-testid="step-next"]'),
};

function setStatus(message, isError = false) {
  elements.frameStatus.textContent = message;
  elements.fallbackStatus.textContent = message;
  elements.fallbackStatus.dataset.state = isError ? "error" : "ready";
}

function setControlState(disabled) {
  elements.frameSlider.disabled = disabled;
  elements.modeInstant.disabled = disabled;
  elements.modeAccumulated.disabled = disabled;
  elements.playPause.disabled = disabled;
  elements.stepPrev.disabled = disabled;
  elements.stepNext.disabled = disabled;
}

function drawShellCanvas() {
  const rect = elements.canvas.getBoundingClientRect();
  const scale = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.floor(rect.width * scale));
  const height = Math.max(1, Math.floor(rect.height * scale));
  elements.canvas.width = width;
  elements.canvas.height = height;

  const context = elements.canvas.getContext("2d");
  if (!context) {
    return;
  }

  context.scale(scale, scale);
  context.fillStyle = "#f8fafc";
  context.fillRect(0, 0, rect.width, rect.height);
  context.strokeStyle = "#cbd5e1";
  context.lineWidth = 1;

  for (let x = 0; x < rect.width; x += 32) {
    context.beginPath();
    context.moveTo(x, 0);
    context.lineTo(x, rect.height);
    context.stroke();
  }

  for (let y = 0; y < rect.height; y += 32) {
    context.beginPath();
    context.moveTo(0, y);
    context.lineTo(rect.width, y);
    context.stroke();
  }
}

function applySceneMetadata(sceneData) {
  const frames = Array.isArray(sceneData.frames) ? sceneData.frames : [];
  const hitCount = frames.filter(
    (frame) => frame.monitor_hit !== null && frame.monitor_hit !== undefined,
  ).length;
  elements.frameSlider.max = String(Math.max(0, frames.length - 1));
  elements.hitCount.textContent = String(hitCount);
  setControlState(frames.length === 0);

  if (frames.length === 0) {
    setStatus("Scene data loaded without frame records.", true);
    return;
  }

  setStatus(`Scene data loaded. Frame 1 of ${frames.length}.`);
}

async function loadSceneData() {
  try {
    const response = await fetch("./scene-data.json", { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`scene-data.json returned ${response.status}`);
    }
    const sceneData = await response.json();
    applySceneMetadata(sceneData);
    return true;
  } catch (error) {
    setControlState(true);
    setStatus(`Scene data unavailable: ${error.message}`, true);
    return false;
  }
}

function loadViewerModules() {
  return Promise.all([
    import("./vendor/three.module.js"),
    import("./vendor/OrbitControls.js"),
  ]);
}

async function boot() {
  setControlState(true);
  drawShellCanvas();
  const hasSceneData = await loadSceneData();
  if (!hasSceneData) {
    return;
  }

  try {
    await loadViewerModules();
  } catch (error) {
    setStatus(`Viewer modules unavailable: ${error.message}`, true);
  }
}

window.addEventListener("resize", drawShellCanvas);
boot();
