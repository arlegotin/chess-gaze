import * as THREE from "./vendor/three.module.js";
import { OrbitControls } from "./vendor/OrbitControls.js";

const MODE_NAMES = {
  instant: "Instant",
  accumulated: "Accumulated",
};

const COLORS = {
  background: 0xf3f6fa,
  grid: 0xc8d2df,
  head: 0x667085,
  leftEye: 0x2f80c2,
  rightEye: 0xd46a5b,
  unigazeRay: 0x006d6f,
  currentHit: 0x4c1d95,
  accumulatedHit: 0xb7791f,
  monitorPlane: 0xd8dde5,
  monitorRectangle: 0x7c8795,
  warning: 0xb5532f,
};

const elements = {
  canvas: document.querySelector('[data-testid="scene-canvas"]'),
  fallbackStatus: document.querySelector(".fallback-status"),
  frameSlider: document.querySelector('[data-testid="frame-slider"]'),
  frameNumber: document.querySelector('[data-testid="frame-number"]'),
  frameLabel: document.querySelector('[data-testid="frame-label"]'),
  frameStatus: document.querySelector('[data-testid="frame-status"]'),
  frameIdentity: document.querySelector('[data-testid="frame-identity"]'),
  hitCount: document.querySelector('[data-testid="hit-count"]'),
  rayStatus: document.querySelector('[data-testid="ray-status"]'),
  hitStatus: document.querySelector('[data-testid="hit-status"]'),
  accumulatedStatus: document.querySelector('[data-testid="accumulated-status"]'),
  modeInstant: document.querySelector('[data-testid="mode-instant"]'),
  modeAccumulated: document.querySelector('[data-testid="mode-accumulated"]'),
  playPause: document.querySelector('[data-testid="play-pause"]'),
  stepPrev: document.querySelector('[data-testid="step-prev"]'),
  stepNext: document.querySelector('[data-testid="step-next"]'),
  toggles: {
    head: document.querySelector('[data-testid="toggle-head"]'),
    eyes: document.querySelector('[data-testid="toggle-eyes"]'),
    ray: document.querySelector('[data-testid="toggle-ray"]'),
    monitorPlane: document.querySelector('[data-testid="toggle-monitor-plane"]'),
    monitorRectangle: document.querySelector(
      '[data-testid="toggle-monitor-rectangle"]',
    ),
    extendedPlane: document.querySelector('[data-testid="toggle-extended-plane"]'),
    axes: document.querySelector('[data-testid="toggle-axes"]'),
    hitPoints: document.querySelector('[data-testid="toggle-hit-points"]'),
  },
};

const state = {
  sceneData: null,
  frameIndex: 0,
  mode: "instant",
  playing: false,
  playTimer: null,
};

const renderer = new THREE.WebGLRenderer({
  canvas: elements.canvas,
  antialias: true,
  alpha: false,
});
renderer.setClearColor(COLORS.background, 1);
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

const scene = new THREE.Scene();
scene.background = new THREE.Color(COLORS.background);

const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 20);
camera.position.set(0.35, 0.28, 1.6);

const controls = new OrbitControls(camera, elements.canvas);
controls.enableDamping = true;
controls.target.set(0, 0, 0.45);
controls.update();

scene.add(new THREE.HemisphereLight(0xffffff, 0xc7d1dd, 2.4));
const keyLight = new THREE.DirectionalLight(0xffffff, 1.8);
keyLight.position.set(1.5, 1.8, 2.2);
scene.add(keyLight);

const groups = {
  static: new THREE.Group(),
  current: new THREE.Group(),
  accumulated: new THREE.Group(),
};
scene.add(groups.static, groups.current, groups.accumulated);

const materials = {
  head: new THREE.MeshStandardMaterial({
    color: COLORS.head,
    transparent: true,
    opacity: 0.34,
    roughness: 0.82,
  }),
  leftEye: new THREE.MeshStandardMaterial({ color: COLORS.leftEye, roughness: 0.55 }),
  rightEye: new THREE.MeshStandardMaterial({
    color: COLORS.rightEye,
    roughness: 0.55,
  }),
  ray: new THREE.LineBasicMaterial({ color: COLORS.unigazeRay, linewidth: 2 }),
  warningRay: new THREE.LineBasicMaterial({ color: COLORS.warning, linewidth: 2 }),
  currentHit: new THREE.MeshStandardMaterial({
    color: COLORS.currentHit,
    roughness: 0.42,
  }),
  accumulatedHit: new THREE.MeshStandardMaterial({
    color: COLORS.accumulatedHit,
    roughness: 0.7,
  }),
  monitorPlane: new THREE.MeshBasicMaterial({
    color: COLORS.monitorPlane,
    transparent: true,
    opacity: 0.38,
    side: THREE.DoubleSide,
  }),
  extendedPlane: new THREE.MeshBasicMaterial({
    color: COLORS.monitorPlane,
    transparent: true,
    opacity: 0.18,
    side: THREE.DoubleSide,
  }),
  monitorRectangle: new THREE.LineBasicMaterial({ color: COLORS.monitorRectangle }),
};

function setStatus(message, isError = false) {
  elements.frameStatus.textContent = message;
  elements.fallbackStatus.textContent = message;
  elements.fallbackStatus.dataset.state = isError ? "error" : "ready";
}

function setControlState(disabled) {
  const controlsToToggle = [
    elements.frameSlider,
    elements.frameNumber,
    elements.modeInstant,
    elements.modeAccumulated,
    elements.playPause,
    elements.stepPrev,
    elements.stepNext,
    ...Object.values(elements.toggles),
  ];
  for (const control of controlsToToggle) {
    control.disabled = disabled;
  }
}

function vector(record) {
  if (!record) {
    return null;
  }
  return new THREE.Vector3(record.x, record.y, record.z);
}

function scaledRayEnd(origin, direction, length) {
  return origin.clone().add(direction.clone().normalize().multiplyScalar(length));
}

function clearGroup(group) {
  while (group.children.length > 0) {
    const child = group.children.pop();
    child.geometry?.dispose?.();
    group.remove(child);
  }
}

function addLine(group, start, end, material) {
  const geometry = new THREE.BufferGeometry().setFromPoints([start, end]);
  const line = new THREE.Line(geometry, material);
  group.add(line);
  return line;
}

function addSphere(group, position, radius, material) {
  const geometry = new THREE.SphereGeometry(radius, 24, 16);
  const sphere = new THREE.Mesh(geometry, material);
  sphere.position.copy(position);
  group.add(sphere);
  return sphere;
}

function addMonitorPlane(width, height, opacityMaterial, center, visible) {
  const geometry = new THREE.PlaneGeometry(width, height);
  const mesh = new THREE.Mesh(geometry, opacityMaterial);
  mesh.position.copy(center);
  mesh.visible = visible;
  groups.static.add(mesh);
  return mesh;
}

function addMonitorRectangle(width, height, center) {
  const halfWidth = width / 2;
  const halfHeight = height / 2;
  const points = [
    new THREE.Vector3(-halfWidth, -halfHeight, 0),
    new THREE.Vector3(halfWidth, -halfHeight, 0),
    new THREE.Vector3(halfWidth, halfHeight, 0),
    new THREE.Vector3(-halfWidth, halfHeight, 0),
    new THREE.Vector3(-halfWidth, -halfHeight, 0),
  ].map((point) => point.add(center));
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  const rectangle = new THREE.Line(geometry, materials.monitorRectangle);
  groups.static.add(rectangle);
  return rectangle;
}

function buildStaticScene() {
  clearGroup(groups.static);
  const plane = state.sceneData.monitor_plane;
  const center = vector(plane.center_scene_m) || new THREE.Vector3(0, 0, 0.7);
  const physicalWidth = plane.physical_width_m || plane.width_m || 0.6;
  const physicalHeight = plane.physical_height_m || plane.height_m || 0.34;
  const extendedWidth = plane.extended_width_m || physicalWidth * 3;
  const extendedHeight = plane.extended_height_m || physicalHeight * 3;

  addMonitorPlane(
    extendedWidth,
    extendedHeight,
    materials.extendedPlane,
    center,
    elements.toggles.extendedPlane.checked,
  ).userData.layer = "extendedPlane";
  addMonitorPlane(
    physicalWidth,
    physicalHeight,
    materials.monitorPlane,
    center,
    elements.toggles.monitorPlane.checked,
  ).userData.layer = "monitorPlane";
  addMonitorRectangle(physicalWidth, physicalHeight, center).userData.layer =
    "monitorRectangle";

  const grid = new THREE.GridHelper(1.4, 14, COLORS.grid, COLORS.grid);
  grid.position.set(0, -0.24, 0.42);
  grid.visible = elements.toggles.axes.checked;
  grid.userData.layer = "axes";
  groups.static.add(grid);

  const axes = new THREE.AxesHelper(0.35);
  axes.visible = elements.toggles.axes.checked;
  axes.userData.layer = "axes";
  groups.static.add(axes);
}

function applyStaticVisibility() {
  for (const child of groups.static.children) {
    const layer = child.userData.layer;
    if (layer === "monitorPlane") {
      child.visible = elements.toggles.monitorPlane.checked;
    } else if (layer === "monitorRectangle") {
      child.visible = elements.toggles.monitorRectangle.checked;
    } else if (layer === "extendedPlane") {
      child.visible = elements.toggles.extendedPlane.checked;
    } else if (layer === "axes") {
      child.visible = elements.toggles.axes.checked;
    }
  }
}

function currentFrame() {
  return state.sceneData.frames[state.frameIndex] || null;
}

function renderCurrentFrame() {
  clearGroup(groups.current);
  const frame = currentFrame();
  if (!frame) {
    return;
  }

  if (elements.toggles.head.checked && frame.head?.valid) {
    const center = vector(frame.head.scene_m || frame.head.ellipsoid_center_scene_m);
    if (center) {
      const radii = frame.head.ellipsoid_radii_m || [0.09, 0.12, 0.1];
      const geometry = new THREE.SphereGeometry(1, 32, 20);
      const head = new THREE.Mesh(geometry, materials.head);
      head.scale.set(radii[0], radii[1], radii[2]);
      head.position.copy(center);
      groups.current.add(head);
    }
  }

  if (elements.toggles.eyes.checked) {
    const leftEye = vector(frame.left_eye?.scene_m);
    const rightEye = vector(frame.right_eye?.scene_m);
    if (frame.left_eye?.valid && leftEye) {
      addSphere(groups.current, leftEye, 0.012, materials.leftEye);
    }
    if (frame.right_eye?.valid && rightEye) {
      addSphere(groups.current, rightEye, 0.012, materials.rightEye);
    }
  }

  const ray = frame.unigaze_ray;
  const hit = frame.main_monitor_hit;
  if (elements.toggles.ray.checked && ray?.valid) {
    const origin = vector(ray.scene_m || ray.origin_scene_m);
    const direction = vector(ray.direction_scene);
    if (origin && direction) {
      if (hit?.valid && hit.point_scene_m) {
        addLine(groups.current, origin, vector(hit.point_scene_m), materials.ray);
      } else {
        addLine(
          groups.current,
          origin,
          scaledRayEnd(origin, direction, 0.5),
          materials.warningRay,
        );
      }
    }
  }

  if (elements.toggles.hitPoints.checked && hit?.valid && hit.point_scene_m) {
    addSphere(groups.current, vector(hit.point_scene_m), 0.014, materials.currentHit);
  }
}

function renderAccumulatedHits() {
  clearGroup(groups.accumulated);
  if (
    state.mode !== "accumulated" ||
    !elements.toggles.hitPoints.checked ||
    !state.sceneData
  ) {
    return;
  }

  for (const hit of state.sceneData.valid_hit_points) {
    if (hit.frame_index <= state.frameIndex) {
      addSphere(
        groups.accumulated,
        vector(hit.point_scene_m),
        0.008,
        materials.accumulatedHit,
      );
    }
  }
}

function updateStatusPanel() {
  const frame = currentFrame();
  const total = state.sceneData?.frame_count || 0;
  const validHitsToFrame =
    state.sceneData?.valid_hit_points.filter(
      (hit) => hit.frame_index <= state.frameIndex,
    ).length || 0;

  elements.frameLabel.textContent =
    total > 0 ? `${state.frameIndex + 1} / ${total}` : "0 / 0";
  elements.frameIdentity.textContent = frame
    ? `${frame.frame_id} (${frame.frame_index})`
    : "-";
  elements.rayStatus.textContent = frame?.unigaze_ray?.valid
    ? "valid appearance_gaze ray"
    : frame?.unigaze_ray?.reason_invalid || "invalid";
  elements.hitStatus.textContent = frame?.main_monitor_hit?.valid
    ? "valid monitor hit"
    : frame?.main_monitor_hit?.reason_invalid || "invalid";
  elements.accumulatedStatus.textContent = `${validHitsToFrame} of ${
    state.sceneData?.valid_hit_points.length || 0
  }`;

  const reason = frame?.main_monitor_hit?.valid
    ? "monitor hit is valid"
    : frame?.main_monitor_hit?.reason_invalid || "monitor hit is invalid";
  setStatus(
    `${MODE_NAMES[state.mode]} mode. Frame ${state.frameIndex + 1} of ${total}: ${reason}.`,
    false,
  );
}

function setFrameIndex(index) {
  const maxIndex = Math.max(0, (state.sceneData?.frames.length || 1) - 1);
  state.frameIndex = Math.min(Math.max(Number(index) || 0, 0), maxIndex);
  elements.frameSlider.value = String(state.frameIndex);
  elements.frameNumber.value = String(state.frameIndex);
  renderCurrentFrame();
  renderAccumulatedHits();
  updateStatusPanel();
}

function setMode(mode) {
  state.mode = mode;
  elements.modeInstant.checked = mode === "instant";
  elements.modeAccumulated.checked = mode === "accumulated";
  renderAccumulatedHits();
  updateStatusPanel();
}

function setPlaying(playing) {
  state.playing = playing;
  elements.playPause.textContent = playing ? "Pause" : "Play";
  if (state.playTimer) {
    window.clearInterval(state.playTimer);
    state.playTimer = null;
  }
  if (!playing) {
    return;
  }
  state.playTimer = window.setInterval(() => {
    const maxIndex = Math.max(0, state.sceneData.frames.length - 1);
    setFrameIndex(state.frameIndex >= maxIndex ? 0 : state.frameIndex + 1);
  }, 120);
}

function resizeRenderer() {
  const rect = elements.canvas.getBoundingClientRect();
  const width = Math.max(1, Math.floor(rect.width));
  const height = Math.max(1, Math.floor(rect.height));
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}

function bindControls() {
  elements.frameSlider.addEventListener("input", () => {
    setFrameIndex(elements.frameSlider.value);
  });
  elements.frameNumber.addEventListener("change", () => {
    setFrameIndex(elements.frameNumber.value);
  });
  elements.stepPrev.addEventListener("click", () => {
    setFrameIndex(state.frameIndex - 1);
  });
  elements.stepNext.addEventListener("click", () => {
    setFrameIndex(state.frameIndex + 1);
  });
  elements.playPause.addEventListener("click", () => {
    setPlaying(!state.playing);
  });
  elements.modeInstant.addEventListener("change", () => {
    setMode("instant");
  });
  elements.modeAccumulated.addEventListener("change", () => {
    setMode("accumulated");
  });
  for (const toggle of Object.values(elements.toggles)) {
    toggle.addEventListener("change", () => {
      applyStaticVisibility();
      renderCurrentFrame();
      renderAccumulatedHits();
    });
  }
}

function applySceneData(sceneData) {
  state.sceneData = sceneData;
  const frames = Array.isArray(sceneData.frames) ? sceneData.frames : [];
  const maxIndex = Math.max(0, frames.length - 1);
  elements.frameSlider.max = String(maxIndex);
  elements.frameNumber.max = String(maxIndex);
  elements.hitCount.textContent = String(sceneData.valid_hit_points?.length || 0);
  setControlState(frames.length === 0);
  buildStaticScene();
  setFrameIndex(0);
}

async function loadSceneData() {
  if (window.__CHESS_GAZE_SCENE_DATA__) {
    return window.__CHESS_GAZE_SCENE_DATA__;
  }
  const response = await fetch("./scene-data.json", { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`scene-data.json returned ${response.status}`);
  }
  return response.json();
}

function animate() {
  resizeRenderer();
  controls.update();
  renderer.render(scene, camera);
  window.requestAnimationFrame(animate);
}

async function boot() {
  setControlState(true);
  bindControls();
  try {
    const sceneData = await loadSceneData();
    applySceneData(sceneData);
  } catch (error) {
    setControlState(true);
    setStatus(`Scene data unavailable: ${error.message}`, true);
  }
  animate();
}

window.addEventListener("resize", resizeRenderer);
boot();
