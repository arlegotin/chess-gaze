import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const MODE_NAMES = {
  instant: "Instant",
  accumulated: "Accumulated",
};

const DEFAULT_HIT_AREA_ANGULAR_ERROR_DEGREES = 8;
const HIT_AREA_MIN_ANGULAR_ERROR_DEGREES = 0;
const HIT_AREA_MAX_ANGULAR_ERROR_DEGREES = 12;
const DEFAULT_HIT_AREA_OPACITY = 0.24;
const HIT_AREA_MIN_OPACITY = 0;
const HIT_AREA_MAX_OPACITY = 1;
const HIT_AREA_SEGMENTS = 72;
const HIT_AREA_PLANE_OFFSET_M = 0.001;
const HIT_AREA_VECTOR_EPSILON = 1e-8;

const COLORS = {
  background: 0xf3f6fa,
  grid: 0xc8d2df,
  head: 0x667085,
  leftEye: 0x2f80c2,
  rightEye: 0xd46a5b,
  unigazeRay: 0x006d6f,
  currentHit: 0x4c1d95,
  accumulatedHit: 0xb7791f,
  hitArea: 0xc43d7a,
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
  hitAreaErrorDegrees: document.querySelector(
    '[data-testid="hit-area-error-degrees"]',
  ),
  hitAreaErrorLabel: document.querySelector('[data-testid="hit-area-error-label"]'),
  hitAreaOpacity: document.querySelector('[data-testid="hit-area-opacity"]'),
  hitAreaOpacityLabel: document.querySelector(
    '[data-testid="hit-area-opacity-label"]',
  ),
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
    hitArea: document.querySelector('[data-testid="toggle-hit-area"]'),
  },
};

const state = {
  sceneData: null,
  frameIndex: 0,
  mode: "accumulated",
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
camera.position.set(0, 0.28, -1.6);

const controls = new OrbitControls(camera, elements.canvas);
controls.enableDamping = true;
controls.target.set(0, 0, 0);
controls.update();

scene.add(new THREE.HemisphereLight(0xffffff, 0xc7d1dd, 2.4));
const keyLight = new THREE.DirectionalLight(0xffffff, 1.8);
keyLight.position.set(-1.0, 1.8, -2.2);
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
    depthWrite: false,
  }),
  extendedPlane: new THREE.MeshBasicMaterial({
    color: COLORS.monitorPlane,
    transparent: true,
    opacity: 0.18,
    side: THREE.DoubleSide,
    depthWrite: false,
  }),
  monitorRectangle: new THREE.LineBasicMaterial({ color: COLORS.monitorRectangle }),
  hitArea: new THREE.MeshBasicMaterial({
    color: COLORS.hitArea,
    transparent: true,
    opacity: DEFAULT_HIT_AREA_OPACITY,
    side: THREE.DoubleSide,
    depthWrite: false,
  }),
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
    elements.hitAreaErrorDegrees,
    elements.hitAreaOpacity,
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

function finiteNumber(value) {
  return typeof value === "number" && Number.isFinite(value);
}

function finiteVector(record) {
  const candidate = vector(record);
  if (
    !candidate ||
    !Number.isFinite(candidate.x) ||
    !Number.isFinite(candidate.y) ||
    !Number.isFinite(candidate.z)
  ) {
    return null;
  }
  return candidate;
}

function normalizedVector(record) {
  const candidate = finiteVector(record);
  if (!candidate || candidate.lengthSq() <= HIT_AREA_VECTOR_EPSILON ** 2) {
    return null;
  }
  return candidate.normalize();
}

function sceneBasisVectors() {
  const basis = state.sceneData?.axis_basis;
  const right = normalizedVector(basis?.right_camera);
  const up = normalizedVector(basis?.up_camera);
  const back = normalizedVector(basis?.back_camera);
  if (!right || !up || !back) {
    return null;
  }
  return { right, up, back };
}

function cameraDirectionToScene(cameraDirection) {
  const basis = sceneBasisVectors();
  if (!basis || !cameraDirection) {
    return null;
  }
  const sceneDirection = new THREE.Vector3(
    cameraDirection.dot(basis.right),
    cameraDirection.dot(basis.up),
    cameraDirection.dot(basis.back),
  );
  if (sceneDirection.lengthSq() <= HIT_AREA_VECTOR_EPSILON ** 2) {
    return null;
  }
  return sceneDirection.normalize();
}

function monitorNormalScene() {
  return (
    cameraDirectionToScene(normalizedVector(state.sceneData?.monitor_plane?.normal_camera)) ||
    new THREE.Vector3(0, 0, 1)
  );
}

function monitorRightScene() {
  return (
    cameraDirectionToScene(normalizedVector(state.sceneData?.monitor_plane?.right_camera)) ||
    new THREE.Vector3(1, 0, 0)
  );
}

function rayDirectionScene(frame) {
  return (
    normalizedVector(frame?.unigaze_ray?.direction_scene) ||
    cameraDirectionToScene(normalizedVector(frame?.unigaze_ray?.direction_camera))
  );
}

function angularErrorDegrees() {
  const rawValue = Number(elements.hitAreaErrorDegrees.value);
  const value = Number.isFinite(rawValue)
    ? rawValue
    : DEFAULT_HIT_AREA_ANGULAR_ERROR_DEGREES;
  return Math.min(
    HIT_AREA_MAX_ANGULAR_ERROR_DEGREES,
    Math.max(HIT_AREA_MIN_ANGULAR_ERROR_DEGREES, value),
  );
}

function updateHitAreaErrorLabel() {
  const degrees = angularErrorDegrees();
  elements.hitAreaErrorDegrees.value = String(degrees);
  const labelValue = Number.isInteger(degrees) ? String(degrees) : degrees.toFixed(1);
  elements.hitAreaErrorLabel.textContent = `${labelValue} deg`;
}

function hitAreaOpacity() {
  const rawValue = Number(elements.hitAreaOpacity.value);
  const value = Number.isFinite(rawValue) ? rawValue : DEFAULT_HIT_AREA_OPACITY;
  return Math.min(HIT_AREA_MAX_OPACITY, Math.max(HIT_AREA_MIN_OPACITY, value));
}

function updateHitAreaOpacityLabel() {
  const opacity = hitAreaOpacity();
  elements.hitAreaOpacity.value = String(opacity);
  elements.hitAreaOpacityLabel.textContent = `${Math.round(opacity * 100)}%`;
}

function applyHitAreaOpacity() {
  materials.hitArea.opacity = hitAreaOpacity();
  materials.hitArea.needsUpdate = true;
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

function addHitArea(group, geometry) {
  const mesh = new THREE.Mesh(geometry, materials.hitArea);
  group.add(mesh);
  return mesh;
}

function axisInMonitorPlane(preferredAxis, normal) {
  if (!preferredAxis) {
    return null;
  }
  const axis = preferredAxis
    .clone()
    .sub(normal.clone().multiplyScalar(preferredAxis.dot(normal)));
  if (axis.lengthSq() > HIT_AREA_VECTOR_EPSILON ** 2) {
    return axis.normalize();
  }
  return null;
}

function fallbackAxisInMonitorPlane(normal) {
  const fallback = Math.abs(normal.z) < 0.9
    ? new THREE.Vector3(0, 0, 1)
    : new THREE.Vector3(1, 0, 0);
  return fallback
    .sub(normal.clone().multiplyScalar(fallback.dot(normal)))
    .normalize();
}

function hitAreaGeometry(frame, angularErrorDegreesValue) {
  const hit = frame?.main_monitor_hit;
  if (!hit?.valid || !frame?.unigaze_ray?.valid || !hit.point_scene_m) {
    return null;
  }

  const center = finiteVector(hit.point_scene_m);
  const rayT = hit.ray_t_m ?? hit.t;
  const direction = rayDirectionScene(frame);
  const normal = monitorNormalScene();
  if (!center || !finiteNumber(rayT) || rayT < 0 || !direction || !normal) {
    return null;
  }

  const normalDirectionDot = Math.abs(normal.dot(direction));
  if (
    !Number.isFinite(normalDirectionDot) ||
    normalDirectionDot <= HIT_AREA_VECTOR_EPSILON
  ) {
    return null;
  }

  const alphaRadians = (angularErrorDegreesValue * Math.PI) / 180;
  const minorRadius = rayT * Math.tan(alphaRadians);
  const majorRadius = minorRadius / normalDirectionDot;
  if (
    !Number.isFinite(minorRadius) ||
    !Number.isFinite(majorRadius) ||
    minorRadius <= 0 ||
    majorRadius <= 0
  ) {
    return null;
  }

  const projectedDirection = direction.clone().sub(
    normal.clone().multiplyScalar(direction.dot(normal)),
  );
  const orientedMajorAxis =
    axisInMonitorPlane(projectedDirection, normal) ||
    axisInMonitorPlane(monitorRightScene(), normal) ||
    fallbackAxisInMonitorPlane(normal);
  const minorAxis = new THREE.Vector3()
    .crossVectors(normal, orientedMajorAxis)
    .normalize();
  const patchCenter = center.add(normal.clone().multiplyScalar(HIT_AREA_PLANE_OFFSET_M));
  const vertices = [patchCenter.x, patchCenter.y, patchCenter.z];
  const indices = [];

  for (let index = 0; index < HIT_AREA_SEGMENTS; index += 1) {
    const theta = (index / HIT_AREA_SEGMENTS) * Math.PI * 2;
    const point = patchCenter
      .clone()
      .add(
        orientedMajorAxis
          .clone()
          .multiplyScalar(Math.cos(theta) * majorRadius),
      )
      .add(minorAxis.clone().multiplyScalar(Math.sin(theta) * minorRadius));
    vertices.push(point.x, point.y, point.z);
    indices.push(0, index + 1, ((index + 1) % HIT_AREA_SEGMENTS) + 1);
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(vertices, 3),
  );
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  return geometry;
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

  renderCurrentHitArea(frame);

  if (elements.toggles.hitPoints.checked && hit?.valid && hit.point_scene_m) {
    addSphere(groups.current, vector(hit.point_scene_m), 0.014, materials.currentHit);
  }
}

function renderCurrentHitArea(frame) {
  if (!elements.toggles.hitArea.checked) {
    return;
  }
  const geometry = hitAreaGeometry(frame, angularErrorDegrees());
  if (geometry) {
    addHitArea(groups.current, geometry);
  }
}

function renderAccumulatedHitAreas() {
  if (!elements.toggles.hitArea.checked || !state.sceneData) {
    return;
  }

  for (const frame of state.sceneData.frames.slice(0, state.frameIndex + 1)) {
    const geometry = hitAreaGeometry(frame, angularErrorDegrees());
    if (geometry) {
      addHitArea(groups.accumulated, geometry);
    }
  }
}

function renderAccumulatedHits() {
  clearGroup(groups.accumulated);
  if (state.mode !== "accumulated" || !state.sceneData) {
    return;
  }

  if (elements.toggles.hitPoints.checked) {
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

  renderAccumulatedHitAreas();
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
  elements.hitAreaErrorDegrees.addEventListener("input", () => {
    updateHitAreaErrorLabel();
    renderCurrentFrame();
    renderAccumulatedHits();
  });
  elements.hitAreaOpacity.addEventListener("input", () => {
    updateHitAreaOpacityLabel();
    applyHitAreaOpacity();
    renderCurrentFrame();
    renderAccumulatedHits();
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
  updateHitAreaErrorLabel();
  updateHitAreaOpacityLabel();
  applyHitAreaOpacity();
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
