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
const HIT_AREA_VERTEX_COUNT = HIT_AREA_SEGMENTS + 1;
const HIT_AREA_INDEX_COUNT = HIT_AREA_SEGMENTS * 3;
const HIT_AREA_PLANE_OFFSET_M = 0.001;
const HIT_AREA_VECTOR_EPSILON = 1e-8;
const HIT_AREA_UNIT_CIRCLE = Array.from(
  { length: HIT_AREA_SEGMENTS },
  (_, index) => {
    const theta = (index / HIT_AREA_SEGMENTS) * Math.PI * 2;
    return { cos: Math.cos(theta), sin: Math.sin(theta) };
  },
);

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
  renderCache: {
    hitPoints: null,
    hitPointFrameIndices: [],
    hitAreas: null,
    hitAreaPatchFrameIndices: [],
    hitAreaPatchBases: [],
    hitAreaPositionAttribute: null,
    hitAreaRadiusScale: null,
  },
  renderRequested: false,
  animationFrameRequested: false,
  canvasWidth: 0,
  canvasHeight: 0,
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
  accumulatedHitPoints: new THREE.PointsMaterial({
    color: COLORS.accumulatedHit,
    size: 0.016,
    sizeAttenuation: true,
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

function removeAccumulatedObject(object) {
  if (!object) {
    return;
  }
  object.geometry?.dispose?.();
  groups.accumulated.remove(object);
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

function upperBoundFrameIndex(frameIndices, frameIndex) {
  let low = 0;
  let high = frameIndices.length;
  while (low < high) {
    const middle = Math.floor((low + high) / 2);
    if (frameIndices[middle] <= frameIndex) {
      low = middle + 1;
    } else {
      high = middle;
    }
  }
  return low;
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

function hitAreaRadiusScale(angularErrorDegreesValue) {
  const alphaRadians = (angularErrorDegreesValue * Math.PI) / 180;
  const radiusScale = Math.tan(alphaRadians);
  if (!Number.isFinite(radiusScale) || radiusScale <= 0) {
    return null;
  }
  return radiusScale;
}

function hitAreaPatchBasis(frame) {
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

  const minorScale = rayT;
  const majorScale = rayT / normalDirectionDot;
  if (!Number.isFinite(minorScale) || !Number.isFinite(majorScale) || rayT <= 0) {
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
  const patchCenter = center
    .clone()
    .add(normal.clone().multiplyScalar(HIT_AREA_PLANE_OFFSET_M));

  return {
    centerX: patchCenter.x,
    centerY: patchCenter.y,
    centerZ: patchCenter.z,
    majorX: orientedMajorAxis.x * majorScale,
    majorY: orientedMajorAxis.y * majorScale,
    majorZ: orientedMajorAxis.z * majorScale,
    minorX: minorAxis.x * minorScale,
    minorY: minorAxis.y * minorScale,
    minorZ: minorAxis.z * minorScale,
  };
}

function writeHitAreaPatchPositions(positions, offset, basis, radiusScale) {
  positions[offset] = basis.centerX;
  positions[offset + 1] = basis.centerY;
  positions[offset + 2] = basis.centerZ;

  for (let index = 0; index < HIT_AREA_SEGMENTS; index += 1) {
    const unit = HIT_AREA_UNIT_CIRCLE[index];
    const majorScale = unit.cos * radiusScale;
    const minorScale = unit.sin * radiusScale;
    const writeOffset = offset + (index + 1) * 3;
    positions[writeOffset] =
      basis.centerX + basis.majorX * majorScale + basis.minorX * minorScale;
    positions[writeOffset + 1] =
      basis.centerY + basis.majorY * majorScale + basis.minorY * minorScale;
    positions[writeOffset + 2] =
      basis.centerZ + basis.majorZ * majorScale + basis.minorZ * minorScale;
  }
}

function hitAreaPatchVertices(frame, angularErrorDegreesValue) {
  const radiusScale = hitAreaRadiusScale(angularErrorDegreesValue);
  const basis = hitAreaPatchBasis(frame);
  if (!radiusScale || !basis) {
    return null;
  }

  const vertices = new Array(HIT_AREA_VERTEX_COUNT * 3);
  writeHitAreaPatchPositions(vertices, 0, basis, radiusScale);
  return vertices;
}

function hitAreaGeometry(frame, angularErrorDegreesValue) {
  const vertices = hitAreaPatchVertices(frame, angularErrorDegreesValue);
  if (!vertices) {
    return null;
  }

  const indices = [];
  for (let index = 0; index < HIT_AREA_SEGMENTS; index += 1) {
    indices.push(0, index + 1, ((index + 1) % HIT_AREA_SEGMENTS) + 1);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(vertices, 3),
  );
  geometry.setIndex(indices);
  return geometry;
}

function buildAccumulatedHitPoints() {
  removeAccumulatedObject(state.renderCache.hitPoints);
  state.renderCache.hitPoints = null;
  state.renderCache.hitPointFrameIndices = [];

  const positions = [];
  const frameIndices = [];
  for (const hit of state.sceneData?.valid_hit_points || []) {
    const point = finiteVector(hit.point_scene_m);
    if (point && Number.isInteger(hit.frame_index)) {
      positions.push(point.x, point.y, point.z);
      frameIndices.push(hit.frame_index);
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(positions, 3),
  );
  geometry.setDrawRange(0, 0);

  const points = new THREE.Points(geometry, materials.accumulatedHitPoints);
  points.userData.layer = "hitPoints";
  points.visible = false;
  groups.accumulated.add(points);
  state.renderCache.hitPoints = points;
  state.renderCache.hitPointFrameIndices = frameIndices;
}

function buildAccumulatedHitAreaMesh() {
  removeAccumulatedObject(state.renderCache.hitAreas);
  state.renderCache.hitAreas = null;
  state.renderCache.hitAreaPatchFrameIndices = [];
  state.renderCache.hitAreaPatchBases = [];
  state.renderCache.hitAreaPositionAttribute = null;
  state.renderCache.hitAreaRadiusScale = null;

  const patchBases = [];
  const patchFrameIndices = [];

  for (const frame of state.sceneData?.frames || []) {
    const patchBasis = hitAreaPatchBasis(frame);
    if (!patchBasis || !Number.isInteger(frame.frame_index)) {
      continue;
    }

    patchBases.push(patchBasis);
    patchFrameIndices.push(frame.frame_index);
  }

  const positions = new Float32Array(
    patchBases.length * HIT_AREA_VERTEX_COUNT * 3,
  );
  const indices = new Uint32Array(patchBases.length * HIT_AREA_INDEX_COUNT);
  for (let patchIndex = 0; patchIndex < patchBases.length; patchIndex += 1) {
    const vertexOffset = patchIndex * HIT_AREA_VERTEX_COUNT;
    const indexOffset = patchIndex * HIT_AREA_INDEX_COUNT;
    for (let index = 0; index < HIT_AREA_SEGMENTS; index += 1) {
      const writeOffset = indexOffset + index * 3;
      indices[writeOffset] = vertexOffset;
      indices[writeOffset + 1] = vertexOffset + index + 1;
      indices[writeOffset + 2] =
        vertexOffset + ((index + 1) % HIT_AREA_SEGMENTS) + 1;
    }
  }

  const geometry = new THREE.BufferGeometry();
  const positionAttribute = new THREE.BufferAttribute(positions, 3);
  geometry.setAttribute("position", positionAttribute);
  geometry.setIndex(new THREE.BufferAttribute(indices, 1));
  geometry.setDrawRange(0, 0);

  const mesh = new THREE.Mesh(geometry, materials.hitArea);
  mesh.userData.layer = "hitArea";
  mesh.visible = false;
  groups.accumulated.add(mesh);
  state.renderCache.hitAreas = mesh;
  state.renderCache.hitAreaPatchFrameIndices = patchFrameIndices;
  state.renderCache.hitAreaPatchBases = patchBases;
  state.renderCache.hitAreaPositionAttribute = positionAttribute;
  updateAccumulatedHitAreaPositions();
}

function visibleHitPointCount() {
  return upperBoundFrameIndex(
    state.renderCache.hitPointFrameIndices,
    state.frameIndex,
  );
}

function visibleHitAreaTriangleIndexCount() {
  if (!hitAreaRadiusScale(angularErrorDegrees())) {
    return 0;
  }
  const visiblePatchCount = upperBoundFrameIndex(
    state.renderCache.hitAreaPatchFrameIndices,
    state.frameIndex,
  );
  return visiblePatchCount * HIT_AREA_INDEX_COUNT;
}

function updateAccumulatedVisibility() {
  const accumulatedVisible = state.mode === "accumulated";

  if (state.renderCache.hitPoints) {
    const visibleHitPointCountValue =
      accumulatedVisible && elements.toggles.hitPoints.checked
        ? visibleHitPointCount()
        : 0;
    state.renderCache.hitPoints.geometry.setDrawRange(0, visibleHitPointCountValue);
    state.renderCache.hitPoints.visible =
      accumulatedVisible &&
      elements.toggles.hitPoints.checked &&
      visibleHitPointCountValue > 0;
  }

  if (state.renderCache.hitAreas) {
    const visibleHitAreaTriangleIndexCountValue =
      accumulatedVisible && elements.toggles.hitArea.checked
        ? visibleHitAreaTriangleIndexCount()
        : 0;
    state.renderCache.hitAreas.geometry.setDrawRange(0, visibleHitAreaTriangleIndexCountValue);
    state.renderCache.hitAreas.visible =
      accumulatedVisible &&
      elements.toggles.hitArea.checked &&
      visibleHitAreaTriangleIndexCountValue > 0;
  }
}

function updateAccumulatedHitAreaPositions() {
  const attribute = state.renderCache.hitAreaPositionAttribute;
  if (!attribute) {
    return;
  }
  const radiusScale = hitAreaRadiusScale(angularErrorDegrees()) || 0;
  if (state.renderCache.hitAreaRadiusScale === radiusScale) {
    return;
  }
  const positions = attribute.array;
  for (
    let patchIndex = 0;
    patchIndex < state.renderCache.hitAreaPatchBases.length;
    patchIndex += 1
  ) {
    writeHitAreaPatchPositions(
      positions,
      patchIndex * HIT_AREA_VERTEX_COUNT * 3,
      state.renderCache.hitAreaPatchBases[patchIndex],
      radiusScale,
    );
  }
  attribute.needsUpdate = true;
  state.renderCache.hitAreaRadiusScale = radiusScale;
}

function updateAccumulatedHitAreasForAngularError() {
  if (!state.sceneData) {
    return;
  }
  if (!state.renderCache.hitAreas) {
    buildAccumulatedHitAreaMesh();
    return;
  }
  updateAccumulatedHitAreaPositions();
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
  return state.sceneData?.frames?.[state.frameIndex] || null;
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

function updateStatusPanel() {
  const frame = currentFrame();
  const total = state.sceneData?.frame_count || 0;
  let validHitsToFrame = 0;
  if (state.sceneData) {
    validHitsToFrame = visibleHitPointCount();
  }

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
  updateAccumulatedVisibility();
  updateStatusPanel();
  requestRender();
}

function setMode(mode) {
  state.mode = mode;
  elements.modeInstant.checked = mode === "instant";
  elements.modeAccumulated.checked = mode === "accumulated";
  updateAccumulatedVisibility();
  updateStatusPanel();
  requestRender();
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
  if (width === state.canvasWidth && height === state.canvasHeight) {
    return;
  }
  state.canvasWidth = width;
  state.canvasHeight = height;
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  requestRender();
}

function requestRender() {
  state.renderRequested = true;
  if (!state.animationFrameRequested) {
    state.animationFrameRequested = true;
    window.requestAnimationFrame(renderFrame);
  }
}

function renderFrame() {
  state.animationFrameRequested = false;
  const controlsNeedRender = controls.enableDamping && controls.update();
  if (state.renderRequested || state.playing || controlsNeedRender) {
    state.renderRequested = false;
    renderer.render(scene, camera);
  }
  if (state.playing || controlsNeedRender || state.renderRequested) {
    requestRender();
  }
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
    requestRender();
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
    updateAccumulatedHitAreasForAngularError();
    updateAccumulatedVisibility();
    requestRender();
  });
  elements.hitAreaOpacity.addEventListener("input", () => {
    updateHitAreaOpacityLabel();
    applyHitAreaOpacity();
    requestRender();
  });
  for (const toggle of Object.values(elements.toggles)) {
    toggle.addEventListener("change", () => {
      applyStaticVisibility();
      renderCurrentFrame();
      updateAccumulatedVisibility();
      requestRender();
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
  buildAccumulatedHitPoints();
  updateAccumulatedHitAreasForAngularError();
  resizeRenderer();
  setFrameIndex(0);
  requestRender();
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
  requestRender();
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

controls.addEventListener("change", requestRender);
const resizeObserver =
  typeof ResizeObserver === "function" ? new ResizeObserver(resizeRenderer) : null;
if (resizeObserver) {
  resizeObserver.observe(elements.canvas);
}
window.addEventListener("resize", resizeRenderer);
boot();
