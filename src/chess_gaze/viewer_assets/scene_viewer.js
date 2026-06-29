import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const MODE_NAMES = {
  instant: "Instant",
  accumulated: "Accumulated",
};

const DEFAULT_SPHERE_RADIUS_M = 0.7;
const SPHERE_MIN_RADIUS_M = 0.35;
const SPHERE_MAX_RADIUS_M = 1.2;
const SPHERE_RADIUS_STEP_M = 0.01;
const SPHERE_SURFACE_OFFSET_M = 0.002;

const DEFAULT_HIT_AREA_ANGULAR_ERROR_DEGREES = 8;
const HIT_AREA_MIN_ANGULAR_ERROR_DEGREES = 0;
const HIT_AREA_MAX_ANGULAR_ERROR_DEGREES = 12;
const DEFAULT_HIT_AREA_OPACITY = 0.24;
const HIT_AREA_MIN_OPACITY = 0;
const HIT_AREA_MAX_OPACITY = 1;
const HIT_AREA_SEGMENTS = 72;
const HIT_AREA_VERTEX_COUNT = HIT_AREA_SEGMENTS + 1;
const HIT_AREA_INDEX_COUNT = HIT_AREA_SEGMENTS * 3;
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
  gazeSphere: 0xd8dde5,
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
  controls: {
    gazeSphere: document.querySelector('[data-testid="toggle-gaze-sphere"]'),
    sphereRadius: document.querySelector('[data-testid="sphere-radius-m"]'),
    sphereRadiusLabel: document.querySelector('[data-testid="sphere-radius-label"]'),
  },
  toggles: {
    head: document.querySelector('[data-testid="toggle-head"]'),
    eyes: document.querySelector('[data-testid="toggle-eyes"]'),
    ray: document.querySelector('[data-testid="toggle-ray"]'),
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
    gazeSphere: null,
    hitPoints: null,
    hitPointFrameIndices: [],
    hitPointRecords: [],
    hitPointPositionAttribute: null,
    hitPointRadius: null,
    hitAreas: null,
    hitAreaPatchFrameIndices: [],
    hitAreaPatchBases: [],
    hitAreaPositionAttribute: null,
    hitAreaRadiusScale: null,
    hitAreaSphereRadius: null,
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
  gazeSphere: new THREE.MeshBasicMaterial({
    color: COLORS.gazeSphere,
    transparent: true,
    opacity: 0.14,
    depthWrite: false,
    side: THREE.DoubleSide,
  }),
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
    ...Object.values(elements.controls),
    ...Object.values(elements.toggles),
  ];
  for (const control of controlsToToggle) {
    if (control) {
      control.disabled = disabled;
    }
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

function rayDirectionScene(frame) {
  return (
    normalizedVector(frame?.unigaze_ray?.direction_scene) ||
    cameraDirectionToScene(normalizedVector(frame?.unigaze_ray?.direction_camera))
  );
}

function sphereCenterScene() {
  return finiteVector(state.sceneData?.gaze_sphere?.center_scene_m) || new THREE.Vector3(0, 0, 0);
}

function sphereRadiusMeters() {
  const parsed = Number.parseFloat(elements.controls.sphereRadius?.value);
  if (!Number.isFinite(parsed)) {
    return state.sceneData?.gaze_sphere?.radius_m || DEFAULT_SPHERE_RADIUS_M;
  }
  return Math.min(SPHERE_MAX_RADIUS_M, Math.max(SPHERE_MIN_RADIUS_M, parsed));
}

function updateSphereRadiusLabel() {
  const radius = sphereRadiusMeters();
  if (elements.controls.sphereRadius) {
    elements.controls.sphereRadius.value = radius.toFixed(2);
  }
  if (elements.controls.sphereRadiusLabel) {
    elements.controls.sphereRadiusLabel.textContent = `${radius.toFixed(2)} m`;
  }
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

function unitOrthogonalVector(direction) {
  const fallback = Math.abs(direction.z) < 0.9
    ? new THREE.Vector3(0, 0, 1)
    : new THREE.Vector3(1, 0, 0);
  const tangent = new THREE.Vector3().crossVectors(direction, fallback);
  if (tangent.lengthSq() <= HIT_AREA_VECTOR_EPSILON ** 2) {
    return new THREE.Vector3(0, 1, 0);
  }
  return tangent.normalize();
}

function intersectRayWithSphere(origin, direction, radius, center = sphereCenterScene()) {
  if (!origin || !direction || !Number.isFinite(radius) || radius <= 0 || !center) {
    return null;
  }
  const normalizedDirection = direction.clone().normalize();
  if (normalizedDirection.lengthSq() <= HIT_AREA_VECTOR_EPSILON ** 2) {
    return null;
  }

  const localOrigin = origin.clone().sub(center);
  const a = normalizedDirection.dot(normalizedDirection);
  const b = 2 * localOrigin.dot(normalizedDirection);
  const c = localOrigin.dot(localOrigin) - radius * radius;
  const discriminant = b * b - 4 * a * c;
  if (!Number.isFinite(discriminant) || discriminant < -1e-9) {
    return null;
  }

  const sqrtDiscriminant = Math.sqrt(Math.max(0, discriminant));
  const denominator = 2 * a;
  if (Math.abs(denominator) <= 1e-9) {
    return null;
  }

  const roots = [(-b - sqrtDiscriminant) / denominator, (-b + sqrtDiscriminant) / denominator]
    .filter((root) => Number.isFinite(root) && root >= -1e-9);
  if (!roots.length) {
    return null;
  }

  const nearestRoot = roots.reduce((best, root) => Math.min(best, root), Number.POSITIVE_INFINITY);
  const rayT = Math.max(0, nearestRoot);
  return center.clone().add(localOrigin).add(normalizedDirection.multiplyScalar(rayT));
}

function sphereHitForFrame(frame) {
  if (!frame?.sphere_hit?.valid) {
    return {
      valid: false,
      point: null,
      reason: frame?.sphere_hit?.reason_invalid || "invalid sphere hit",
    };
  }
  const radius = sphereRadiusMeters();
  const origin = finiteVector(frame?.unigaze_ray?.origin_scene_m || frame?.unigaze_ray?.scene_m);
  const direction = rayDirectionScene(frame);
  const projected = intersectRayWithSphere(origin, direction, radius);
  if (!projected) {
    return {
      valid: false,
      point: null,
      reason: "no sphere intersection at selected radius",
    };
  }
  return { valid: true, point: projected, reason: null };
}

function surfaceOffsetPoint(point) {
  if (!point) {
    return null;
  }
  const normal = point.clone().sub(sphereCenterScene());
  if (normal.lengthSq() <= HIT_AREA_VECTOR_EPSILON ** 2) {
    return point.clone();
  }
  return point.clone().add(normal.normalize().multiplyScalar(SPHERE_SURFACE_OFFSET_M));
}

function hitAreaRadiusScale(angularErrorDegreesValue) {
  const alphaRadians = (angularErrorDegreesValue * Math.PI) / 180;
  const radiusScale = Math.tan(alphaRadians);
  if (!Number.isFinite(radiusScale) || radiusScale <= 0) {
    return null;
  }
  return radiusScale;
}

function sphereHitAreaPatchBasis(frame) {
  if (!frame?.sphere_hit?.valid || !frame?.unigaze_ray?.valid) {
    return null;
  }
  const origin = finiteVector(frame?.unigaze_ray?.origin_scene_m || frame?.unigaze_ray?.scene_m);
  const direction = rayDirectionScene(frame);
  if (!origin || !direction) {
    return null;
  }

  const tangent = unitOrthogonalVector(direction);
  const bitangent = new THREE.Vector3().crossVectors(direction, tangent).normalize();
  if (bitangent.lengthSq() <= HIT_AREA_VECTOR_EPSILON ** 2) {
    return null;
  }

  return {
    origin,
    direction: direction.clone().normalize(),
    tangent,
    bitangent,
  };
}

function writeSphereHitAreaPatchPositions(positions, offset, basis, radiusScale) {
  const centerHit = intersectRayWithSphere(
    basis.origin,
    basis.direction,
    sphereRadiusMeters(),
  );
  if (!centerHit) {
    return false;
  }

  const centerPoint = surfaceOffsetPoint(centerHit);
  positions[offset] = centerPoint.x;
  positions[offset + 1] = centerPoint.y;
  positions[offset + 2] = centerPoint.z;

  for (let index = 0; index < HIT_AREA_SEGMENTS; index += 1) {
    const unit = HIT_AREA_UNIT_CIRCLE[index];
    const boundaryDirection = basis.direction
      .clone()
      .add(basis.tangent.clone().multiplyScalar(unit.cos * radiusScale))
      .add(basis.bitangent.clone().multiplyScalar(unit.sin * radiusScale))
      .normalize();
    const boundaryHit = intersectRayWithSphere(
      basis.origin,
      boundaryDirection,
      sphereRadiusMeters(),
    );
    if (!boundaryHit) {
      return false;
    }
    const writeOffset = offset + (index + 1) * 3;
    const boundaryPoint = surfaceOffsetPoint(boundaryHit);
    positions[writeOffset] = boundaryPoint.x;
    positions[writeOffset + 1] = boundaryPoint.y;
    positions[writeOffset + 2] = boundaryPoint.z;
  }
  return true;
}

function hitAreaGeometry(frame, angularErrorDegreesValue) {
  const radiusScale = hitAreaRadiusScale(angularErrorDegreesValue);
  const basis = sphereHitAreaPatchBasis(frame);
  if (!radiusScale || !basis) {
    return null;
  }

  const vertices = new Float32Array(HIT_AREA_VERTEX_COUNT * 3);
  const wrotePatch = writeSphereHitAreaPatchPositions(vertices, 0, basis, radiusScale);
  if (!wrotePatch) {
    return null;
  }

  const indices = [];
  for (let index = 0; index < HIT_AREA_SEGMENTS; index += 1) {
    indices.push(0, index + 1, ((index + 1) % HIT_AREA_SEGMENTS) + 1);
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(vertices, 3));
  geometry.setIndex(indices);
  return geometry;
}

function buildGazeSphere() {
  const geometry = new THREE.SphereGeometry(1, 48, 24);
  const material = materials.gazeSphere;
  const mesh = new THREE.Mesh(geometry, material);
  mesh.position.copy(sphereCenterScene());
  mesh.scale.setScalar(sphereRadiusMeters());
  mesh.userData.layer = "gazeSphere";
  mesh.visible = Boolean(elements.controls.gazeSphere?.checked);
  groups.static.add(mesh);
  state.renderCache.gazeSphere = mesh;
  return mesh;
}

function buildAccumulatedHitPoints() {
  removeAccumulatedObject(state.renderCache.hitPoints);
  state.renderCache.hitPoints = null;
  state.renderCache.hitPointFrameIndices = [];
  state.renderCache.hitPointRecords = [];
  state.renderCache.hitPointPositionAttribute = null;
  state.renderCache.hitPointRadius = null;

  const hitPointRecords = [];
  for (const frame of state.sceneData?.frames || []) {
    if (!frame?.sphere_hit?.valid || !Number.isInteger(frame.frame_index)) {
      continue;
    }
    hitPointRecords.push(frame);
  }

  const positions = new Float32Array(hitPointRecords.length * 3);
  const geometry = new THREE.BufferGeometry();
  const positionAttribute = new THREE.BufferAttribute(positions, 3);
  geometry.setAttribute("position", positionAttribute);
  geometry.setDrawRange(0, 0);

  const points = new THREE.Points(geometry, materials.accumulatedHitPoints);
  points.userData.layer = "hitPoints";
  points.visible = false;
  groups.accumulated.add(points);

  state.renderCache.hitPoints = points;
  state.renderCache.hitPointFrameIndices = hitPointRecords.map((frame) => frame.frame_index);
  state.renderCache.hitPointRecords = hitPointRecords;
  state.renderCache.hitPointPositionAttribute = positionAttribute;
  updateAccumulatedHitPoints();
}

function buildAccumulatedHitAreaMesh() {
  removeAccumulatedObject(state.renderCache.hitAreas);
  state.renderCache.hitAreas = null;
  state.renderCache.hitAreaPatchFrameIndices = [];
  state.renderCache.hitAreaPatchBases = [];
  state.renderCache.hitAreaPositionAttribute = null;
  state.renderCache.hitAreaRadiusScale = null;
  state.renderCache.hitAreaSphereRadius = null;

  const patchBases = [];
  for (const frame of state.sceneData?.frames || []) {
    const patchBasis = sphereHitAreaPatchBasis(frame);
    if (!patchBasis || !Number.isInteger(frame.frame_index)) {
      continue;
    }
    patchBases.push({ frameIndex: frame.frame_index, basis: patchBasis });
  }

  const positions = new Float32Array(patchBases.length * HIT_AREA_VERTEX_COUNT * 3);
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
  mesh.frustumCulled = false;
  mesh.visible = false;
  groups.accumulated.add(mesh);

  state.renderCache.hitAreas = mesh;
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

function updateAccumulatedHitPoints() {
  const attribute = state.renderCache.hitPointPositionAttribute;
  if (!attribute) {
    return;
  }
  const radius = sphereRadiusMeters();
  if (state.renderCache.hitPointRadius === radius) {
    return;
  }

  const positions = attribute.array;
  let index = 0;
  const validFrameIndices = [];
  for (const frame of state.renderCache.hitPointRecords) {
    const hitResult = sphereHitForFrame(frame);
    const hitPoint = hitResult.valid ? surfaceOffsetPoint(hitResult.point) : null;
    if (!hitPoint || !Number.isInteger(frame.frame_index)) {
      continue;
    }
    positions[index] = hitPoint.x;
    positions[index + 1] = hitPoint.y;
    positions[index + 2] = hitPoint.z;
    validFrameIndices.push(frame.frame_index);
    index += 3;
  }
  attribute.needsUpdate = true;
  state.renderCache.hitPointFrameIndices = validFrameIndices;
  state.renderCache.hitPointRadius = radius;
}

function updateAccumulatedHitAreaPositions() {
  const attribute = state.renderCache.hitAreaPositionAttribute;
  if (!attribute) {
    return;
  }
  const radiusScale = hitAreaRadiusScale(angularErrorDegrees()) || 0;
  const radius = sphereRadiusMeters();
  if (
    state.renderCache.hitAreaRadiusScale === radiusScale &&
    state.renderCache.hitAreaSphereRadius === radius
  ) {
    return;
  }

  const positions = attribute.array;
  const validFrameIndices = [];
  let validPatchIndex = 0;
  for (
    let patchIndex = 0;
    patchIndex < state.renderCache.hitAreaPatchBases.length;
    patchIndex += 1
  ) {
    const patch = state.renderCache.hitAreaPatchBases[patchIndex];
    const wrotePatch = writeSphereHitAreaPatchPositions(
      positions,
      validPatchIndex * HIT_AREA_VERTEX_COUNT * 3,
      patch.basis,
      radiusScale,
    );
    if (wrotePatch) {
      validFrameIndices.push(patch.frameIndex);
      validPatchIndex += 1;
    }
  }
  attribute.needsUpdate = true;
  state.renderCache.hitAreaPatchFrameIndices = validFrameIndices;
  state.renderCache.hitAreaRadiusScale = radiusScale;
  state.renderCache.hitAreaSphereRadius = radius;
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

function updateHitAreaPatches() {
  updateAccumulatedHitAreasForAngularError();
  updateAccumulatedVisibility();
}

function buildStaticScene() {
  clearGroup(groups.static);
  state.renderCache.gazeSphere = null;
  buildGazeSphere();

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

function rebuildStaticProjectionSurface() {
  if (!state.sceneData) {
    return;
  }
  if (!state.renderCache.gazeSphere) {
    buildStaticScene();
    return;
  }
  state.renderCache.gazeSphere.position.copy(sphereCenterScene());
  state.renderCache.gazeSphere.scale.setScalar(sphereRadiusMeters());
  applyStaticVisibility();
}

function applyStaticVisibility() {
  for (const child of groups.static.children) {
    const layer = child.userData.layer;
    if (layer === "gazeSphere") {
      child.visible = Boolean(elements.controls.gazeSphere?.checked);
    } else if (layer === "axes") {
      child.visible = elements.toggles.axes.checked;
    }
  }
}

function currentFrame() {
  return state.sceneData?.frames?.[state.frameIndex] || null;
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
  const hitResult = sphereHitForFrame(frame);
  const hitPoint = hitResult.valid ? surfaceOffsetPoint(hitResult.point) : null;
  if (elements.toggles.ray.checked && ray?.valid) {
    const origin = vector(ray.origin_scene_m || ray.scene_m);
    const direction = rayDirectionScene(frame);
    if (origin && direction) {
      if (hitPoint) {
        addLine(groups.current, origin, hitPoint, materials.ray);
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

  if (elements.toggles.hitPoints.checked && hitPoint) {
    addSphere(groups.current, hitPoint, 0.014, materials.currentHit);
  }
}

function rebuildCurrentFrame() {
  renderCurrentFrame();
}

function updateStatusPanel() {
  const frame = currentFrame();
  const total = state.sceneData?.frame_count || 0;
  let validHitsToFrame = 0;
  if (state.sceneData) {
    validHitsToFrame = visibleHitPointCount();
  }
  const totalValidHits =
    state.renderCache.hitPointFrameIndices.length || 0;

  elements.frameLabel.textContent =
    total > 0 ? `${state.frameIndex + 1} / ${total}` : "0 / 0";
  elements.frameIdentity.textContent = frame
    ? `${frame.frame_id} (${frame.frame_index})`
    : "-";
  elements.rayStatus.textContent = frame?.unigaze_ray?.valid
    ? "valid appearance_gaze ray"
    : frame?.unigaze_ray?.reason_invalid || "invalid";
  const hitResult = sphereHitForFrame(frame);
  elements.hitStatus.textContent = hitResult.valid
    ? "valid sphere hit"
    : hitResult.reason || "invalid";
  elements.accumulatedStatus.textContent = `${validHitsToFrame} of ${totalValidHits}`;
  elements.hitCount.textContent = String(totalValidHits);

  const reason = hitResult.valid
    ? "sphere hit is valid"
    : hitResult.reason || "sphere hit is invalid";
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
  rebuildCurrentFrame();
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

function render() {
  requestRender();
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
    rebuildCurrentFrame();
    updateAccumulatedHitAreasForAngularError();
    updateAccumulatedVisibility();
    requestRender();
  });
  elements.hitAreaOpacity.addEventListener("input", () => {
    updateHitAreaOpacityLabel();
    applyHitAreaOpacity();
    requestRender();
  });
  elements.controls.sphereRadius?.addEventListener("input", () => {
    updateSphereRadiusLabel();
    rebuildStaticProjectionSurface();
    rebuildCurrentFrame();
    updateAccumulatedHitPoints();
    updateHitAreaPatches();
    updateStatusPanel();
    render();
  });

  const layerToggles = [
    ...Object.values(elements.toggles),
    elements.controls.gazeSphere,
  ].filter(Boolean);
  for (const toggle of layerToggles) {
    toggle.addEventListener("change", () => {
      applyStaticVisibility();
      rebuildCurrentFrame();
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
  elements.controls.sphereRadius.min = String(SPHERE_MIN_RADIUS_M);
  elements.controls.sphereRadius.max = SPHERE_MAX_RADIUS_M.toFixed(2);
  elements.controls.sphereRadius.step = SPHERE_RADIUS_STEP_M.toFixed(2);
  const persistedRadius = sceneData.gaze_sphere?.radius_m || DEFAULT_SPHERE_RADIUS_M;
  elements.controls.sphereRadius.value = String(
    Math.min(SPHERE_MAX_RADIUS_M, Math.max(SPHERE_MIN_RADIUS_M, persistedRadius)).toFixed(2),
  );
  updateSphereRadiusLabel();
  elements.hitCount.textContent = String(
    sceneData.frames?.filter((frame) => frame?.sphere_hit?.valid).length || 0,
  );
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

controls.addEventListener("change", requestRender);
const resizeObserver =
  typeof ResizeObserver === "function" ? new ResizeObserver(resizeRenderer) : null;
if (resizeObserver) {
  resizeObserver.observe(elements.canvas);
}
window.addEventListener("resize", resizeRenderer);
boot();
