# ADR-0003: Load Three Viewer Modules From Pinned CDN URLs

Date: 2026-06-26

## Status

Accepted

## Context

The first 3D scene viewer implementation vendored Three.js `0.185.0` files into
`src/chess_gaze/viewer_assets/vendor/` and copied them into every generated
`viewer/` directory. That made the viewer fully offline, but it also committed
large third-party JavaScript files and duplicated them into every run artifact.

The current task asks whether those Three.js scripts are really necessary in
the source tree and whether the page can connect to remote modules when it
renders. Browser ES modules can load pinned HTTPS modules if the generated page
provides a stable import map. This keeps the Python package small while
preserving the current no-frontend-build viewer architecture.

Verified on 2026-06-26:

- `npm view three@0.185.0 version license repository.url dist.tarball dist.integrity --json`
  returned version `0.185.0`, MIT, repository
  `git+https://github.com/mrdoob/three.js.git`, tarball
  `https://registry.npmjs.org/three/-/three-0.185.0.tgz`, and integrity
  `sha512-+yRrcRO2iZa8uzvNNl0d7cL4huhgKgBvVJ0njcTe8xFqZ6DMAFZdCKDP91SEAuj25bNAj7k1QQdf+srZywVK6w==`.
- The official Three.js `r185` `package.json` declares module export
  `./build/three.module.js` and addon export `./addons/*` to
  `./examples/jsm/*`.
- The official Three.js `r185` `examples/jsm/controls/OrbitControls.js`
  imports from bare specifier `three`, so consumers need an import map or
  bundler.
- jsDelivr served the pinned files with `HTTP 200`, `access-control-allow-origin: *`,
  `cache-control: public, max-age=31536000, s-maxage=31536000, immutable`, and
  `x-jsd-version: 0.185.0`:
  - `https://cdn.jsdelivr.net/npm/three@0.185.0/build/three.module.js`
  - `https://cdn.jsdelivr.net/npm/three@0.185.0/build/three.core.js`
  - `https://cdn.jsdelivr.net/npm/three@0.185.0/examples/jsm/controls/OrbitControls.js`

## Alternatives and Evidence

| Alternative | Evidence | Decision |
| --- | --- | --- |
| Keep vendored Three.js files | Works offline and was already tested, but keeps large third-party JS in `src/` and copies it into every run. | Rejected for current requirement. |
| Load floating CDN URLs such as `latest` | Smaller source tree, but non-reproducible and can change viewer behavior without a repository change. | Rejected. |
| Load pinned jsDelivr npm URLs with an import map | npm and official release metadata pin the package to `0.185.0`; jsDelivr serves the exact version with CORS and immutable cache headers; import map resolves OrbitControls' bare `three` import. | Selected. |
| Add a frontend build/bundler | Would allow local bundling and subresource tooling, but adds Node project maintenance that this repo has intentionally avoided. | Rejected until viewer complexity justifies it. |

## Decision

Generated viewers must not copy or embed local Three.js source files. They must
load Three.js `0.185.0` from pinned jsDelivr npm URLs at page render time using
an import map:

```json
{
  "imports": {
    "three": "https://cdn.jsdelivr.net/npm/three@0.185.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.185.0/examples/jsm/"
  }
}
```

The app's own `scene_viewer.js`, `styles.css`, and generated scene data remain
Python-packaged local viewer assets. `viewer/index.html` continues embedding the
scene data and app source so direct file opening still avoids local JSON/module
fetches.

This decision makes jsDelivr and the npm `three@0.185.0` package a trusted
runtime dependency. Pinned URLs reduce accidental drift, but the browser does
not enforce npm integrity metadata for ES module imports. Remote Three.js
modules execute in the same page as embedded scene data and could read or
transmit that data if the CDN, package, or browser cache were compromised. The
claim that scene JSON, frames, crops, and model data are not uploaded by the
viewer is therefore conditional on trusting the pinned remote module provider.
Users who need offline or stronger supply-chain isolation should not use this
remote-loading mode without adding a local cache/offline mode or equivalent
browser-enforced integrity control.

The viewer is no longer fully offline: first render requires network access to
jsDelivr unless the browser cache already contains the pinned modules.

## Consequences

- `src/chess_gaze/viewer_assets/vendor/` is removed.
- Generated `viewer/` directories are smaller and no longer duplicate Three.js
  source files.
- Runtime privacy posture changes from no remote viewer requests to exactly the
  pinned jsDelivr module requests. Scene JSON, frames, crops, and model data are
  not intentionally uploaded by project code, but remote module code is trusted
  because it executes in the same page as embedded scene data.
- Direct `file://` opening depends on browser support for import maps and remote
  ESM modules.
- If CDN availability becomes unacceptable, the next change should add an
  explicit offline mode or local cache instead of silently reintroducing
  vendored scripts.

## Verification

Future agents should verify:

- packaged viewer assets do not include `viewer_assets/vendor/`;
- generated viewers remove stale `viewer/vendor/` directories;
- generated HTML contains only the pinned jsDelivr Three.js import-map URLs as
  external URLs;
- browser network requests may include the transitive pinned
  `three.core.js` module imported by `three.module.js`;
- generated HTML does not embed `three-core-source`, `three-module-source`, or
  `orbit-controls-source`;
- browser smoke shows network requests to the pinned jsDelivr module URLs, no
  `/vendor/...` requests, no console errors, and a nonblank canvas.
