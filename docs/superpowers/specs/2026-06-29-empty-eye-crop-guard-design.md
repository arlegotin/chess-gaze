# Empty Eye Crop Guard Design

Date: 2026-06-29

## Problem

The `nepo_2.mp4` analysis run
`artifacts/output/nepo_2/runs/20260629T164915Z-766007f3` failed after the last
committed frame `25920` with:

```text
SCHEMA_VALIDATION_FAILED: cannot write empty image
```

The run's frame journal is contiguous through frame `25920`; frame `25921` and
later frames in that batch were not committed. The last committed frame already
has the selected face and left eye crossing the right image boundary. Direct
inspection of decoded frames `25921..25927` shows the subject's face moving
against the right edge. Re-running MediaPipe face and eye observation on the
batch reproduces the failure beginning at frame `25924`.

## Root Cause

`eye_observation._crop_bounds()` treats padded eye crop bounds as a clamp but not
as an intersection with the actual image rectangle. When every eye contour
landmark is outside the right or bottom edge, it can return bounds such as:

```text
x_min = 1275
x_max = 1246
```

With crop persistence enabled this produces a NumPy slice with zero width, and
Pillow raises `ValueError: cannot write empty image` while writing PNG. With
crop persistence disabled the same invalid bounds can still reach the strict
`BBox` schema and fail with `x_max must be greater than x_min`.

The detector output itself is plausible: MediaPipe landmarks are model output,
not a proof that every landmark lies inside the decoded frame. NumPy slicing also
does not signal non-overlap; it returns an array with a zero-sized dimension.

## Decision

Eye crop bounds are an image-intersection contract:

- padded eye bbox partially overlapping the image remains valid;
- the crop bbox and transform use the positive in-frame intersection;
- an empty intersection marks that eye missing with the eye-specific
  `*_EYE_NOT_FOUND` reason;
- original finite landmarks, normalized landmarks, eye bbox, normalized bbox,
  and open metric are retained on the missing-eye observation;
- no crop file is written for an empty intersection, even when
  `save_crop_images=True`.

Image writers also reject zero-width or zero-height arrays with a local,
explicit error before entering Pillow/OpenCV. This is defense-in-depth for other
artifact paths, not the primary analysis-quality fix.

## External Evidence Checked

- Pillow `Image.fromarray()` creates images from array-interface objects and
  `Image.save()` delegates to the format writer. Local Pillow 12.2.0 raises
  `ValueError: cannot write empty image` for zero-sized PNG writes.
- NumPy basic slicing permits ranges whose result has shape `(height, 0, 3)` or
  `(0, width, 3)`; it does not raise for empty overlap.
- MediaPipe Face Landmarker returns face landmarks for detected faces, but the
  local `NormalizedLandmark` contract says coordinates "should" be in range, not
  that downstream code may skip intersection validation.
- PyTorch MPS environment variables affect backend execution. The reproduced
  failure occurs in eye-crop artifact geometry and Pillow/Pydantic validation
  before MPS-specific gaze inference is relevant.

## Acceptance Criteria

- Focused tests fail before the implementation and pass after it:
  - fully off-right eye with `save_crop_images=True` does not raise, writes no
    crop for that eye, and records the eye as missing;
  - fully off-right eye with `save_crop_images=False` also records the eye as
    missing instead of failing schema validation;
  - partially clipped eye remains present with a positive crop bbox clipped to
    the frame;
  - `save_rgb_png()` and `save_bgr_jpeg()` reject empty images explicitly.
- Real `nepo_2.mp4` failing window frames `25924..25927` can be observed without
  an empty-image or invalid-bbox exception.
- A real CLI analysis of `artifacts/input/nakamura_short.mp4` completes with
  default crop retention disabled and no crop files.
