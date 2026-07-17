# Reality Pulse Architecture

Reality Pulse is a native macOS SwiftUI app built around Apple Object Capture's `PhotogrammetrySession`. The app's main responsibility is to turn single-session reconstruction into a persistent, scheduled, retryable queue.

## Runtime Flow

```text
ObjectCaptureReconstructionApp
  -> ContentView
    -> AppDataModel
      -> JobScheduler
        -> PhotogrammetrySession
      -> JobStore
        -> SwiftData
```

1. `ObjectCaptureReconstructionApp` creates the SwiftData model container.
2. `ContentView` creates `AppDataModel`, runs launch recovery, and loads persisted jobs.
3. The queue UI edits `ReconstructionJob` values through `JobScheduler`.
4. `JobScheduler` selects the next pending job, resolves folder bookmarks, and creates a `PhotogrammetrySession`.
5. Session outputs update progress, completion metadata, job status, and user notifications.
6. `JobStore` persists every meaningful queue and schedule mutation through SwiftData.

## Main Components

### App Entry

`ObjectCaptureReconstructionApp` owns the app-level SwiftData container and creates a single `Reality Pulse` window.

### App State

`AppDataModel` holds shared UI state:

- the `JobScheduler`
- job setup sheet state
- schedule sheet state
- editing state
- persistence alert state

SwiftUI views receive it through `@Environment(AppDataModel.self)`.

### Queue And Scheduler

`JobScheduler` owns the in-memory job array and scheduler state. It handles:

- add, remove, edit, retry, and reorder operations
- pending-job selection
- delay and allowed-hours scheduling
- pause-between-jobs behavior
- cancellation
- sleep prevention
- notification dispatch
- per-output retry preparation

The scheduler does not auto-start restored work after launch. The user must press **Start**.

### Persistence

`JobStore` is the persistence boundary used by the scheduler. It stores:

- `PersistentJob`
- `PersistentScheduleSettings`
- `PersistentMigrationState`

`ReconstructionJob` remains the app-facing model. `PersistentJob` maps it to SwiftData fields, including encoded folders, bookmarks, Object Capture settings, queue order, status, progress, errors, timestamps, and completed output filenames.

Launch recovery:

- migrates legacy JSON files once if present
- marks previously running jobs as `interrupted`
- preserves completed, failed, cancelled, interrupted, and pending history

### Reconstruction Jobs

`ReconstructionJob` describes one input image folder and one output model name. A job can request multiple detail levels. Each level produces one `PhotogrammetrySession.Request.modelFile` and one USDZ file:

```text
<model-name>-<detail-level>.usdz
```

Completed output filenames are recorded when `PhotogrammetrySession.Output.requestComplete` is received.

### Interrupted Retry Handling

Retry handling is deliberately conservative:

- if an output was recorded complete and the USDZ still exists, skip it
- if a destination USDZ exists without a recorded completion, delete it before retry
- if every requested output is already complete, mark the job complete without starting a new session

This avoids `file already exists` failures while preventing partial or ambiguous files from being treated as finished models.

### Settings UI

The `Settings/` views configure `PhotogrammetrySession.Configuration`, including:

- detail level
- additional model outputs
- mesh primitive
- masking
- bounding-box behavior
- custom polygon count
- texture maps
- texture format
- texture dimension

### Processing UI

The `Processing/` views display progress, estimated time remaining, completion state, and USDZ preview with RealityKit.

## Data And File Access

The app is sandboxed. User-selected input and output folders are stored with security-scoped bookmarks so the queue can access them after relaunch.

## Provenance Watermarking

Jobs can opt in (per-job toggle, default off) to embedding an imperceptible,
per-copy provenance watermark in every exported file, so a copy found in the
wild can be traced back to the export that produced it. The implementation
lives in the local Swift package `RealityMarkKit/` (`WatermarkCore` +
`ModelFileIO`), shared by the app and the internal `watermark-verify` CLI so
embedding and detection can never drift apart.

### Open design

This repository is public, so the scheme follows Kerckhoffs's principle: the
algorithm hides nothing, and all security rests on a per-copy 256-bit secret
key. Every keyed decision (bit sequence, bin permutation, chip signs) is
derived from that key via HMAC-SHA256; without the key the mark can be
neither read, forged, nor selectively targeted. Wrong keys detect at exactly
chance level.

### Channels

- **Geometry** — a blind keyed statistical watermark on the distribution of
  vertex radial norms (Cho–Prost–Jung, IEEE TSP 2007, hardened with
  trimmed-quantile range normalization). Norms are computed against the
  centroid and normalized over a trimmed range, so detection is invariant to
  vertex reordering, translation, rotation, and uniform scale, and needs only
  the position multiset — no topology. One bit per norm bin; vertices move at
  most one bin width (≲0.5% of the bounding-box diagonal). Applied to
  glTF/GLB vertices and Gaussian-splat PLY points.
- **Texture** — a blind additive spread-spectrum mark in the mid-band 8×8 DCT
  coefficients of the base-color luma plane (PSNR ≥ 45 dB). Survives PNG/JPEG
  re-encoding and mild rescaling (the detector resamples suspects back to the
  recorded size). Applied to glTF/GLB base-color images and, via a
  stored-zip repack that never touches the `.usdc` geometry bytes, to the
  USDZ itself.

Coverage per format: USDZ = texture only (v1), PLY = geometry only,
glTF/GLB = both. The scheduler finalizes a job by exporting derived formats
first (from the still-pristine USDZ) and stamping the USDZ last, so every
distributed file maps to exactly one record and one fresh key.

### Records and verification

Each stamped file gets a `PersistentExportRecord` row (SwiftData): per-copy
key, channels, embedding parameters, and the file's SHA-256. Records are
append-only provenance history and survive job deletion. An internal record
export ("Export Provenance Records…" in the job context menu, hidden behind
`defaults write <bundle-id> RPWatermarkRecordExport -bool YES`) writes them as
`*.wmrecord.json` for the offline verifier:

```
swift build --package-path RealityMarkKit -c release
RealityMarkKit/.build/release/watermark-verify --record <record.json> <suspect-file>
```

The CLI loads geometry/images from usdz, glb, gltf, ply, png, or jpg
suspects, reports per-channel match statistics with binomial/normal-tail
p-values, and prints a MATCH / LIKELY / NO MATCH verdict (exit codes 0/1/2).
Record JSONs contain the secret keys — treat them as credentials.

### Known limitations

- Aggressive remeshing plus full texture replacement strips both channels;
  heavy decimation (≫50%), non-uniform scaling, or shearing breaks the
  geometry statistic; large crops/warps or AI re-texturing break the texture
  channel.
- The algorithm being public means an attacker can run targeted
  distribution-flattening or mid-band-scrubbing attacks — at a measurable
  quality cost. The mark traces and deters; it is not DRM.
- Cropping or part-extraction shifts the centroid, degrading partial-mesh
  geometry detection; texture detection assumes approximate origin alignment.
- Keys live only in the app's SwiftData store and exported record JSONs;
  losing the store means losing verifiability, and anyone with the Mac
  account (or a record file) can remove or transplant marks.
- Manual re-exports from an already-stamped USDZ get texture-double-marked
  (benign: independent-key patterns are quasi-orthogonal and each record
  still verifies independently).

## Test Coverage

The focused test target verifies:

- stable queue ordering
- status history persistence
- schedule persistence
- completed output metadata persistence
- launch recovery from running to interrupted
- interrupted retry reset behavior
- completed-output preservation on retry
- stale output deletion before retry
- one-time legacy JSON migration
- watermark embed → export → read-back → detect round trips (GLB, PLY, USDZ)
- provenance-record persistence, idempotence lookups, and job-deletion survival

`RealityMarkKit/` has its own test suite (`swift test --package-path
RealityMarkKit`) covering keyed-PRF determinism, geometry roundtrip and
robustness (noise, similarity transforms, reordering, subsampling), false
positives at chance level, imperceptibility bounds, texture PSNR and
JPEG/rescale robustness, and usdz archive alignment.
