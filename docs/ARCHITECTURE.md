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
