# Reality Pulse Architecture

Reality Pulse is a native SwiftUI app built around Apple Object Capture's `PhotogrammetrySession`. The app's main responsibility is to turn single-session reconstruction into a persistent, scheduled, retryable queue.

One target and one scheme build the app for macOS, iOS, and iPadOS. The queue, scheduler, persistence, and export code are shared; the platform differences live in `ReconstructionCapability` and `Platform/`.

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

`ObjectCaptureReconstructionApp` owns the app-level SwiftData container and creates the app's scene: a single `Reality Pulse` window with a minimum size on macOS, and a window group on iPhone and iPad, where `ContentView` also wraps the dashboard in a navigation stack.

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

### Platform Capabilities

`ReconstructionCapability` answers what Object Capture supports where, so feature code asks it instead of testing the platform:

- `supportedDetailLevels`: every level on macOS, and `.reduced` alone on iOS and iPadOS, which is all Object Capture exposes there
- `defaultDetailLevel`: the level a new job starts with
- `supportsMultipleDetailLevels`, `supportsCustomDetailSpecification`, `supportsMeshPrimitiveSelection`: which settings the job-setup UI shows
- `isSupportedOnThisDevice`: `PhotogrammetrySession.isSupported`, which gates the queue's **Start** button and `JobScheduler.start()`
- the messages shown for an unsupported device and for a job that asks for a detail level this platform cannot produce

`CodableDetailLevel` keeps every case on every platform so stored jobs decode identically, and maps to the framework type through the optional `frameworkDetail`, which is `nil` for a level the platform cannot request. `CodableSessionConfiguration` does the same for settings: it stores mesh primitive and custom detail specification everywhere, and only converts them to framework values on macOS.

`Platform/` holds the rest of the differences:

- `FolderBookmark`: security-scoped bookmark options, which iOS bookmarks do not need
- `SecurityScopedAccess`: access that lives as long as the object, used while the share sheet reads models
- `OutputReveal`: reveal in the Finder on macOS; the mobile builds share instead
- `QueueAwakeAssertion`: a `ProcessInfo` activity on macOS, and the idle timer on iPhone and iPad
- `PlatformUI`: folder and file icons, the share sheet, and the toggle, list, and sheet-sizing shims

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
- additional model outputs (macOS)
- mesh primitive (macOS)
- masking
- bounding-box behavior
- custom polygon count (macOS)
- texture maps (macOS)
- texture format (macOS)
- texture dimension (macOS)

The macOS-only controls guard their whole file with `#if os(macOS)`, and `ReconstructionOptionsView` only references them from a guarded call site. On iPhone and iPad the quality row shows the single supported level instead of a picker.

### Processing UI

The `Processing/` views display progress, estimated time remaining, completion state, and USDZ preview with RealityKit.

## Data And File Access

The app is sandboxed. User-selected input and output folders are stored with security-scoped bookmarks so the queue can access them after relaunch. macOS creates and resolves those bookmarks with an explicit security scope; bookmarks on iPhone and iPad are implicitly scoped, so `FolderBookmark` supplies the right options for each.

The mobile builds declare `UIFileSharingEnabled` and `LSSupportsOpeningDocumentsInPlace`, so capture folders can be copied into the app's documents folder with the Files app and finished models copied back out.

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
- platform detail-level support, the requests a job creates from it, and stored settings round-tripping unchanged on every platform
