# Copilot instructions for `reality-pulse`

Reality Pulse is a macOS SwiftUI app for queued Apple Object Capture / RealityKit photogrammetry. It processes folders of images into USDZ files with scheduling, SwiftData persistence, history, and interrupted-job recovery.

## Build, test, and lint

- Open the project in Xcode with `open RealityPulse.xcodeproj`.
- Build from the CLI with:

```bash
xcodebuild -project RealityPulse.xcodeproj \
  -scheme RealityPulse \
  -configuration Debug \
  -destination 'platform=macOS' \
  build
```

- Run tests with:

```bash
xcodebuild -project RealityPulse.xcodeproj \
  -scheme RealityPulse \
  -configuration Debug \
  -destination 'platform=macOS' \
  test
```

- The shared scheme is `RealityPulse`.
- There is a focused test target in `ObjectCaptureReconstructionTests`.
- There is no repository lint configuration (`SwiftLint`, `SwiftFormat`, etc.) checked in.
- `Configuration/SampleCode.xcconfig` derives `SAMPLE_CODE_DISAMBIGUATOR` from `DEVELOPMENT_TEAM`, so if builds fail on a fresh machine, set a development team in Xcode first.

## High-level architecture

- `ObjectCaptureReconstruction/ObjectCaptureReconstructionApp.swift` creates the SwiftData `ModelContainer` and the main `Reality Pulse` window.
- `ObjectCaptureReconstruction/ContentView.swift` owns the `AppDataModel`, performs launch recovery, loads persisted state, and routes the UI.
- `ObjectCaptureReconstruction/AppDataModel.swift` is the central application state object. It owns `JobScheduler` plus sheet/editing state for queue workflows.
- `ObjectCaptureReconstruction/Scheduler/JobScheduler.swift` owns queue operations, sequential processing, schedule-window enforcement, pause/cancel semantics, sleep prevention, notifications, and retry preparation.
- `ObjectCaptureReconstruction/Store/JobStore.swift` is the scheduler-facing persistence API backed by SwiftData. It also handles one-time migration from legacy `jobs.json` and `schedule.json`.
- `ObjectCaptureReconstruction/Models/ReconstructionJob.swift` is the app-facing value model for one reconstruction job. It maps requested detail levels to one or more `PhotogrammetrySession.Request.modelFile` requests.
- `ObjectCaptureReconstruction/Models/PersistentJob.swift`, `PersistentScheduleSettings.swift`, and `PersistentMigrationState.swift` are SwiftData storage models.
- `ObjectCaptureReconstruction/Queue/` contains the queue dashboard, job rows, job setup sheet, and schedule settings sheet.
- `ObjectCaptureReconstruction/Settings/` contains folder selection and Object Capture configuration controls.
- `ObjectCaptureReconstruction/Processing/` renders reconstruction progress and previews completed USDZ output.

## Core behavior to preserve

- Queue jobs, completed history, failed jobs, cancelled jobs, interrupted jobs, and schedule settings persist with SwiftData.
- On launch, any job previously stored as `running` must be converted to `interrupted`.
- Restored jobs must not auto-start. The user must press **Start**.
- Completed jobs stay visible as history until removed by the user.
- Pending jobs run sequentially in queue order.
- Schedule windows determine when new jobs start. If an allowed window closes mid-job, the active reconstruction continues because `PhotogrammetrySession` cannot pause in-progress processing.
- Pause takes effect between jobs unless no job is active.
- Retry of failed, cancelled, or interrupted jobs resets status to `pending` and clears the error.
- Retry of an interrupted multi-output job should skip an output only when completion was recorded and the USDZ file still exists. Existing unrecorded destination files should be deleted before retry to avoid `file already exists` failures.

## Key conventions

- Use `@Environment(AppDataModel.self)` for shared SwiftUI state where the existing UI does.
- Keep `ReconstructionJob` as the app-facing value type; add SwiftData fields through `PersistentJob` and explicit mapping helpers.
- Preserve stable queue ordering with `queueOrder`.
- Use `JobStore.saveJob(_:)` for individual job mutations and `JobStore.saveJobs(_:)` when the queue order changes.
- Resolve security-scoped bookmarks before processing input or output folders, and stop accessing them afterward.
- Most files use the same logger pattern:

```swift
private let logger = Logger(
    subsystem: ObjectCaptureReconstructionApp.subsystem,
    category: "<TypeName>"
)
```

- Existing sample-derived Swift files keep the Apple header block with an `Abstract:` section. Preserve that style when editing neighboring files.
- `ImageHelper.validImageSuffixes` is the source of truth for supported input image extensions.

## Testing guidance

Add or update tests for changes to:

- SwiftData schema and model mapping
- launch recovery
- legacy JSON migration
- queue ordering
- schedule persistence
- retry, cancel, failed, completed, or interrupted status behavior
- output request generation and multi-detail retry handling

The test suite uses in-memory SwiftData containers and temporary directories where possible.
