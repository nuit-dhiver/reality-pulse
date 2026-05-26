# Contributing To Reality Pulse

Thanks for helping improve Reality Pulse, a macOS SwiftUI app for queued Apple Object Capture reconstruction. This guide keeps local changes aligned with the current codebase.

## Development Setup

Requirements:

- macOS 14.0 or newer
- Xcode 15.0 or newer
- Apple Object Capture / RealityKit support

Open the project in Xcode:

```bash
open RealityPulse.xcodeproj
```

Build from the command line:

```bash
xcodebuild -project RealityPulse.xcodeproj \
  -scheme RealityPulse \
  -configuration Debug \
  -destination 'platform=macOS' \
  build
```

Run tests:

```bash
xcodebuild -project RealityPulse.xcodeproj \
  -scheme RealityPulse \
  -configuration Debug \
  -destination 'platform=macOS' \
  test
```

`Configuration/SampleCode.xcconfig` derives `SAMPLE_CODE_DISAMBIGUATOR` from `DEVELOPMENT_TEAM`. If signing fails on a fresh machine, set a development team in Xcode.

## Project Map

- `ObjectCaptureReconstruction/ObjectCaptureReconstructionApp.swift`: app entry point, SwiftData container setup, main window.
- `ObjectCaptureReconstruction/ContentView.swift`: top-level SwiftUI routing.
- `ObjectCaptureReconstruction/AppDataModel.swift`: central UI and app state.
- `ObjectCaptureReconstruction/Scheduler/JobScheduler.swift`: queue processing, schedule enforcement, sleep prevention, retry handling, notifications.
- `ObjectCaptureReconstruction/Store/JobStore.swift`: SwiftData persistence, launch recovery, and legacy JSON migration.
- `ObjectCaptureReconstruction/Models/`: Codable job, schedule, and SwiftData model types.
- `ObjectCaptureReconstruction/Queue/`: queue dashboard, job rows, job setup, schedule sheet.
- `ObjectCaptureReconstruction/Settings/`: folder selection and Object Capture configuration controls.
- `ObjectCaptureReconstruction/Processing/`: reconstruction progress and USDZ preview flow.
- `ObjectCaptureReconstructionTests/`: persistence and scheduler behavior tests.

## Coding Conventions

- Keep SwiftUI state flowing through `AppDataModel`, `JobScheduler`, and the existing `@Environment` patterns.
- Preserve the sample-derived file header style in neighboring Swift files.
- Use the existing `Logger` pattern for new diagnostics:

```swift
private let logger = Logger(
    subsystem: ObjectCaptureReconstructionApp.subsystem,
    category: "TypeName"
)
```

- Treat `ReconstructionJob` as the app-facing value model and `PersistentJob` as the SwiftData storage model.
- Keep persistence changes incremental where possible: update individual jobs through `JobStore.saveJob(_:)` and queue ordering through `JobStore.saveJobs(_:)`.
- Do not auto-start queue processing on launch. Restored pending jobs should remain idle until the user presses **Start**.
- Remember that `PhotogrammetrySession` does not support pausing an active reconstruction. Pause and schedule windows take effect between jobs.
- For interrupted retries, only skip an output when completion was recorded and the USDZ file still exists. Existing unrecorded output files should be treated as stale and replaced.

## Testing Guidance

Add or update tests when changing:

- SwiftData schema or mapping
- launch recovery
- queue ordering
- schedule persistence
- retry, cancel, failed, or interrupted status behavior
- multi-output reconstruction request generation
- legacy JSON migration

The current focused suite uses in-memory SwiftData containers and temporary directories so persistence behavior can be tested without relying on a real user data store.

## Pull Request Checklist

- Build passes with `xcodebuild ... build`.
- Tests pass with `xcodebuild ... test`.
- README or contributor docs are updated when behavior changes.
- User-facing recovery or persistence failures are logged and surfaced clearly.
- New queue behavior does not discard completed history unless the user explicitly removes jobs.
