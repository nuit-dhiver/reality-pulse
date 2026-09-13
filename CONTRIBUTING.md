# Contributing To Reality Pulse

Thanks for helping improve Reality Pulse, a SwiftUI app for queued Apple Object Capture reconstruction on Mac, iPhone, and iPad. This guide keeps local changes aligned with the current codebase.

## Development Setup

Requirements:

- a Mac running macOS 15.0 or newer to build, and macOS 15.0 or iOS / iPadOS 18.0 to run
- Xcode 16.0 or newer
- Apple Object Capture / RealityKit support on whatever device you reconstruct with

Open the project in Xcode:

```bash
open RealityPulse.xcodeproj
```

The single `RealityPulse` scheme and `ObjectCaptureReconstruction` target build every destination; the `-destination` flag picks the platform.

Build from the command line:

```bash
# macOS
xcodebuild -project RealityPulse.xcodeproj \
  -scheme RealityPulse \
  -configuration Debug \
  -destination 'platform=macOS' \
  build

# iPhone simulator
xcodebuild -project RealityPulse.xcodeproj \
  -scheme RealityPulse \
  -configuration Debug \
  -destination 'platform=iOS Simulator,name=iPhone 16' \
  build

# iPad simulator
xcodebuild -project RealityPulse.xcodeproj \
  -scheme RealityPulse \
  -configuration Debug \
  -destination 'platform=iOS Simulator,name=iPad (10th generation)' \
  build
```

Run tests:

```bash
xcodebuild -project RealityPulse.xcodeproj \
  -scheme RealityPulse \
  -configuration Debug \
  -destination 'platform=macOS' \
  test

xcodebuild -project RealityPulse.xcodeproj \
  -scheme RealityPulse \
  -configuration Debug \
  -destination 'platform=iOS Simulator,name=iPhone 16' \
  test
```

`.github/workflows/swift.yml` runs the same commands: a macOS build, plus iPhone and iPad simulator builds and a test run. It resolves simulator UDIDs with `.github/scripts/pick-simulator.py` instead of hard-coding device names.

Create a release by pushing a version tag:

```bash
git tag v1.1.0
git push origin v1.1.0
```

The release workflow runs tests, builds the Release app, creates a zip and checksum, and uploads them to the GitHub Release.

`Configuration/SampleCode.xcconfig` derives `SAMPLE_CODE_DISAMBIGUATOR` from `DEVELOPMENT_TEAM`. If signing fails on a fresh machine, set a development team in Xcode.

## Project Map

- `ObjectCaptureReconstruction/ObjectCaptureReconstructionApp.swift`: app entry point, SwiftData container setup, main window.
- `ObjectCaptureReconstruction/ContentView.swift`: top-level SwiftUI routing.
- `ObjectCaptureReconstruction/AppDataModel.swift`: central UI and app state.
- `ObjectCaptureReconstruction/Scheduler/JobScheduler.swift`: queue processing, schedule enforcement, sleep prevention, retry handling, notifications.
- `ObjectCaptureReconstruction/Store/JobStore.swift`: SwiftData persistence, launch recovery, and legacy JSON migration.
- `ObjectCaptureReconstruction/Models/`: Codable job, schedule, and SwiftData model types, plus `ReconstructionCapability` for what Object Capture supports on the current platform.
- `ObjectCaptureReconstruction/Platform/`: folder bookmarks and security-scoped access, revealing or sharing finished models, keeping the system awake, and the SwiftUI shims that differ between macOS and the mobile builds.
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

## Platform Conventions

The app ships from one target for macOS, iOS, and iPadOS, so platform differences belong in a few known places:

- Ask `ReconstructionCapability` what Object Capture supports instead of testing the platform in feature code. Object Capture on iOS and iPadOS offers only the reduced detail level, and has no mesh primitive or custom detail specification.
- Keep `#if os(macOS)` around framework conversions and platform-only views, not around stored data. `CodableSessionConfiguration` and `CodableDetailLevel` keep every field and case on all platforms so a saved job decodes the same way everywhere.
- Put AppKit and UIKit differences behind the helpers in `Platform/`: `PlatformFolderIcon`, `PlatformFileIcon`, the `platform…` view modifiers, `FolderBookmark`, `SecurityScopedAccess`, `OutputReveal`, and `QueueAwakeAssertion`.
- A view that only exists on macOS guards its whole file with `#if os(macOS)` and is referenced from a guarded call site.
- New Object Capture API needs an availability check before use. Several `PhotogrammetrySession` members, including `Request.Detail.medium` and `Configuration.meshPrimitive`, do not exist in the iOS SDK at all, so they fail to compile rather than fail at runtime.

## Testing Guidance

Add or update tests when changing:

- SwiftData schema or mapping
- launch recovery
- queue ordering
- schedule persistence
- retry, cancel, failed, or interrupted status behavior
- multi-output reconstruction request generation
- legacy JSON migration
- platform capability differences, such as which detail levels a build can request

The current focused suite uses in-memory SwiftData containers and temporary directories so persistence behavior can be tested without relying on a real user data store.

## Pull Request Checklist

- Build passes for macOS and for an iPhone or iPad simulator with `xcodebuild ... build`.
- Tests pass with `xcodebuild ... test` on macOS and on an iOS simulator.
- README or contributor docs are updated when behavior changes.
- User-facing recovery or persistence failures are logged and surfaced clearly.
- New queue behavior does not discard completed history unless the user explicitly removes jobs.
