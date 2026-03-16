# Copilot instructions for `reality-pulse`

## Build, test, and lint

- Open the project in Xcode with `open ObjectCaptureReconstruction.xcodeproj`.
- Build from the CLI with:

```bash
xcodebuild -project ObjectCaptureReconstruction.xcodeproj \
  -scheme ObjectCaptureReconstruction \
  -configuration Debug \
  -destination 'platform=macOS' \
  build
```

- The project has a shared scheme named `ObjectCaptureReconstruction` and a single app target with `Debug` and `Release` configurations.
- There is no test target in `ObjectCaptureReconstruction.xcodeproj`, so there is no full-suite or single-test command to run yet.
- There is no repository lint configuration (`SwiftLint`, `SwiftFormat`, etc.) checked in.
- `Configuration/SampleCode.xcconfig` derives `SAMPLE_CODE_DISAMBIGUATOR` from `DEVELOPMENT_TEAM`, so if builds fail on a fresh machine, set a development team in Xcode first.

## High-level architecture

- This is a macOS SwiftUI app around Apple Object Capture / `PhotogrammetrySession`.
- `ObjectCaptureReconstruction/ObjectCaptureReconstructionApp.swift` creates a single fixed-size window and shows `ContentView`.
- `ObjectCaptureReconstruction/ContentView.swift` owns the only `AppDataModel` instance and injects it with `.environment(appDataModel)`.
- `ObjectCaptureReconstruction/AppDataModel.swift` is the central state store. It keeps the selected folders, model name, `PhotogrammetrySession.Configuration`, current `PhotogrammetrySession`, alert text, and the app state enum (`ready`, `reconstructing`, `viewing`, `error`).
- `ContentView` is effectively a state router:
  - `.ready` -> `SettingsView`
  - `.reconstructing` and `.viewing` -> `ProcessingView`
  - `.error` -> empty content plus the shared alert on `ContentView`
- `AppDataModel.startReconstruction()` is the main control point. It validates `imageFolder`, `modelFolder`, and `modelName`, builds one `PhotogrammetrySession.Request.modelFile` per selected detail level, creates the session, then starts processing.
- The `Settings/` tree writes directly into `AppDataModel` and `sessionConfiguration`. The important split is:
  - `Settings/FolderOptions/` for image folder, output folder, and model name
  - `Settings/ReconstructionOptions/` for mesh/detail/masking/crop
  - `Settings/AdvancedReconstructionOptions/` for custom detail configuration and multi-detail processing
- `Settings/FolderOptions/ImageFolderView.swift` triggers the image scan. It uses `ImageHelper` to collect valid image URLs and asynchronously inspect sample metadata. That metadata controls whether crop-related UI is enabled.
- `ImageHelper.loadMetadataAvailability` uses a bounded task group (up to 5 concurrent tasks) and short-circuits once enough images expose depth and gravity metadata.
- The `Processing/` tree renders progress and output:
  - `ReconstructionProgressView` consumes `session.outputs`
  - it updates progress / ETA
  - it switches the app to `.viewing` as soon as the first model finishes
  - it sets `processingComplete` only after `.processingComplete`
  - `ModelView` displays the first generated USDZ with `RealityView`
  - `TopBarView` exposes “Open in Finder” for the selected output folder
  - `BottomBarView` swaps to `ReprocessView` only when all requests are done

## Key conventions

- Use `AppDataModel` as the single shared mutable store. This repo does not use a separate view-model layer.
- Shared state is passed with `@Environment(AppDataModel.self)`. Use `@Binding` only for local parent/child synchronization inside the processing flow.
- Keep the state-machine flow intact: user-facing errors are surfaced by setting `appDataModel.alertMessage` and moving `appDataModel.state` to `.error`, which `ContentView` turns into the alert.
- When adding a new reconstruction option, wire it all the way through `AppDataModel` and `PhotogrammetrySession.Configuration` rather than storing parallel local state in a leaf view.
- `QualityView` has an important cross-file behavior: when the selected detail level changes away from `.custom`, it resets `sessionConfiguration.customDetailSpecification`. Preserve that unless you also change the advanced-options flow.
- Crop / bounding-box behavior depends on metadata gathered in `ImageFolderView` + `ImageHelper`; `IgnoreBoundingBoxView` is intentionally disabled until that scan marks `appDataModel.boundingBoxAvailable = true`.
- Multiple output models are produced by combining the main quality picker with the advanced multi-detail toggles in `AppDataModel.createReconstructionRequests()`. Changes to detail-level handling should be verified there, not only in the UI.
- Most files follow the same logger pattern:

```swift
private let logger = Logger(
    subsystem: ObjectCaptureReconstructionApp.subsystem,
    category: "<TypeName>"
)
```

- Existing sample-derived Swift files also keep the Apple header block with an `Abstract:` section. Preserve that style when editing neighboring files.
- `ImageHelper.validImageSuffixes` is the source of truth for supported input image extensions; image-folder and thumbnail behavior both rely on it.

## Repository context from existing docs/config

- `README.md` only records the licensing context: the repository is GPLv3 and includes Apple components released under MIT.
- `ObjectCaptureReconstruction/ObjectCaptureReconstruction.entitlements` grants sandboxed, user-selected read/write file access. Be careful not to break the folder-picking flow when changing file access code.
