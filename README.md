# Reality Pulse

[![Build macOS and mobile apps](https://github.com/nuit-dhiver/reality-pulse/actions/workflows/swift.yml/badge.svg)](https://github.com/nuit-dhiver/reality-pulse/actions/workflows/swift.yml)
[![Swift](https://img.shields.io/badge/Swift-5-orange.svg)](https://www.swift.org/)
[![macOS](https://img.shields.io/badge/macOS-15%2B-blue.svg)](https://developer.apple.com/macos/)
[![iOS](https://img.shields.io/badge/iOS%20%7C%20iPadOS-18%2B-blue.svg)](https://developer.apple.com/ios/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

**Reality Pulse is a photogrammetry queue for Apple Object Capture on Mac, iPhone, and iPad.** It turns folders of photos into USDZ 3D models with a SwiftUI batch interface, RealityKit reconstruction settings, scheduled processing windows, SwiftData persistence, and crash-aware retry recovery.

Use it when you have many capture sets to reconstruct and want a local Apple Object Capture workflow that can run overnight, survive relaunches, keep completed history, and export multiple quality levels from one job. The same app builds for macOS and for iPhone and iPad, where reconstruction runs on device with the detail level Object Capture supports there.

Project site: [nuit-dhiver.github.io/reality-pulse](https://nuit-dhiver.github.io/reality-pulse/)

## Why Reality Pulse?

Apple Object Capture is powerful, but running one folder at a time is tedious when you are scanning products, props, archive objects, handmade pieces, or environment assets. Reality Pulse wraps `PhotogrammetrySession` in a persistent queue so you can prepare a batch, choose output quality, and let the device process the work in order — on a Mac overnight, or on an iPhone or iPad in the field right after a capture session.

## Features

- **Mac, iPhone, and iPad builds**: one app target and one scheme, building for macOS and for iOS and iPadOS, with on-device reconstruction on each.
- **Batch Apple Object Capture queue**: add multiple image folders and process them sequentially.
- **USDZ photogrammetry output**: export one or more `.usdz` models per job.
- **Multiple detail levels**: generate preview, reduced, medium, full, raw, or custom outputs from the same capture on macOS, and reduced-detail models on iPhone and iPad.
- **Custom reconstruction settings**: configure masking and bounding-box handling everywhere, plus mesh primitive, polygon limits, texture maps, texture format, and texture resolution on macOS.
- **Scheduled processing**: delay a queue run or restrict processing to allowed hours, including overnight windows.
- **Sleep prevention**: optionally keep the Mac awake, or the iPhone or iPad screen awake, while the queue is active.
- **SwiftData persistence**: queue jobs, history, schedule settings, and recovery metadata survive app quits and crashes.
- **Interrupted-job recovery**: running jobs are restored as interrupted on launch and can be retried.
- **Retry-safe exports**: completed outputs are skipped on retry, while stale partial files are replaced before reconstruction resumes.
- **Sandbox-friendly folder access**: user-selected input and output folders are restored with security-scoped bookmarks.
- **Progress and notifications**: track job progress, estimated time remaining, failures, and queue completion.

## Screens And Workflow

1. Add a job from the queue dashboard.
2. Pick an image folder and an output folder.
3. Name the model and choose Object Capture settings.
4. Optionally enable additional detail-level exports.
5. Add more jobs, reorder them, and configure a schedule.
6. Press **Start** when you want the queue to run.

Reality Pulse restores the queue on launch, but it does not automatically start processing after a relaunch. The user stays in control.

## Requirements

- macOS 15.0 or newer, or iOS / iPadOS 18.0 or newer
- Xcode 16.0 or newer
- A device supported by Apple Object Capture / RealityKit photogrammetry
- Photo sets suitable for `PhotogrammetrySession`

The app checks `PhotogrammetrySession.isSupported` at runtime and disables **Start** with an explanation on hardware that cannot reconstruct.

## Build

Clone the repo and build for the platform you want. One target, `RealityPulse`, builds all three destinations:

```bash
git clone https://github.com/nuit-dhiver/reality-pulse.git
cd reality-pulse

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

Reconstruction itself needs real hardware. Build and run on a connected iPhone or iPad with `-destination 'generic/platform=iOS'` and a development team set, or run the app from Xcode with the device selected.

You can also open `RealityPulse.xcodeproj` in Xcode, pick a destination, and build the `RealityPulse` scheme.

## Test

Run the persistence and scheduler-focused test suite with:

```bash
# macOS
xcodebuild -project RealityPulse.xcodeproj \
  -scheme RealityPulse \
  -configuration Debug \
  -destination 'platform=macOS' \
  test

# iPhone simulator
xcodebuild -project RealityPulse.xcodeproj \
  -scheme RealityPulse \
  -configuration Debug \
  -destination 'platform=iOS Simulator,name=iPhone 16' \
  test
```

The current tests cover SwiftData persistence, schedule reloads, launch recovery, interrupted jobs, retry behavior, one-time legacy JSON migration, and the platform reconstruction capabilities. The glTF and Gaussian splat tests build their sample model with Model I/O USD export and skip themselves where that is unavailable.

## Release

Releases are driven by Git tags. To publish a new GitHub Release:

```bash
git tag v1.1.0
git push origin v1.1.0
```

The release workflow builds `Reality Pulse.app` in Release configuration, runs the test suite, packages the app as a zip, writes a SHA-256 checksum, and attaches both files to the GitHub Release.

The current release artifact is ad-hoc signed for local/open-source distribution, not notarized. Releases publish the macOS app only; iPhone and iPad builds are produced from Xcode with your own signing identity.

## Usage Notes

### Input Images

The app is designed for folders of still images supported by Apple Object Capture. Image metadata is inspected where available so the UI can enable bounding-box related options only when the capture set supports them.

### Output Files

Each requested detail level writes a separate USDZ file named:

```text
<model-name>-<detail-level>.usdz
```

For example, a model named `Vase` with medium and raw outputs produces:

```text
Vase-medium.usdz
Vase-raw.usdz
```

### On iPhone And iPad

The mobile builds run the same queue, scheduler, and persistence code as the Mac build. Object Capture itself is smaller there, so the app adapts:

- **Detail level**: Object Capture on iOS and iPadOS only offers reduced detail, so that is the single quality option. Mesh primitive, additional detail levels, and the custom polygon and texture controls are macOS-only and are left out of the mobile UI.
- **Queued jobs from a Mac**: a job that asks for a detail level this device cannot produce fails with a message naming the levels it supports rather than writing a partial model.
- **Foreground processing**: reconstruction only advances while the app is in the foreground, so the sleep-prevention setting keeps the device awake instead of letting the screen sleep. Leaving the app suspends the active job, which the next launch restores as interrupted so it can be retried.
- **Files**: pick input and output folders with the system file picker. The app's own documents folder is visible in the Files app, so capture folders can be dropped in and finished models copied out.
- **Sharing**: the queue's context menu shares finished models and exported files through the system share sheet, in place of the Mac's **Show in Finder**.
- **Reordering**: tap **Edit** in the navigation bar to drag jobs into a different queue order. Removing, editing, retrying, and exporting a job stay on the row's context menu, which opens with a long press.

### Scheduling

Scheduling controls are intentionally conservative:

- delayed start waits until a selected date and time
- allowed hours constrain when new jobs begin
- if a processing window closes during an active reconstruction, the current job continues because `PhotogrammetrySession` does not support pausing an in-progress request
- pause takes effect between jobs

### Persistence And Recovery

Reality Pulse stores jobs and schedule settings with SwiftData. On launch it:

- loads the persisted queue and history
- restores the saved schedule
- converts any previously running job to `interrupted`
- keeps completed, failed, cancelled, and interrupted jobs visible until the user removes them
- leaves pending and scheduled jobs idle until the user presses **Start**

When retrying an interrupted multi-output job, previously completed outputs are skipped only if the app recorded their completion and the file still exists. Existing files without a recorded completion are treated as stale partial outputs and replaced before retrying.

## Architecture

- `ObjectCaptureReconstructionApp`: creates the SwiftData model container and the main SwiftUI scene, a window on macOS and a window group on iPhone and iPad.
- `ContentView`: owns the app model and routes between the queue dashboard and processing views.
- `AppDataModel`: central UI/application state.
- `JobScheduler`: sequential queue processor, schedule enforcement, sleep prevention, notifications, and retry recovery.
- `JobStore`: SwiftData-backed persistence API for jobs, schedule settings, and legacy JSON migration.
- `ReconstructionJob`: app-facing value model for input/output folders, Object Capture settings, progress, status, and completed outputs.
- `Settings/`: SwiftUI controls for folder selection and `PhotogrammetrySession.Configuration`.
- `ReconstructionCapability`: the detail levels, configuration options, and device support Object Capture offers on the current platform.
- `Platform/`: the shims that differ between macOS and the mobile builds — folder bookmarks and security-scoped access, revealing or sharing finished models, keeping the system awake, and the few platform-specific controls.
- `Processing/`: progress display and USDZ preview with RealityKit.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for a deeper code map and [CONTRIBUTING.md](CONTRIBUTING.md) for development conventions and test commands.

## GitHub Topics

Recommended repository topics:

```text
apple-object-capture, photogrammetry, macos, ios, ipados, swiftui, swiftdata, realitykit, usdz, 3d-reconstruction, object-capture, batch-processing
```

These topics help developers find the project when searching for SwiftUI photogrammetry tools, RealityKit Object Capture examples, macOS and iOS USDZ exporters, and batch 3D reconstruction workflows.

## Roadmap Ideas

- screenshot and demo media for the README
- export presets for common asset pipelines
- richer queue filtering and search
- per-job logs in the UI
- release packaging and notarized builds

## License

This project is licensed under the GNU General Public License v3.0. It also contains components originally provided by Apple Inc. under the MIT License. See [LICENSE](LICENSE) for the full license text and retain upstream notices in sample-derived files.
