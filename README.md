# Reality Pulse

[![Build macOS app](https://github.com/nuit-dhiver/reality-pulse/actions/workflows/swift.yml/badge.svg)](https://github.com/nuit-dhiver/reality-pulse/actions/workflows/swift.yml)
[![Swift](https://img.shields.io/badge/Swift-5-orange.svg)](https://www.swift.org/)
[![macOS](https://img.shields.io/badge/macOS-14%2B-blue.svg)](https://developer.apple.com/macos/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

**Reality Pulse is a macOS photogrammetry queue for Apple Object Capture.** It turns folders of photos into USDZ 3D models with a SwiftUI batch interface, RealityKit reconstruction settings, scheduled processing windows, SwiftData persistence, and crash-aware retry recovery.

Use it when you have many capture sets to reconstruct and want a local Apple Object Capture workflow that can run overnight, survive relaunches, keep completed history, and export multiple quality levels from one job.

## Why Reality Pulse?

Apple Object Capture is powerful, but running one folder at a time is tedious when you are scanning products, props, archive objects, handmade pieces, or environment assets. Reality Pulse wraps `PhotogrammetrySession` in a persistent queue so you can prepare a batch, choose output quality, and let the Mac process the work in order.

## Features

- **Batch Apple Object Capture queue**: add multiple image folders and process them sequentially.
- **USDZ photogrammetry output**: export one or more `.usdz` models per job.
- **Multiple detail levels**: generate preview, reduced, medium, full, raw, or custom outputs from the same capture.
- **Custom reconstruction settings**: configure mesh primitive, masking, bounding-box handling, polygon limits, texture maps, texture format, and texture resolution.
- **Scheduled processing**: delay a queue run or restrict processing to allowed hours, including overnight windows.
- **Sleep prevention**: optionally keep macOS awake while the queue is active.
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

- macOS 14.0 or newer
- Xcode 15.0 or newer
- A Mac supported by Apple Object Capture / RealityKit photogrammetry
- Photo sets suitable for `PhotogrammetrySession`

## Build

Clone the repo and build the macOS app:

```bash
git clone https://github.com/nuit-dhiver/reality-pulse.git
cd reality-pulse

xcodebuild -project RealityPulse.xcodeproj \
  -scheme RealityPulse \
  -configuration Debug \
  -destination 'platform=macOS' \
  build
```

You can also open `RealityPulse.xcodeproj` in Xcode and build the `RealityPulse` scheme.

## Test

Run the persistence and scheduler-focused test suite with:

```bash
xcodebuild -project RealityPulse.xcodeproj \
  -scheme RealityPulse \
  -configuration Debug \
  -destination 'platform=macOS' \
  test
```

The current tests cover SwiftData persistence, schedule reloads, launch recovery, interrupted jobs, retry behavior, and one-time legacy JSON migration.

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

- `ObjectCaptureReconstructionApp`: creates the SwiftData model container and main SwiftUI window.
- `ContentView`: owns the app model and routes between the queue dashboard and processing views.
- `AppDataModel`: central UI/application state.
- `JobScheduler`: sequential queue processor, schedule enforcement, sleep prevention, notifications, and retry recovery.
- `JobStore`: SwiftData-backed persistence API for jobs, schedule settings, and legacy JSON migration.
- `ReconstructionJob`: app-facing value model for input/output folders, Object Capture settings, progress, status, and completed outputs.
- `Settings/`: SwiftUI controls for folder selection and `PhotogrammetrySession.Configuration`.
- `Processing/`: progress display and USDZ preview with RealityKit.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for a deeper code map and [CONTRIBUTING.md](CONTRIBUTING.md) for development conventions and test commands.

## GitHub Topics

Recommended repository topics:

```text
apple-object-capture, photogrammetry, macos, swiftui, swiftdata, realitykit, usdz, 3d-reconstruction, object-capture, batch-processing
```

These topics help developers find the project when searching for SwiftUI photogrammetry tools, RealityKit Object Capture examples, macOS USDZ exporters, and batch 3D reconstruction workflows.

## Roadmap Ideas

- screenshot and demo media for the README
- export presets for common asset pipelines
- richer queue filtering and search
- per-job logs in the UI
- release packaging and notarized builds

## License

This project is licensed under the GNU General Public License v3.0. It also contains components originally provided by Apple Inc. under the MIT License. See [LICENSE](LICENSE) for the full license text and retain upstream notices in sample-derived files.
