---
title: Getting Started
description: Build, test, and run Reality Pulse on macOS.
order: 1
---

Reality Pulse is a native macOS app. Clone the repository, open the Xcode project, and build the `RealityPulse` scheme.

## Requirements

- macOS 14.0 or newer
- Xcode 15.0 or newer
- A Mac supported by Apple Object Capture / RealityKit photogrammetry
- Photo sets suitable for `PhotogrammetrySession`

## Build

```bash
git clone https://github.com/nuit-dhiver/reality-pulse.git
cd reality-pulse

xcodebuild -project RealityPulse.xcodeproj \
  -scheme RealityPulse \
  -configuration Debug \
  -destination 'platform=macOS' \
  build
```

You can also open `RealityPulse.xcodeproj` in Xcode and build from the IDE.

## Test

```bash
xcodebuild -project RealityPulse.xcodeproj \
  -scheme RealityPulse \
  -configuration Debug \
  -destination 'platform=macOS' \
  test
```

The test suite covers SwiftData persistence, schedule reloads, launch recovery, interrupted jobs, retry behavior, and legacy JSON migration.

## Basic Workflow

1. Add a job from the queue dashboard.
2. Pick an image folder and an output folder.
3. Name the model and choose Object Capture settings.
4. Optionally enable additional detail-level exports.
5. Add more jobs, reorder them, and configure a schedule.
6. Press **Start** when you want the queue to run.

Reality Pulse restores the queue on launch, but it does not automatically start processing after a relaunch.

## Releases

Releases are published from Git tags:

```bash
git tag v1.1.0
git push origin v1.1.0
```

The release workflow builds `Reality Pulse.app`, runs tests, packages a zip, and attaches a SHA-256 checksum to the GitHub Release.
