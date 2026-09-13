---
title: Getting Started
description: Build, test, and run Reality Pulse on Mac, iPhone, and iPad.
order: 1
---

Reality Pulse is a native app for Mac, iPhone, and iPad. Clone the repository, open the Xcode project, and build the `RealityPulse` scheme for the destination you want.

## Requirements

- macOS 15.0 or newer, or iOS / iPadOS 18.0 or newer
- Xcode 16.0 or newer
- A device supported by Apple Object Capture / RealityKit photogrammetry
- Photo sets suitable for `PhotogrammetrySession`

## Build

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

You can also open `RealityPulse.xcodeproj` in Xcode, pick a destination, and build from the IDE. Reconstruction needs real hardware, so run on a device rather than a simulator to process a job.

## Test

```bash
xcodebuild -project RealityPulse.xcodeproj \
  -scheme RealityPulse \
  -configuration Debug \
  -destination 'platform=macOS' \
  test
```

Swap in an `-destination 'platform=iOS Simulator,name=iPhone 16'` to run the same suite on iOS. It covers SwiftData persistence, schedule reloads, launch recovery, interrupted jobs, retry behavior, legacy JSON migration, and platform reconstruction capabilities.

## On iPhone And iPad

The mobile builds run the same queue and scheduler as the Mac build, with the Object Capture features iOS offers:

- reduced detail is the only quality level, and the mesh primitive and custom polygon and texture controls are macOS-only
- reconstruction advances only while the app is in the foreground, so the sleep-prevention setting keeps the device awake
- input and output folders come from the system file picker, and the app's documents folder is visible in the Files app
- finished models are shared with the system share sheet instead of revealed in the Finder

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

The release workflow builds `Reality Pulse.app`, runs tests, packages a zip, and attaches a SHA-256 checksum to the GitHub Release. Releases publish the macOS app; iPhone and iPad builds come from Xcode with your own signing identity.
