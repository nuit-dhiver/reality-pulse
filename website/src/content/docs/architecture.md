---
title: Architecture
description: How Reality Pulse is structured around a persistent Object Capture queue.
order: 2
---

Reality Pulse is a native SwiftUI app built around Apple Object Capture's `PhotogrammetrySession`, building from one target for macOS, iOS, and iPadOS. The app's main responsibility is to turn single-session reconstruction into a persistent, scheduled, retryable queue.

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

`ObjectCaptureReconstructionApp` owns the app-level SwiftData container and creates the app scene: a single `Reality Pulse` window on macOS, and a window group on iPhone and iPad.

### App State

`AppDataModel` holds shared UI state: the scheduler, job setup sheet state, schedule sheet state, editing state, and persistence alert state. SwiftUI views receive it through `@Environment(AppDataModel.self)`.

### Queue And Scheduler

`JobScheduler` owns the in-memory job array and scheduler state. It handles add/remove/edit/retry/reorder operations, pending-job selection, delay and allowed-hours scheduling, pause-between-jobs behavior, cancellation, sleep prevention, notifications, and retry recovery.

### Persistence

`JobStore` is the SwiftData-backed persistence layer. It stores jobs, schedule settings, and handles one-time legacy JSON migration.

### Reconstruction Model

`ReconstructionJob` is the app-facing value model for input/output folders, Object Capture settings, progress, status, and completed outputs.

### Platform Capabilities

`ReconstructionCapability` reports what Object Capture supports on the current platform: every detail level on macOS, reduced detail alone on iOS and iPadOS, whether the custom detail and mesh primitive settings exist, and whether the device can reconstruct at all. `Platform/` holds the matching shims for bookmarks, security-scoped access, revealing versus sharing finished models, and keeping the system awake.

### UI Areas

- `Settings/`: SwiftUI controls for folder selection and `PhotogrammetrySession.Configuration`.
- `Platform/`: the AppKit and UIKit differences behind shared SwiftUI helpers.
- `Processing/`: progress display and USDZ preview with RealityKit.
- `Queue/`: dashboard and job setup views.

For deeper implementation notes, see the [architecture guide in the repository](https://github.com/nuit-dhiver/reality-pulse/blob/main/docs/ARCHITECTURE.md).
