---
title: Architecture
description: How Reality Pulse automates Object Capture reconstruction and multi-format export.
order: 2
---

Reality Pulse is a native macOS SwiftUI app built around Apple Object Capture's `PhotogrammetrySession`. Its main responsibility is to turn single-session reconstruction into a persistent, scheduled, retryable automation pipeline—from photo folders to USDZ meshes and optional glTF, glB, or Gaussian splat exports.

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
6. Optional post-processing converts completed USDZ files to glTF, glB, or Gaussian splat `.ply`.
7. `JobStore` persists every meaningful queue and schedule mutation through SwiftData.

## Main Components

### App Entry

`ObjectCaptureReconstructionApp` owns the app-level SwiftData container and creates a single `Reality Pulse` window.

### App State

`AppDataModel` holds shared UI state: the scheduler, job setup sheet state, schedule sheet state, editing state, and persistence alert state. SwiftUI views receive it through `@Environment(AppDataModel.self)`.

### Queue And Scheduler

`JobScheduler` owns the in-memory job array and scheduler state. It handles add/remove/edit/retry/reorder operations, pending-job selection, delay and allowed-hours scheduling, pause-between-jobs behavior, cancellation, sleep prevention, notifications, and retry recovery.

### Persistence

`JobStore` is the SwiftData-backed persistence layer. It stores jobs, schedule settings, and handles one-time legacy JSON migration.

### Reconstruction Model

`ReconstructionJob` is the app-facing value model for input/output folders, Object Capture settings, progress, status, and completed outputs.

### UI Areas

- `Settings/`: SwiftUI controls for folder selection, `PhotogrammetrySession.Configuration`, and export format selection.
- `Processing/`: progress display and USDZ preview with RealityKit.
- `Queue/`: dashboard and job setup views.
- `Export/`: USDZ to glTF/glB conversion and Gaussian splat generation.

For deeper implementation notes, see the [architecture guide in the repository](https://github.com/nuit-dhiver/reality-pulse/blob/main/docs/ARCHITECTURE.md).
