# Reality Pulse

A macOS app that automates 3D model reconstruction from photos using Apple's Object Capture framework. Process multiple photo sets with various quality settings on a schedule, all in one batch queue.

## Features

- **Batch Job Queue** — Add multiple photo folders and process them sequentially without user interaction
- **Flexible Reconstruction Settings** — Configure separate detail levels, mesh types, masking, and texture options per job
- **Time-Window Scheduling** — Define allowed processing hours (e.g., process overnight) or delay start by a specific time
- **Persistence** — Queue and settings survive app restarts
- **Real-Time Progress** — Track individual job progress with estimated time remaining
- **Notifications** — Get notified when the queue completes or jobs fail
- **Error Recovery** — Retry failed jobs or remove them from the queue

## Getting Started

### Requirements

- macOS 14.0+
- Xcode 15.0+
- Apple Object Capture framework (built-in)

### Building

```bash
xcodebuild -project ObjectCaptureReconstruction.xcodeproj \
  -scheme ObjectCaptureReconstruction \
  -configuration Debug \
  -destination 'platform=macOS' \
  build
```

Or open `ObjectCaptureReconstruction.xcodeproj` in Xcode and build from there.

## Usage

### Adding Jobs

1. Click **Add Job** to open the job setup sheet
2. Select a folder of images (HEIF, JPEG, PNG supported)
3. Choose an output folder for the generated model
4. Enter a model name (e.g., "Vase-High-Quality")
5. Select reconstruction settings:
   - **Detail Level** — High, Medium, Low, or custom polygon count
   - **Mesh Type** — Standard or raw
   - **Masking** — Auto, manual, or disabled
   - **Texture Format** — PNG or JPEG with quality
   - **Texture Resolution** — Up to 4K
6. Optionally generate multiple models per job (e.g., High + Medium detail)
7. Click **Add to Queue**

### Scheduling

1. Click **Schedule** to configure processing time windows
2. Choose:
   - **Delayed Start** — Process starts at a specific date/time
   - **Allowed Hours** — Only process between two times each day (e.g., 22:00–06:00)
3. Save settings

### Processing

1. Click **Start** to begin processing jobs in order
2. Monitor progress:
   - Active job shows percentage complete
   - Failed jobs display error details
3. Use **Pause** to suspend processing or **Stop** to cancel
4. Right-click on a job to retry or edit (if pending)

### Output Models

Completed models are saved as USDZ files in the output folder you specified for each job.

## Architecture

- **AppDataModel** — Holds JobScheduler and UI state
- **JobScheduler** — Sequential processor respecting time windows and schedule
- **ReconstructionJob** — Per-job data including folders, settings, and status
- **JobStore** — Persists queue and settings to Application Support
- **Settings Views** — Modular UI for configuring reconstruction parameters

## Sandbox & Security

The app uses macOS sandbox with user-selected folder access. Folder bookmarks are persisted securely to maintain access across restarts.

# Licensing

This project is licensed under GPLv3, but contains components originally licensed by Apple Inc. under the MIT License.
