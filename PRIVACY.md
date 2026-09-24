# Privacy Policy

_Last updated: September 24, 2026_

Reality Pulse is a macOS app that turns folders of photos into 3D models using Apple Object Capture. This policy explains what happens to your data when you use it.

## The short version

Reality Pulse does not collect, transmit, sell, or share any personal data. Everything the app does happens on your Mac.

## Data the app handles

- **Your photos and videos.** Reality Pulse reads the image or video folders you choose so it can reconstruct 3D models. These files are processed locally by Apple's RealityKit frameworks and never leave your Mac.
- **Generated models.** USDZ, glTF, glb, and Gaussian Splat (`.ply`) files are written only to the output folders you choose.
- **Queue and settings.** Your job queue, job history, schedule, and preferences are stored locally inside the app's sandboxed container on your Mac so they survive relaunches. This includes security-scoped bookmarks that let the app reopen folders you previously selected.
- **File metadata.** The app reads file creation dates of images in the folders you select, only to order and describe your capture sets.

## Network access

Reality Pulse makes no network connections. It contains no analytics, advertising, crash-reporting, or tracking SDKs.

## Notifications

If you turn on "Notify me when jobs finish" in Schedule Settings, the app asks macOS for permission to show local notifications about job and queue status. These notifications are created on your Mac and are not sent to any server. You can turn them off at any time in the app or in System Settings › Notifications.

## Data deletion

Removing a job from the queue deletes its record from the app's local storage. Deleting the app removes its container, including all queue and settings data. Photos you selected and models you exported stay where you put them, under your control.

## Children

Reality Pulse does not knowingly collect information from anyone, including children under 13.

## Changes to this policy

If this policy changes, the updated version will be published in this file with a new "Last updated" date. The full history of changes is available in this repository's commit history.

## Contact

Questions about this policy can be raised by opening an issue at [github.com/nuit-dhiver/reality-pulse/issues](https://github.com/nuit-dhiver/reality-pulse/issues).
