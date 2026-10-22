# App Review sample files

Sample photos for reviewing **Reality Pulse** (macOS). They let a reviewer exercise the app's main feature, turning a folder of photos into a 3D model, without needing their own photogrammetry capture.

## Download

<https://github.com/nuit-dhiver/reality-pulse/releases/download/review-samples-1/dice-sample-images.zip>

- File: `dice-sample-images.zip` (~58 MB, 32 JPEG photos plus an attribution file)
- Hosted as a GitHub release asset under the tag `review-samples-1`, which is kept permanently so the link stays valid for future reviews.
- SHA-256 is in [CHECKSUMS.txt](CHECKSUMS.txt).
- The photos are a third-party CC BY 4.0 test set. See [ATTRIBUTION.md](ATTRIBUTION.md).

## What the photos show

One purple die on a white background, photographed on a turntable from varied angles. Each photo is 4272 × 2848 px. This is a standard set for testing photogrammetry software.

## How to test

1. Download and unzip `dice-sample-images.zip`. You get a folder named `dice-sample-images`.
2. Open Reality Pulse and click **Add Job**.
3. Leave **Input Type** on **Image Folder** and click **Choose…**. Select the unzipped `dice-sample-images` folder. The app should report 32 images.
4. Enter any **Model Name**, for example `Dice`.
5. Click **Where:** **Choose…** and pick any output folder.
6. Leave **Quality** on **Medium** and click **Add to Queue**.
7. Click **Start**.

## Expected result

- A progress bar and percentage appear on the job.
- On an Apple silicon Mac with 16 GB of memory this took about 3 minutes. Other Macs may be faster or slower.
- When the job completes it shows a green check, and the output folder contains `<Model Name>-medium.usdz` (about 16 MB). Open it with Quick Look or Preview to see the 3D die.

## Requirements

- macOS 15.0 or newer
- A Mac that supports Apple Object Capture (RealityKit photogrammetry)
