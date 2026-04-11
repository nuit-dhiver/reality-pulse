/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Helper class for extracting frames from a video file using AVFoundation.
RealityKit's PhotogrammetrySession requires a folder of still images as input,
so this helper converts video files into per-frame JPEG images saved in a
temporary directory that is then used as the image folder for reconstruction.
*/

import AVFoundation
import CoreImage
import Foundation
import ImageIO
import UniformTypeIdentifiers
import os

private let logger = Logger(subsystem: ObjectCaptureReconstructionApp.subsystem,
                            category: "VideoHelper")

class VideoHelper {
    /// Supported video file extensions.
    static let validVideoSuffixes: Set<String> = ["mp4", "mov", "m4v"]

    enum ExtractionError: Error, LocalizedError {
        case invalidVideoFile
        case noVideoTrack
        case cannotCreateOutputDirectory

        var errorDescription: String? {
            switch self {
            case .invalidVideoFile:
                return "The selected file is not a valid video or its duration could not be read."
            case .noVideoTrack:
                return "No video track found in the selected file."
            case .cannotCreateOutputDirectory:
                return "Cannot create a temporary directory for the extracted frames."
            }
        }
    }

    /// Extracts frames from a video file and saves them as JPEG images.
    ///
    /// - Parameters:
    ///   - videoURL: The URL of the video file to process.
    ///   - framesPerSecond: How many frames to capture per second of video (default: 2).
    ///     Values below 0.5 are clamped to 0.5 (one frame every two seconds).
    ///   - progressHandler: Called on the main actor with a value in 0.0–1.0 as frames are written.
    /// - Returns: A tuple of the output directory URL and the number of successfully written frames.
    static func extractFrames(
        from videoURL: URL,
        framesPerSecond: Double = 2.0,
        progressHandler: @escaping @MainActor (Double) -> Void
    ) async throws -> (outputDirectory: URL, frameCount: Int) {
        let asset = AVURLAsset(url: videoURL)

        let tracks = try await asset.loadTracks(withMediaType: .video)
        guard !tracks.isEmpty else {
            throw ExtractionError.noVideoTrack
        }

        let duration = try await asset.load(.duration)
        let durationSeconds = CMTimeGetSeconds(duration)
        guard durationSeconds > 0 else {
            throw ExtractionError.invalidVideoFile
        }

        // Create a unique temporary directory for this extraction run.
        let outputDirectory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appending(path: "reality-pulse-frames-\(UUID().uuidString)", directoryHint: .isDirectory)
        do {
            try FileManager.default.createDirectory(at: outputDirectory, withIntermediateDirectories: true)
        } catch {
            throw ExtractionError.cannotCreateOutputDirectory
        }

        // Build the list of sample times at the requested cadence.
        let frameInterval = 1.0 / max(framesPerSecond, 0.5)
        var times: [CMTime] = []
        var t = 0.0
        while t < durationSeconds {
            times.append(CMTime(seconds: t, preferredTimescale: 600))
            t += frameInterval
        }

        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        let tolerance = CMTime(seconds: frameInterval / 2.0, preferredTimescale: 600)
        generator.requestedTimeToleranceBefore = tolerance
        generator.requestedTimeToleranceAfter = tolerance

        let totalFrames = times.count
        var frameCount = 0

        for (index, time) in times.enumerated() {
            if Task.isCancelled { break }

            let cgImage: CGImage
            do {
                (cgImage, _) = try await generator.image(at: time)
            } catch {
                logger.warning("Frame extraction failed at \(CMTimeGetSeconds(time), format: .fixed(precision: 2))s: \(error)")
                continue
            }

            let frameFilename = String(format: "frame_%06d.jpg", index)
            let frameURL = outputDirectory.appending(path: frameFilename)

            if let destination = CGImageDestinationCreateWithURL(
                frameURL as CFURL, UTType.jpeg.identifier as CFString, 1, nil
            ) {
                let properties = [kCGImageDestinationLossyCompressionQuality: 0.9] as CFDictionary
                CGImageDestinationAddImage(destination, cgImage, properties)
                if CGImageDestinationFinalize(destination) {
                    frameCount += 1
                }
            }

            await MainActor.run { progressHandler(Double(index + 1) / Double(totalFrames)) }
        }
        logger.log("Extracted \(frameCount) of \(totalFrames) frames from \(videoURL.lastPathComponent)")
        return (outputDirectory, frameCount)
    }

    /// Returns `true` when the URL's file extension matches a supported video format.
    static func isVideoFile(_ url: URL) -> Bool {
        guard url.isFileURL else { return false }
        return validVideoSuffixes.contains(url.pathExtension.lowercased())
    }

    /// Generates a thumbnail from the first frame of the video at the given URL.
    static func thumbnail(for videoURL: URL) async -> CGImage? {
        let asset = AVURLAsset(url: videoURL)
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        return try? await generator.image(at: .zero).image
    }
}
