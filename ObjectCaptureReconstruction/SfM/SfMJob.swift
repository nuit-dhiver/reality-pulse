/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Data model for a standalone SfM (Structure from Motion) job in the batch queue.
*/

import Foundation
import os

private let logger = Logger(subsystem: ObjectCaptureReconstructionApp.subsystem,
                            category: "SfMJob")

/// Represents a standalone SfM job that runs COLMAP to estimate camera poses
/// and produce a sparse point cloud, exporting results in COLMAP binary format.
struct SfMJob: Identifiable, Codable {
    let id: UUID
    var imageFolder: URL
    var outputFolder: URL
    var jobName: String

    var sfmConfiguration: CodableSfMConfiguration
    var status: JobStatus = .pending
    var progress: Double = 0
    var currentPhase: String = ""
    var errorMessage: String?
    var createdAt: Date

    /// Security-scoped bookmark data for persisting sandbox access across launches.
    var imageFolderBookmark: Data?
    var outputFolderBookmark: Data?

    /// Summary of results after completion.
    var resultSummary: SfMResultSummary?

    init(
        imageFolder: URL,
        outputFolder: URL,
        jobName: String,
        sfmConfiguration: CodableSfMConfiguration = CodableSfMConfiguration()
    ) {
        self.id = UUID()
        self.imageFolder = imageFolder
        self.outputFolder = outputFolder
        self.jobName = jobName
        self.sfmConfiguration = sfmConfiguration
        self.createdAt = Date()

        self.imageFolderBookmark = try? imageFolder.bookmarkData(
            options: .withSecurityScope,
            includingResourceValuesForKeys: nil,
            relativeTo: nil
        )
        self.outputFolderBookmark = try? outputFolder.bookmarkData(
            options: .withSecurityScope,
            includingResourceValuesForKeys: nil,
            relativeTo: nil
        )
    }

    /// The directory where COLMAP binary outputs will be written.
    var colmapOutputDirectory: URL {
        outputFolder.appending(path: "\(jobName)-sfm")
    }

    // MARK: - Bookmark resolution

    mutating func resolveBookmarks() -> (image: URL?, output: URL?) {
        var imageURL: URL?
        var outputURL: URL?

        if let data = imageFolderBookmark {
            var stale = false
            if let url = try? URL(resolvingBookmarkData: data, options: .withSecurityScope, bookmarkDataIsStale: &stale) {
                imageURL = url
                imageFolder = url
                if stale { imageFolderBookmark = try? url.bookmarkData(options: .withSecurityScope) }
            }
        }

        if let data = outputFolderBookmark {
            var stale = false
            if let url = try? URL(resolvingBookmarkData: data, options: .withSecurityScope, bookmarkDataIsStale: &stale) {
                outputURL = url
                outputFolder = url
                if stale { outputFolderBookmark = try? url.bookmarkData(options: .withSecurityScope) }
            }
        }

        return (imageURL, outputURL)
    }
}

/// Lightweight summary stored after SfM completion (avoids persisting full result).
struct SfMResultSummary: Codable, Equatable {
    var registeredImages: Int
    var totalImages: Int
    var sparsePoints: Int
    var cameras: Int
    var meanReprojectionError: Double?
}
