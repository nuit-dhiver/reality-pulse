/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Data model for a single Gaussian Splat training job in the batch queue.
*/

import Foundation

/// Represents a single Gaussian Splat training job with either an in-app COLMAP
/// preprocessing step or a user-supplied COLMAP dataset.
struct GaussianSplatTrainingJob: Identifiable, Codable {
    let id: UUID
    var imageFolder: URL
    var outputFolder: URL
    var jobName: String

    var inputMode: GaussianSplatTrainingInputMode
    var importedCOLMAPFolder: URL?
    var trainingConfiguration: CodableGaussianSplatTrainingConfiguration
    var sfmConfiguration: CodableSfMConfiguration

    var status: JobStatus = .pending
    var progress: Double = 0
    var currentPhase: String = ""
    var errorMessage: String?
    var createdAt: Date
    var resultSummary: GaussianSplatTrainingResultSummary?

    /// Security-scoped bookmark data for persisting sandbox access across launches.
    var imageFolderBookmark: Data?
    var outputFolderBookmark: Data?
    var importedCOLMAPFolderBookmark: Data?

    init(
        imageFolder: URL,
        outputFolder: URL,
        jobName: String,
        inputMode: GaussianSplatTrainingInputMode,
        importedCOLMAPFolder: URL? = nil,
        trainingConfiguration: CodableGaussianSplatTrainingConfiguration = CodableGaussianSplatTrainingConfiguration(),
        sfmConfiguration: CodableSfMConfiguration = CodableSfMConfiguration()
    ) {
        self.id = UUID()
        self.imageFolder = imageFolder
        self.outputFolder = outputFolder
        self.jobName = jobName
        self.inputMode = inputMode
        self.importedCOLMAPFolder = importedCOLMAPFolder
        self.trainingConfiguration = trainingConfiguration
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
        if let importedCOLMAPFolder {
            self.importedCOLMAPFolderBookmark = try? importedCOLMAPFolder.bookmarkData(
                options: .withSecurityScope,
                includingResourceValuesForKeys: nil,
                relativeTo: nil
            )
        }
    }

    /// Directory where this job's training outputs will live.
    var trainingOutputDirectory: URL {
        outputFolder.appending(path: "\(jobName)-gaussian-splat")
    }

    /// Resolve security-scoped bookmarks to restore sandbox access after relaunch.
    /// Returns updated URLs; callers must call `startAccessingSecurityScopedResource`.
    mutating func resolveBookmarks() -> (image: URL?, output: URL?, importedCOLMAP: URL?) {
        var imageURL: URL?
        var outputURL: URL?
        var importedCOLMAPURL: URL?

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

        if let data = importedCOLMAPFolderBookmark {
            var stale = false
            if let url = try? URL(resolvingBookmarkData: data, options: .withSecurityScope, bookmarkDataIsStale: &stale) {
                importedCOLMAPURL = url
                importedCOLMAPFolder = url
                if stale { importedCOLMAPFolderBookmark = try? url.bookmarkData(options: .withSecurityScope) }
            }
        }

        return (imageURL, outputURL, importedCOLMAPURL)
    }
}

enum GaussianSplatTrainingInputMode: String, Codable, CaseIterable, Sendable {
    case runCOLMAPInApp
    case useExistingCOLMAP

    var displayName: String {
        switch self {
        case .runCOLMAPInApp:
            return "Run COLMAP in App"
        case .useExistingCOLMAP:
            return "Use Existing COLMAP"
        }
    }

    var shortDescription: String {
        switch self {
        case .runCOLMAPInApp:
            return "In-app COLMAP"
        case .useExistingCOLMAP:
            return "Imported COLMAP"
        }
    }
}

struct GaussianSplatTrainingResultSummary: Codable, Equatable, Sendable {
    var exportedPLYCount: Int
    var totalIterations: Int
}