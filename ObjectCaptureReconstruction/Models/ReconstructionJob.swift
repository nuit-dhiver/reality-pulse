/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Data model for a single reconstruction job in the batch queue.
*/

import Foundation
import RealityKit

/// Represents a single reconstruction job: one image folder producing one or
/// more 3D models at the selected detail levels.
struct ReconstructionJob: Identifiable, Codable {
    let id: UUID
    var imageFolder: URL
    var modelFolder: URL
    var modelName: String

    var sessionConfiguration: CodableSessionConfiguration
    var primaryDetailLevel: CodableDetailLevel
    var additionalDetailLevels: CodableDetailLevelOptions

    var status: JobStatus = .pending
    var progress: Double = 0
    var errorMessage: String?
    var boundingBoxAvailable: Bool = false
    var createdAt: Date

    /// Whether to run SfM (COLMAP camera pose estimation) before reconstruction.
    var runSfMFirst: Bool = false
    var sfmConfiguration: CodableSfMConfiguration?

    /// Security-scoped bookmark data for persisting sandbox access across launches.
    var imageFolderBookmark: Data?
    var modelFolderBookmark: Data?

    init(
        imageFolder: URL,
        modelFolder: URL,
        modelName: String,
        sessionConfiguration: CodableSessionConfiguration = CodableSessionConfiguration(),
        primaryDetailLevel: CodableDetailLevel = .medium,
        additionalDetailLevels: CodableDetailLevelOptions = CodableDetailLevelOptions()
    ) {
        self.id = UUID()
        self.imageFolder = imageFolder
        self.modelFolder = modelFolder
        self.modelName = modelName
        self.sessionConfiguration = sessionConfiguration
        self.primaryDetailLevel = primaryDetailLevel
        self.additionalDetailLevels = additionalDetailLevels
        self.createdAt = Date()

        self.imageFolderBookmark = try? imageFolder.bookmarkData(
            options: .withSecurityScope,
            includingResourceValuesForKeys: nil,
            relativeTo: nil
        )
        self.modelFolderBookmark = try? modelFolder.bookmarkData(
            options: .withSecurityScope,
            includingResourceValuesForKeys: nil,
            relativeTo: nil
        )
    }

    // MARK: - Detail level helpers

    /// All detail levels requested for this job (primary + any advanced selections).
    var allRequestedDetailLevels: Set<CodableDetailLevel> {
        var levels: Set<CodableDetailLevel> = [primaryDetailLevel]
        if additionalDetailLevels.isSelected {
            if additionalDetailLevels.preview { levels.insert(.preview) }
            if additionalDetailLevels.reduced { levels.insert(.reduced) }
            if additionalDetailLevels.medium { levels.insert(.medium) }
            if additionalDetailLevels.full { levels.insert(.full) }
            if additionalDetailLevels.raw { levels.insert(.raw) }
        }
        return levels
    }

    /// Build `PhotogrammetrySession.Request` entries for all requested detail levels.
    func createReconstructionRequests() -> [PhotogrammetrySession.Request] {
        allRequestedDetailLevels.map { level in
            let url = modelFolder.appending(path: "\(modelName)-\(level.rawValue).usdz")
            return .modelFile(url: url, detail: level.toFrameworkType)
        }
    }

    // MARK: - Bookmark resolution

    /// Resolve security-scoped bookmarks to restore sandbox access after relaunch.
    /// Returns updated URLs; callers must call `startAccessingSecurityScopedResource`.
    mutating func resolveBookmarks() -> (image: URL?, model: URL?) {
        var imageURL: URL?
        var modelURL: URL?

        if let data = imageFolderBookmark {
            var stale = false
            if let url = try? URL(resolvingBookmarkData: data, options: .withSecurityScope, bookmarkDataIsStale: &stale) {
                imageURL = url
                imageFolder = url
                if stale { imageFolderBookmark = try? url.bookmarkData(options: .withSecurityScope) }
            }
        }

        if let data = modelFolderBookmark {
            var stale = false
            if let url = try? URL(resolvingBookmarkData: data, options: .withSecurityScope, bookmarkDataIsStale: &stale) {
                modelURL = url
                modelFolder = url
                if stale { modelFolderBookmark = try? url.bookmarkData(options: .withSecurityScope) }
            }
        }

        return (imageURL, modelURL)
    }
}

// MARK: - Supporting types

enum JobStatus: String, Codable, CaseIterable {
    case pending
    case running
    case completed
    case failed
    case cancelled
}

enum CodableDetailLevel: String, Codable, CaseIterable, Hashable {
    case preview, reduced, medium, full, raw, custom

    init(from detail: PhotogrammetrySession.Request.Detail) {
        switch detail {
        case .preview:  self = .preview
        case .reduced:  self = .reduced
        case .medium:   self = .medium
        case .full:     self = .full
        case .raw:      self = .raw
        case .custom:   self = .custom
        @unknown default: self = .medium
        }
    }

    var toFrameworkType: PhotogrammetrySession.Request.Detail {
        switch self {
        case .preview:  return .preview
        case .reduced:  return .reduced
        case .medium:   return .medium
        case .full:     return .full
        case .raw:      return .raw
        case .custom:   return .custom
        }
    }
}

struct CodableDetailLevelOptions: Codable, Equatable {
    var isSelected: Bool = false
    var preview: Bool = false
    var reduced: Bool = false
    var medium: Bool = false
    var full: Bool = false
    var raw: Bool = false
}
