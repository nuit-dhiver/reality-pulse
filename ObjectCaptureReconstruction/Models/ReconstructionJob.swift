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
    var completedOutputFilenames: Set<String>?
    var exportFormats: Set<ModelExportFormat> = []

    /// Security-scoped bookmark data for persisting sandbox access across launches.
    var imageFolderBookmark: Data?
    var modelFolderBookmark: Data?

    init(
        id: UUID = UUID(),
        imageFolder: URL,
        modelFolder: URL,
        modelName: String,
        sessionConfiguration: CodableSessionConfiguration = CodableSessionConfiguration(),
        primaryDetailLevel: CodableDetailLevel = .medium,
        additionalDetailLevels: CodableDetailLevelOptions = CodableDetailLevelOptions(),
        status: JobStatus = .pending,
        progress: Double = 0,
        errorMessage: String? = nil,
        boundingBoxAvailable: Bool = false,
        createdAt: Date = Date(),
        completedOutputFilenames: Set<String>? = [],
        exportFormats: Set<ModelExportFormat> = [],
        imageFolderBookmark: Data? = nil,
        modelFolderBookmark: Data? = nil
    ) {
        self.id = id
        self.imageFolder = imageFolder
        self.modelFolder = modelFolder
        self.modelName = modelName
        self.sessionConfiguration = sessionConfiguration
        self.primaryDetailLevel = primaryDetailLevel
        self.additionalDetailLevels = additionalDetailLevels
        self.status = status
        self.progress = progress
        self.errorMessage = errorMessage
        self.boundingBoxAvailable = boundingBoxAvailable
        self.createdAt = createdAt
        self.completedOutputFilenames = completedOutputFilenames
        self.exportFormats = exportFormats

        self.imageFolderBookmark = imageFolderBookmark ?? (try? imageFolder.bookmarkData(
            options: .withSecurityScope,
            includingResourceValuesForKeys: nil,
            relativeTo: nil
        ))
        self.modelFolderBookmark = modelFolderBookmark ?? (try? modelFolder.bookmarkData(
            options: .withSecurityScope,
            includingResourceValuesForKeys: nil,
            relativeTo: nil
        ))
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

    var requestedDetailLevels: [CodableDetailLevel] {
        allRequestedDetailLevels.sorted { $0.rawValue < $1.rawValue }
    }

    var requestedOutputCount: Int {
        requestedDetailLevels.count
    }

    func outputURL(for level: CodableDetailLevel) -> URL {
        modelFolder.appending(path: outputFilename(for: level))
    }

    func outputFilename(for level: CodableDetailLevel) -> String {
        "\(modelName)-\(level.rawValue).usdz"
    }

    func exportFilename(for level: CodableDetailLevel, format: ModelExportFormat) -> String {
        "\(modelName)-\(level.rawValue).\(format.fileExtension)"
    }

    func exportURL(for level: CodableDetailLevel, format: ModelExportFormat) -> URL {
        modelFolder.appending(path: exportFilename(for: level, format: format))
    }

    func hasCompletedOutputFile(
        for level: CodableDetailLevel,
        fileManager: FileManager = .default
    ) -> Bool {
        let filename = outputFilename(for: level)
        return completedOutputFilenames?.contains(filename) == true &&
            fileManager.fileExists(atPath: outputURL(for: level).path)
    }

    func completedOutputCount(fileManager: FileManager = .default) -> Int {
        requestedDetailLevels.filter {
            hasCompletedOutputFile(for: $0, fileManager: fileManager)
        }.count
    }

    func completedOutputFraction(fileManager: FileManager = .default) -> Double {
        let total = requestedOutputCount
        guard total > 0 else { return 0 }
        return Double(completedOutputCount(fileManager: fileManager)) / Double(total)
    }

    mutating func markOutputCompleted(at url: URL) {
        var filenames = completedOutputFilenames ?? []
        filenames.insert(url.lastPathComponent)
        completedOutputFilenames = filenames
    }

    /// Build `PhotogrammetrySession.Request` entries for all requested detail levels.
    func createReconstructionRequests(
        skippingCompletedOutputs: Bool = false,
        fileManager: FileManager = .default
    ) -> [PhotogrammetrySession.Request] {
        requestedDetailLevels.compactMap { level in
            if skippingCompletedOutputs &&
                hasCompletedOutputFile(for: level, fileManager: fileManager) {
                return nil
            }

            let url = outputURL(for: level)
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

enum ModelExportFormat: String, Codable, CaseIterable, Hashable {
    case gltf
    case glb

    var fileExtension: String { rawValue }

    var displayName: String {
        switch self {
        case .gltf: return "glTF (.gltf)"
        case .glb: return "glb (.glb)"
        }
    }
}

enum JobStatus: String, Codable, CaseIterable {
    case pending
    case running
    case completed
    case failed
    case cancelled
    case interrupted
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
