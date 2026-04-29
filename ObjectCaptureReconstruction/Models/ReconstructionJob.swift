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
    var exportFormats: Set<ModelExportFormat>

    var status: JobStatus = .pending
    var progress: Double = 0
    var errorMessage: String?
    var boundingBoxAvailable: Bool = false
    var createdAt: Date

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
        exportFormats: Set<ModelExportFormat> = [.usdz]
    ) {
        self.id = id
        self.imageFolder = imageFolder
        self.modelFolder = modelFolder
        self.modelName = modelName
        self.sessionConfiguration = sessionConfiguration
        self.primaryDetailLevel = primaryDetailLevel
        self.additionalDetailLevels = additionalDetailLevels
        self.exportFormats = exportFormats.isEmpty ? [.usdz] : exportFormats
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
        createModelExportRequests().map(\.photogrammetryRequest)
    }

    /// Build Object Capture requests plus final export targets for each detail level.
    func createModelExportRequests() -> [ModelExportRequest] {
        let formats = exportFormats.isEmpty ? Set([.usdz]) : exportFormats
        return allRequestedDetailLevels
            .sorted { $0.rawValue < $1.rawValue }
            .flatMap { level -> [ModelExportRequest] in
                var requests: [ModelExportRequest] = []

                if formats.contains(.usdz) {
                    let url = modelFolder.appending(path: "\(modelName)-\(level.rawValue).usdz")
                    requests.append(ModelExportRequest(
                        detailLevel: level,
                        photogrammetryRequest: .modelFile(url: url, detail: level.toFrameworkType),
                        targets: [ModelExportTarget(format: .usdz, url: url)],
                        intermediateDirectory: nil
                    ))
                }

                let gltfFormats = formats.intersection([.gltf, .glb])
                if !gltfFormats.isEmpty {
                    let sourceDirectory = modelFolder.appending(
                        path: ".\(modelName)-\(level.rawValue)-gltf-source-\(id.uuidString)"
                    )
                    let targets = gltfFormats
                        .sorted { $0.rawValue < $1.rawValue }
                        .map { format in
                            ModelExportTarget(
                                format: format,
                                url: modelFolder.appending(path: "\(modelName)-\(level.rawValue).\(format.fileExtension)")
                            )
                        }
                    requests.append(ModelExportRequest(
                        detailLevel: level,
                        photogrammetryRequest: .modelFile(url: sourceDirectory, detail: level.toFrameworkType),
                        targets: targets,
                        intermediateDirectory: sourceDirectory
                    ))
                }

                return requests
            }
    }

    func outputFilenames() -> [String] {
        let formats = exportFormats.isEmpty ? Set([.usdz]) : exportFormats
        return allRequestedDetailLevels
            .sorted { $0.rawValue < $1.rawValue }
            .flatMap { level in
                formats
                    .sorted { $0.rawValue < $1.rawValue }
                    .map { "\(modelName)-\(level.rawValue).\($0.fileExtension)" }
            }
    }

    // MARK: - Codable

    enum CodingKeys: String, CodingKey {
        case id
        case imageFolder
        case modelFolder
        case modelName
        case sessionConfiguration
        case primaryDetailLevel
        case additionalDetailLevels
        case exportFormats
        case status
        case progress
        case errorMessage
        case boundingBoxAvailable
        case createdAt
        case imageFolderBookmark
        case modelFolderBookmark
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(UUID.self, forKey: .id)
        imageFolder = try container.decode(URL.self, forKey: .imageFolder)
        modelFolder = try container.decode(URL.self, forKey: .modelFolder)
        modelName = try container.decode(String.self, forKey: .modelName)
        sessionConfiguration = try container.decode(CodableSessionConfiguration.self, forKey: .sessionConfiguration)
        primaryDetailLevel = try container.decode(CodableDetailLevel.self, forKey: .primaryDetailLevel)
        additionalDetailLevels = try container.decode(CodableDetailLevelOptions.self, forKey: .additionalDetailLevels)
        exportFormats = try container.decodeIfPresent(Set<ModelExportFormat>.self, forKey: .exportFormats) ?? [.usdz]
        status = try container.decodeIfPresent(JobStatus.self, forKey: .status) ?? .pending
        progress = try container.decodeIfPresent(Double.self, forKey: .progress) ?? 0
        errorMessage = try container.decodeIfPresent(String.self, forKey: .errorMessage)
        boundingBoxAvailable = try container.decodeIfPresent(Bool.self, forKey: .boundingBoxAvailable) ?? false
        createdAt = try container.decode(Date.self, forKey: .createdAt)
        imageFolderBookmark = try container.decodeIfPresent(Data.self, forKey: .imageFolderBookmark)
        modelFolderBookmark = try container.decodeIfPresent(Data.self, forKey: .modelFolderBookmark)
        if exportFormats.isEmpty {
            exportFormats = [.usdz]
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

struct ModelExportRequest {
    var detailLevel: CodableDetailLevel
    var photogrammetryRequest: PhotogrammetrySession.Request
    var targets: [ModelExportTarget]
    var intermediateDirectory: URL?

    var outputURL: URL? {
        if case .modelFile(let url, _, _) = photogrammetryRequest {
            return url
        }
        return nil
    }
}

struct ModelExportTarget {
    var format: ModelExportFormat
    var url: URL
}
