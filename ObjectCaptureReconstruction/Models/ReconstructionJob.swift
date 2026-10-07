/*
See the LICENSE file for licensing information.

Abstract:
Data model for a single reconstruction job in the batch queue.
*/

import Foundation
import RealityKit

/// Represents a single queue job: either one image folder producing one or
/// more 3D models at the selected detail levels, or one existing USDZ file
/// converted to the selected export formats.
struct ReconstructionJob: Identifiable, Codable {
    let id: UUID
    var inputKind: JobInputKind = .images
    /// The job's input: an image folder for `.images` jobs, or the source USDZ
    /// file for `.usdzModel` conversion jobs. `imageFolderBookmark` follows suit.
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
    /// Settings for the `.textSculpture` export. `nil` means defaults.
    var textSculptureOptions: TextSculptureOptions?

    /// Security-scoped bookmark data for persisting sandbox access across launches.
    var imageFolderBookmark: Data?
    var modelFolderBookmark: Data?

    init(
        id: UUID = UUID(),
        inputKind: JobInputKind = .images,
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
        textSculptureOptions: TextSculptureOptions? = nil,
        imageFolderBookmark: Data? = nil,
        modelFolderBookmark: Data? = nil
    ) {
        self.id = id
        self.inputKind = inputKind
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
        self.textSculptureOptions = textSculptureOptions

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

    // MARK: - Conversion helpers

    /// Whether this job converts an existing USDZ file instead of reconstructing.
    var isConversionJob: Bool {
        inputKind == .usdzModel
    }

    func conversionFilename(for format: ModelExportFormat) -> String {
        format.conversionFilename(modelName: modelName)
    }

    func conversionURL(for format: ModelExportFormat) -> URL {
        modelFolder.appending(path: conversionFilename(for: format))
    }

    var sortedExportFormats: [ModelExportFormat] {
        exportFormats.sorted { $0.rawValue < $1.rawValue }
    }

    var conversionOutputURLs: [URL] {
        sortedExportFormats.map { conversionURL(for: $0) }
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
        format.exportFilename(modelName: modelName, level: level)
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

// MARK: - Decoding

extension ReconstructionJob {
    /// Tolerates keys added after the legacy `jobs.json` format (`inputKind`,
    /// `exportFormats`, `textSculptureOptions`) so the one-time JSON migration
    /// can still read old files.
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(UUID.self, forKey: .id)
        inputKind = try container.decodeIfPresent(JobInputKind.self, forKey: .inputKind) ?? .images
        imageFolder = try container.decode(URL.self, forKey: .imageFolder)
        modelFolder = try container.decode(URL.self, forKey: .modelFolder)
        modelName = try container.decode(String.self, forKey: .modelName)
        sessionConfiguration = try container.decode(CodableSessionConfiguration.self, forKey: .sessionConfiguration)
        primaryDetailLevel = try container.decode(CodableDetailLevel.self, forKey: .primaryDetailLevel)
        additionalDetailLevels = try container.decode(CodableDetailLevelOptions.self, forKey: .additionalDetailLevels)
        status = try container.decode(JobStatus.self, forKey: .status)
        progress = try container.decode(Double.self, forKey: .progress)
        errorMessage = try container.decodeIfPresent(String.self, forKey: .errorMessage)
        boundingBoxAvailable = try container.decode(Bool.self, forKey: .boundingBoxAvailable)
        createdAt = try container.decode(Date.self, forKey: .createdAt)
        completedOutputFilenames = try container.decodeIfPresent(Set<String>.self, forKey: .completedOutputFilenames)
        exportFormats = try container.decodeIfPresent(Set<ModelExportFormat>.self, forKey: .exportFormats) ?? []
        textSculptureOptions = try container.decodeIfPresent(TextSculptureOptions.self, forKey: .textSculptureOptions)
        imageFolderBookmark = try container.decodeIfPresent(Data.self, forKey: .imageFolderBookmark)
        modelFolderBookmark = try container.decodeIfPresent(Data.self, forKey: .modelFolderBookmark)
    }
}

// MARK: - Supporting types

enum JobInputKind: String, Codable {
    /// Reconstruct models from an image folder (also used for extracted video frames).
    case images
    /// Convert an existing USDZ file to the job's export formats.
    case usdzModel
}

enum ModelExportFormat: String, Codable, CaseIterable, Hashable {
    case gltf
    case glb
    case gaussianSplat
    case textSculpture

    var fileExtension: String {
        switch self {
        case .gltf: return "gltf"
        case .glb: return "glb"
        case .gaussianSplat: return "ply"
        case .textSculpture: return "glb"
        }
    }

    /// Appended to the base filename so formats sharing an extension don't collide.
    var filenameSuffix: String {
        switch self {
        case .textSculpture: return "-text"
        case .gltf, .glb, .gaussianSplat: return ""
        }
    }

    var displayName: String {
        switch self {
        case .gltf: return "glTF (.gltf)"
        case .glb: return "glb (.glb)"
        case .gaussianSplat: return "Gaussian Splat (.ply)"
        case .textSculpture: return "Text Sculpture (.glb)"
        }
    }

    func exportFilename(modelName: String, level: CodableDetailLevel) -> String {
        conversionFilename(modelName: "\(modelName)-\(level.rawValue)")
    }

    /// Filename for a conversion job's output, which has no detail level.
    func conversionFilename(modelName: String) -> String {
        "\(modelName)\(filenameSuffix).\(fileExtension)"
    }

    /// Formats that are still under development and may produce rough results.
    var isExperimental: Bool {
        self == .gaussianSplat
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
