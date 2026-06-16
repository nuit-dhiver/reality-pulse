/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
SwiftData model for persisted reconstruction jobs.
*/

import Foundation
import SwiftData

@Model
final class PersistentJob {
    @Attribute(.unique) var id: UUID
    var queueOrder: Int
    var imageFolderPath: String
    var modelFolderPath: String
    var modelName: String
    var sessionConfigurationData: Data
    var primaryDetailLevelRawValue: String
    var additionalDetailLevelsData: Data
    var statusRawValue: String
    var progress: Double
    var errorMessage: String?
    var boundingBoxAvailable: Bool
    var createdAt: Date
    var updatedAt: Date
    var completedOutputFilenamesData: Data?
    var exportFormatsData: Data?
    var imageFolderBookmark: Data?
    var modelFolderBookmark: Data?

    init(job: ReconstructionJob, queueOrder: Int) throws {
        id = job.id
        self.queueOrder = queueOrder
        imageFolderPath = job.imageFolder.path
        modelFolderPath = job.modelFolder.path
        modelName = job.modelName
        sessionConfigurationData = try JSONEncoder().encode(job.sessionConfiguration)
        primaryDetailLevelRawValue = job.primaryDetailLevel.rawValue
        additionalDetailLevelsData = try JSONEncoder().encode(job.additionalDetailLevels)
        statusRawValue = job.status.rawValue
        progress = job.progress
        errorMessage = job.errorMessage
        boundingBoxAvailable = job.boundingBoxAvailable
        createdAt = job.createdAt
        updatedAt = Date()
        completedOutputFilenamesData = try JSONEncoder().encode(job.completedOutputFilenames ?? [])
        exportFormatsData = try JSONEncoder().encode(job.exportFormats)
        imageFolderBookmark = job.imageFolderBookmark
        modelFolderBookmark = job.modelFolderBookmark
    }

    func update(from job: ReconstructionJob, queueOrder: Int? = nil) throws {
        if let queueOrder {
            self.queueOrder = queueOrder
        }
        imageFolderPath = job.imageFolder.path
        modelFolderPath = job.modelFolder.path
        modelName = job.modelName
        sessionConfigurationData = try JSONEncoder().encode(job.sessionConfiguration)
        primaryDetailLevelRawValue = job.primaryDetailLevel.rawValue
        additionalDetailLevelsData = try JSONEncoder().encode(job.additionalDetailLevels)
        statusRawValue = job.status.rawValue
        progress = job.progress
        errorMessage = job.errorMessage
        boundingBoxAvailable = job.boundingBoxAvailable
        createdAt = job.createdAt
        updatedAt = Date()
        completedOutputFilenamesData = try JSONEncoder().encode(job.completedOutputFilenames ?? [])
        exportFormatsData = try JSONEncoder().encode(job.exportFormats)
        imageFolderBookmark = job.imageFolderBookmark
        modelFolderBookmark = job.modelFolderBookmark
    }

    func toJob() throws -> ReconstructionJob {
        let sessionConfiguration = try JSONDecoder().decode(
            CodableSessionConfiguration.self,
            from: sessionConfigurationData
        )
        let additionalDetailLevels = try JSONDecoder().decode(
            CodableDetailLevelOptions.self,
            from: additionalDetailLevelsData
        )
        let completedOutputFilenames = try completedOutputFilenamesData.map {
            try JSONDecoder().decode(Set<String>.self, from: $0)
        } ?? []
        let exportFormats = try exportFormatsData.map {
            try JSONDecoder().decode(Set<ModelExportFormat>.self, from: $0)
        } ?? []

        return ReconstructionJob(
            id: id,
            imageFolder: URL(fileURLWithPath: imageFolderPath),
            modelFolder: URL(fileURLWithPath: modelFolderPath),
            modelName: modelName,
            sessionConfiguration: sessionConfiguration,
            primaryDetailLevel: CodableDetailLevel(rawValue: primaryDetailLevelRawValue) ?? .medium,
            additionalDetailLevels: additionalDetailLevels,
            status: JobStatus(rawValue: statusRawValue) ?? .pending,
            progress: progress,
            errorMessage: errorMessage,
            boundingBoxAvailable: boundingBoxAvailable,
            createdAt: createdAt,
            completedOutputFilenames: completedOutputFilenames,
            exportFormats: exportFormats,
            imageFolderBookmark: imageFolderBookmark,
            modelFolderBookmark: modelFolderBookmark
        )
    }
}
