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
            imageFolderBookmark: imageFolderBookmark,
            modelFolderBookmark: modelFolderBookmark
        )
    }
}
