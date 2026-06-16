/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Observable draft for editing a reconstruction job in the setup sheet.
Exposes the same property interface as the old AppDataModel so that
settings views require only a type-name change in @Environment.
*/

import Foundation
import RealityKit
import os

private let logger = Logger(subsystem: ObjectCaptureReconstructionApp.subsystem,
                            category: "JobDraft")

@MainActor @Observable class JobDraft {

    // MARK: - Input mode

    enum InputMode: String, CaseIterable {
        case images
        case video
    }

    var inputMode: InputMode = .images

    // MARK: - Folder / name (same names as old AppDataModel)

    var imageFolder: URL?
    var modelFolder: URL?
    var modelName: String?
    var boundingBoxAvailable = false

    /// The source video file selected by the user (video input mode only).
    var videoFile: URL?

    // MARK: - Session configuration (live framework type for picker bindings)

    var sessionConfiguration: PhotogrammetrySession.Configuration = PhotogrammetrySession.Configuration()
    var detailLevelOptionUnderQualityMenu: PhotogrammetrySession.Request.Detail = .medium
    var detailLevelOptionsUnderAdvancedMenu = CodableDetailLevelOptions()
    var exportFormats: Set<ModelExportFormat> = []

    // MARK: - Error surface (used by ImageFolderView, ModelFolderView, etc.)

    var alertMessage: String = ""
    var hasError: Bool = false

    // MARK: - Init

    /// Create an empty draft for a new job.
    init() {}

    /// Create a draft pre-filled from an existing job for editing.
    init(from job: ReconstructionJob) {
        imageFolder = job.imageFolder
        modelFolder = job.modelFolder
        modelName = job.modelName
        boundingBoxAvailable = job.boundingBoxAvailable

        sessionConfiguration = job.sessionConfiguration.toSessionConfiguration()
        detailLevelOptionUnderQualityMenu = job.primaryDetailLevel.toFrameworkType
        detailLevelOptionsUnderAdvancedMenu = job.additionalDetailLevels
        exportFormats = job.exportFormats
    }

    // MARK: - Conversion

    /// Build a `ReconstructionJob` from the current draft.
    /// Returns `nil` if required fields are missing.
    func toJob(existingId: UUID? = nil, createdAt: Date = Date()) -> ReconstructionJob? {
        guard let imageFolder, let modelFolder, let modelName, !modelName.isEmpty else {
            return nil
        }

        var job = ReconstructionJob(
            id: existingId ?? UUID(),
            imageFolder: imageFolder,
            modelFolder: modelFolder,
            modelName: modelName,
            sessionConfiguration: CodableSessionConfiguration(from: sessionConfiguration),
            primaryDetailLevel: CodableDetailLevel(from: detailLevelOptionUnderQualityMenu),
            additionalDetailLevels: detailLevelOptionsUnderAdvancedMenu,
            createdAt: createdAt
        )
        job.boundingBoxAvailable = boundingBoxAvailable
        job.exportFormats = exportFormats
        return job
    }

    /// Validate required fields and set the error state if anything is missing.
    func validate() -> Bool {
        if imageFolder == nil {
            if inputMode == .video {
                alertMessage = videoFile == nil
                    ? "No video file selected"
                    : "Frame extraction is not complete yet"
            } else {
                alertMessage = "Image folder is not selected"
            }
            hasError = true
            return false
        }
        if modelName == nil || modelName!.isEmpty {
            alertMessage = "Model name is not entered"
            hasError = true
            return false
        }
        if modelFolder == nil {
            alertMessage = "Output folder is not selected"
            hasError = true
            return false
        }
        return true
    }
}
