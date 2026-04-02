/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Data model for maintaining the app state and the job queue.
*/

import Foundation
import RealityKit
import os

private let logger = Logger(subsystem: ObjectCaptureReconstructionApp.subsystem,
                            category: "AppDataModel")

@MainActor @Observable class AppDataModel {
    enum State: Equatable {
        case idle
        case scheduling
        case error
    }

    var state: State = .idle {
        didSet {
            logger.log("State switched to \(String(describing: self.state))")
        }
    }

    let scheduler = JobScheduler()

    var alertMessage: String = ""

    /// Whether the job-setup sheet is presented.
    var showingJobSetup = false

    /// Whether the SfM job-setup sheet is presented.
    var showingSfMJobSetup = false

    /// Whether the Gaussian Splat job-setup sheet is presented.
    var showingGaussianSplatJobSetup = false

    /// Whether the schedule-settings sheet is presented.
    var showingScheduleSettings = false

    /// Draft job being edited in the setup sheet (`nil` when adding a new job).
    var editingJob: ReconstructionJob?

    init() {
        scheduler.loadFromDisk()
    }
}

extension PhotogrammetrySession.Error: @retroactive CustomStringConvertible {
    public var description: String {
        switch self {
        case .invalidImages:
            return "No valid images found in selected folder"
        case .invalidOutput:
            return "Cannot save to selected folder"
        case .insufficientStorage:
            return "Not enough disk space available to begin processing."
        @unknown default:
            logger.warning("Unknown Error case: \(self)")
            return "\(self)"
        }
    }
}

extension PhotogrammetrySession.Configuration.CustomDetailSpecification.TextureFormat: @retroactive Hashable {
    public func hash(into hasher: inout Hasher) {
        switch self {
        case .png: break
        case .jpeg(let compressionQuality):
            hasher.combine(compressionQuality)
        @unknown default:
            fatalError("Unknown texture format: \(self)")
        }
    }
}
