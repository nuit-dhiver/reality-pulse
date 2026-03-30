/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Tracks and reports COLMAP SfM progress by parsing subprocess output.
*/

import Foundation
import os

private let logger = Logger(subsystem: ObjectCaptureReconstructionApp.subsystem,
                            category: "SfMProgressTracker")

/// Aggregates progress updates from `COLMAPRunner` and provides a unified
/// progress value (0.0–1.0) across all SfM phases.
@MainActor @Observable
class SfMProgressTracker {

    var currentPhase: COLMAPRunner.Phase = .featureExtraction
    var phaseProgress: Double = 0
    var overallProgress: Double = 0
    var statusMessage: String = "Preparing..."

    /// Weight of each phase in the overall progress.
    private static let phaseWeights: [COLMAPRunner.Phase: Double] = [
        .featureExtraction: 0.30,
        .featureMatching: 0.30,
        .sparseReconstruction: 0.40,
        .complete: 0.0
    ]

    /// Cumulative progress offset for each phase.
    private static let phaseOffsets: [COLMAPRunner.Phase: Double] = {
        var offsets: [COLMAPRunner.Phase: Double] = [:]
        var cumulative: Double = 0
        for phase in COLMAPRunner.Phase.allCases {
            offsets[phase] = cumulative
            cumulative += phaseWeights[phase] ?? 0
        }
        return offsets
    }()

    /// Handle a progress update from COLMAPRunner.
    func update(_ progress: COLMAPRunner.ProgressUpdate) {
        currentPhase = progress.phase
        phaseProgress = progress.fraction

        let offset = Self.phaseOffsets[progress.phase] ?? 0
        let weight = Self.phaseWeights[progress.phase] ?? 0
        overallProgress = min(offset + weight * progress.fraction, 1.0)

        if !progress.message.isEmpty {
            statusMessage = progress.message
        }

        logger.debug("SfM progress: \(self.currentPhase.rawValue) \(Int(self.phaseProgress * 100))% (overall \(Int(self.overallProgress * 100))%)")
    }

    /// Reset tracker for a new SfM run.
    func reset() {
        currentPhase = .featureExtraction
        phaseProgress = 0
        overallProgress = 0
        statusMessage = "Preparing..."
    }
}
