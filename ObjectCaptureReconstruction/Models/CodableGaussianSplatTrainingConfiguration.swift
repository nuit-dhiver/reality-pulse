/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Codable Swift-side configuration for Gaussian Splat training jobs.
*/

import Foundation

/// Persists the subset of training settings the Swift app currently exposes.
struct CodableGaussianSplatTrainingConfiguration: Codable, Equatable, Sendable {
    var totalTrainSteps: Int = 30_000
    var refineEvery: Int = 200
    var maxResolution: Int = 1_920
    var exportEvery: Int = 5_000
    var evalEvery: Int = 1_000
    var seed: Int = 42
    var shDegree: Int = 3
    var maxSplats: Int = 10_000_000
    var lodLevels: Int = 0
    var lodRefineSteps: Int = 5_000
    var lodDecimationKeep: Int = 50
    var lodImageScale: Int = 50
    var lpipsLossWeight: Double = 0
    var rerunEnabled: Bool = false

    var totalIterations: Int {
        totalTrainSteps + lodLevels * lodRefineSteps
    }
}