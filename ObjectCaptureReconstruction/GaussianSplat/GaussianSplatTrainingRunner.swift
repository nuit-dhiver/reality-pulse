/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Runs Gaussian Splat training through the dynamically loaded Rust bridge.
*/

import Foundation
import os

private let logger = Logger(
    subsystem: ObjectCaptureReconstructionApp.subsystem,
    category: "GaussianSplatTrainingRunner"
)

struct GaussianSplatTrainingProgressUpdate: Sendable {
    var fraction: Double
    var phase: String
    var detail: String?
    var splatCount: Int
    var currentLOD: Int
    var totalLODs: Int
    var avgPSNR: Double?
    var avgSSIM: Double?
}

struct GaussianSplatTrainingRunResult: Sendable {
    var exportedPLYCount: Int
    var totalIterations: Int
}

enum GaussianSplatTrainingRunnerError: LocalizedError {
    case bridgeFailed(String)

    var errorDescription: String? {
        switch self {
        case .bridgeFailed(let message):
            return message
        }
    }
}

private final class GaussianSplatTrainingCallbackBox: @unchecked Sendable {
    let handler: @Sendable (GaussianSplatTrainingProgressUpdate) async -> Void

    init(handler: @escaping @Sendable (GaussianSplatTrainingProgressUpdate) async -> Void) {
        self.handler = handler
    }

    func handle(progress: BrushTrainingProgress) {
        guard let kind = BrushTrainingEventKind(rawValue: progress.kind) else { return }

        let phase = Self.phaseText(for: kind, progress: progress)
        let detail = progress.message.map { String(cString: $0) }
        let fraction: Double

        switch kind {
        case .done:
            fraction = 1
        case .trainStep, .refineStep, .evalResult, .splatsUpdated:
            if progress.totalIters > 0 {
                fraction = min(Double(progress.iter) / Double(progress.totalIters), 1)
            } else {
                fraction = 0
            }
        default:
            fraction = 0
        }

        let update = GaussianSplatTrainingProgressUpdate(
            fraction: fraction,
            phase: phase,
            detail: detail,
            splatCount: Int(progress.splatCount),
            currentLOD: Int(progress.currentLOD),
            totalLODs: Int(progress.totalLODs),
            avgPSNR: progress.avgPSNR > 0 ? Double(progress.avgPSNR) : nil,
            avgSSIM: progress.avgSSIM > 0 ? Double(progress.avgSSIM) : nil
        )

        Task {
            await handler(update)
        }
    }

    private static func phaseText(for kind: BrushTrainingEventKind, progress: BrushTrainingProgress) -> String {
        switch kind {
        case .processStarted:
            return "Starting"
        case .loadingStarted:
            return "Loading Dataset"
        case .configResolved:
            return "Resolved Configuration"
        case .datasetLoaded:
            return "Dataset Loaded"
        case .splatsUpdated:
            return "Updated Splats"
        case .trainStep:
            if progress.currentLOD > 0 {
                return "Training LOD \(progress.currentLOD)/\(max(progress.totalLODs, 1))"
            }
            return "Training"
        case .refineStep:
            return "Refining"
        case .evalResult:
            return "Evaluating"
        case .loadingFinished:
            return "Loading Complete"
        case .warning:
            return "Warning"
        case .done:
            return "Complete"
        case .error:
            return "Failed"
        }
    }
}

private let gaussianSplatTrainingCallback: BrushTrainingProgressCallback = { progressPointer, userData in
    guard let progressPointer, let userData else { return }
    let callbackBox = Unmanaged<GaussianSplatTrainingCallbackBox>
        .fromOpaque(userData)
        .takeUnretainedValue()
    let progress = progressPointer
        .assumingMemoryBound(to: BrushTrainingProgress.self)
        .pointee
    callbackBox.handle(progress: progress)
}

actor GaussianSplatTrainingRunner {
    private var activeBridge: GaussianSplatTrainingBridge?
    private var cancellationRequested = false

    func run(
        datasetRoot: URL,
        exportsDirectory: URL,
        configuration: CodableGaussianSplatTrainingConfiguration,
        progressHandler: @escaping @Sendable (GaussianSplatTrainingProgressUpdate) async -> Void
    ) async throws -> GaussianSplatTrainingRunResult {
        cancellationRequested = false

        let bridge = try await GaussianSplatTrainingManager.shared.preparedBridge()
        bridge.resetCancel()
        activeBridge = bridge

        let callbackBox = GaussianSplatTrainingCallbackBox(handler: progressHandler)
        let retainedCallbackBox = Unmanaged.passRetained(callbackBox)
        let userData = retainedCallbackBox.toOpaque()

        defer {
            retainedCallbackBox.release()
            activeBridge = nil
        }

        let exitCode = await Task.detached(priority: .userInitiated) {
            bridge.run(
                datasetRoot: datasetRoot,
                exportsDirectory: exportsDirectory,
                configuration: configuration,
                callback: gaussianSplatTrainingCallback,
                userData: userData
            )
        }.value

        if cancellationRequested || Task.isCancelled {
            throw CancellationError()
        }

        guard exitCode == 0 else {
            let message = bridge.lastErrorMessage()
            if message.localizedCaseInsensitiveContains("cancel") {
                throw CancellationError()
            }
            throw GaussianSplatTrainingRunnerError.bridgeFailed(message)
        }

        let exportedPLYCount = countPLYFiles(in: exportsDirectory)
        logger.log("Gaussian Splat training complete with \(exportedPLYCount) PLY export(s)")

        return GaussianSplatTrainingRunResult(
            exportedPLYCount: exportedPLYCount,
            totalIterations: configuration.totalIterations
        )
    }

    func cancel() {
        cancellationRequested = true
        activeBridge?.requestCancel()
    }

    private func countPLYFiles(in directory: URL) -> Int {
        guard let enumerator = FileManager.default.enumerator(at: directory, includingPropertiesForKeys: nil) else {
            return 0
        }

        var count = 0
        for case let fileURL as URL in enumerator where fileURL.pathExtension.lowercased() == "ply" {
            count += 1
        }
        return count
    }
}