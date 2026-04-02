/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Loads the Brush training bridge dylib and exposes typed Swift bindings.
*/

import Foundation
import Darwin
import os

private let logger = Logger(
    subsystem: ObjectCaptureReconstructionApp.subsystem,
    category: "GaussianSplatTrainingManager"
)

typealias BrushTrainingProgressCallback = @convention(c) (
    UnsafeRawPointer?,
    UnsafeMutableRawPointer?
) -> Void

typealias BrushTrainingRunFunction = @convention(c) (
    UnsafeRawPointer?,
    BrushTrainingProgressCallback?,
    UnsafeMutableRawPointer?
) -> UInt32

typealias BrushTrainingLastErrorFunction = @convention(c) () -> UnsafePointer<CChar>?
typealias BrushTrainingVersionFunction = @convention(c) () -> UnsafePointer<CChar>?
typealias BrushTrainingCancelFunction = @convention(c) () -> Void

struct BrushTrainingRunConfig {
    var datasetPath: UnsafePointer<CChar>?
    var outputPath: UnsafePointer<CChar>?
    var outputName: UnsafePointer<CChar>?
    var totalTrainSteps: UInt32
    var refineEvery: UInt32
    var maxResolution: UInt32
    var exportEvery: UInt32
    var evalEvery: UInt32
    var seed: UInt64
    var shDegree: UInt32
    var maxSplats: UInt32
    var lodLevels: UInt32
    var lodRefineSteps: UInt32
    var lodDecimationKeep: UInt32
    var lodImageScale: UInt32
    var lpipsLossWeight: Float
    var rerunEnabled: UInt8
}

struct BrushTrainingProgress {
    var kind: UInt32
    var iter: UInt32
    var totalIters: UInt32
    var elapsedMillis: UInt64
    var currentLOD: UInt32
    var totalLODs: UInt32
    var splatCount: UInt32
    var shDegree: UInt32
    var trainViewCount: UInt32
    var evalViewCount: UInt32
    var avgPSNR: Float
    var avgSSIM: Float
    var message: UnsafePointer<CChar>?
}

enum BrushTrainingEventKind: UInt32 {
    case processStarted = 0
    case loadingStarted = 1
    case configResolved = 2
    case datasetLoaded = 3
    case splatsUpdated = 4
    case trainStep = 5
    case refineStep = 6
    case evalResult = 7
    case loadingFinished = 8
    case warning = 9
    case done = 10
    case error = 11
}

enum GaussianSplatTrainingManagerError: LocalizedError {
    case bridgeNotFound
    case bridgeBuildFailed(String)
    case bridgeLoadFailed(String)
    case missingSymbol(String)

    var errorDescription: String? {
        switch self {
        case .bridgeNotFound:
            return "The Gaussian Splat bridge dylib was not found in the app bundle or source tree."
        case .bridgeBuildFailed(let output):
            return "Building the Gaussian Splat bridge failed.\n\n\(output)"
        case .bridgeLoadFailed(let message):
            return "Loading the Gaussian Splat bridge failed: \(message)"
        case .missingSymbol(let name):
            return "The Gaussian Splat bridge is missing required symbol \(name)."
        }
    }
}

struct GaussianSplatTrainingBridge: @unchecked Sendable {
    let handle: UnsafeMutableRawPointer
    let runFunction: BrushTrainingRunFunction
    let lastErrorFunction: BrushTrainingLastErrorFunction
    let versionFunction: BrushTrainingVersionFunction
    let requestCancelFunction: BrushTrainingCancelFunction
    let resetCancelFunction: BrushTrainingCancelFunction

    func versionString() -> String {
        guard let pointer = versionFunction() else { return "unknown" }
        return String(cString: pointer)
    }

    func lastErrorMessage() -> String {
        guard let pointer = lastErrorFunction() else { return "Unknown Gaussian Splat bridge error" }
        return String(cString: pointer)
    }

    func requestCancel() {
        requestCancelFunction()
    }

    func resetCancel() {
        resetCancelFunction()
    }

    func run(
        datasetRoot: URL,
        exportsDirectory: URL,
        configuration: CodableGaussianSplatTrainingConfiguration,
        callback: BrushTrainingProgressCallback?,
        userData: UnsafeMutableRawPointer?
    ) -> UInt32 {
        let datasetCString = strdup(datasetRoot.path())
        let exportsCString = strdup(exportsDirectory.path())
        let outputNameCString = strdup("export_{iter}.ply")

        defer {
            free(datasetCString)
            free(exportsCString)
            free(outputNameCString)
        }

        var config = BrushTrainingRunConfig(
            datasetPath: UnsafePointer(datasetCString),
            outputPath: UnsafePointer(exportsCString),
            outputName: UnsafePointer(outputNameCString),
            totalTrainSteps: UInt32(max(configuration.totalTrainSteps, 0)),
            refineEvery: UInt32(max(configuration.refineEvery, 0)),
            maxResolution: UInt32(max(configuration.maxResolution, 0)),
            exportEvery: UInt32(max(configuration.exportEvery, 0)),
            evalEvery: UInt32(max(configuration.evalEvery, 0)),
            seed: UInt64(max(configuration.seed, 0)),
            shDegree: UInt32(max(configuration.shDegree, 0)),
            maxSplats: UInt32(max(configuration.maxSplats, 0)),
            lodLevels: UInt32(max(configuration.lodLevels, 0)),
            lodRefineSteps: UInt32(max(configuration.lodRefineSteps, 0)),
            lodDecimationKeep: UInt32(max(configuration.lodDecimationKeep, 0)),
            lodImageScale: UInt32(max(configuration.lodImageScale, 0)),
            lpipsLossWeight: Float(configuration.lpipsLossWeight),
            rerunEnabled: configuration.rerunEnabled ? 1 : 0
        )

        return withUnsafePointer(to: &config) { configPointer in
            runFunction(UnsafeRawPointer(configPointer), callback, userData)
        }
    }
}

actor GaussianSplatTrainingManager {
    static let shared = GaussianSplatTrainingManager()

    private var cachedBridge: GaussianSplatTrainingBridge?

    func preparedBridge() async throws -> GaussianSplatTrainingBridge {
        if let cachedBridge {
            return cachedBridge
        }

        if let bundleDylibURL = Self.bundleDylibURL(), FileManager.default.fileExists(atPath: bundleDylibURL.path()) {
            let bridge = try loadBridge(from: bundleDylibURL)
            cachedBridge = bridge
            return bridge
        }

        if let sourceDylibURL = Self.sourceTreeDylibURL(), FileManager.default.fileExists(atPath: sourceDylibURL.path()) {
            let bridge = try loadBridge(from: sourceDylibURL)
            cachedBridge = bridge
            return bridge
        }

        try await buildSourceTreeBridge()

        guard let dylibURL = Self.sourceTreeDylibURL(), FileManager.default.fileExists(atPath: dylibURL.path()) else {
            throw GaussianSplatTrainingManagerError.bridgeNotFound
        }

        let bridge = try loadBridge(from: dylibURL)
        cachedBridge = bridge
        return bridge
    }

    private func buildSourceTreeBridge() async throws {
        let scriptURL = Self.engineRoot()
            .appending(path: "build")
            .appending(path: "macos-build.sh")

        guard FileManager.default.isExecutableFile(atPath: scriptURL.path()) else {
            throw GaussianSplatTrainingManagerError.bridgeNotFound
        }

        let output = try await Task.detached(priority: .userInitiated) {
            try Self.runProcess(
                executableURL: scriptURL,
                currentDirectoryURL: scriptURL.deletingLastPathComponent().deletingLastPathComponent()
            )
        }.value

        logger.log("Built Gaussian Splat bridge: \(output)")
    }

    private func loadBridge(from dylibURL: URL) throws -> GaussianSplatTrainingBridge {
        guard let handle = dlopen(dylibURL.path(), RTLD_NOW | RTLD_LOCAL) else {
            throw GaussianSplatTrainingManagerError.bridgeLoadFailed(String(cString: dlerror()))
        }

        let runFunction: BrushTrainingRunFunction = try loadSymbol(named: "brush_training_run", from: handle)
        let lastErrorFunction: BrushTrainingLastErrorFunction = try loadSymbol(
            named: "brush_training_last_error_message",
            from: handle
        )
        let versionFunction: BrushTrainingVersionFunction = try loadSymbol(
            named: "brush_training_bridge_version",
            from: handle
        )
        let requestCancelFunction: BrushTrainingCancelFunction = try loadSymbol(
            named: "brush_training_request_cancel",
            from: handle
        )
        let resetCancelFunction: BrushTrainingCancelFunction = try loadSymbol(
            named: "brush_training_reset_cancel",
            from: handle
        )

        let bridge = GaussianSplatTrainingBridge(
            handle: handle,
            runFunction: runFunction,
            lastErrorFunction: lastErrorFunction,
            versionFunction: versionFunction,
            requestCancelFunction: requestCancelFunction,
            resetCancelFunction: resetCancelFunction
        )

        logger.log("Loaded Gaussian Splat bridge \(bridge.versionString()) from \(dylibURL.path())")
        return bridge
    }

    private func loadSymbol<T>(named name: String, from handle: UnsafeMutableRawPointer) throws -> T {
        guard let rawSymbol = dlsym(handle, name) else {
            throw GaussianSplatTrainingManagerError.missingSymbol(name)
        }
        return unsafeBitCast(rawSymbol, to: T.self)
    }

    nonisolated private static func runProcess(
        executableURL: URL,
        currentDirectoryURL: URL
    ) throws -> String {
        let process = Process()
        let outputPipe = Pipe()
        let errorPipe = Pipe()

        process.executableURL = executableURL
        process.currentDirectoryURL = currentDirectoryURL
        process.standardOutput = outputPipe
        process.standardError = errorPipe

        try process.run()
        process.waitUntilExit()

        let stdout = String(data: outputPipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
        let stderr = String(data: errorPipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
        let combined = [stdout, stderr]
            .filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
            .joined(separator: "\n")

        guard process.terminationStatus == 0 else {
            throw GaussianSplatTrainingManagerError.bridgeBuildFailed(combined)
        }

        return combined
    }

    nonisolated private static var repositoryRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    nonisolated private static func engineRoot() -> URL {
        repositoryRoot
            .appending(path: "ObjectCaptureReconstruction")
            .appending(path: "GaussianSplat")
            .appending(path: "Engine")
    }

    nonisolated private static func sourceTreeDylibURL() -> URL? {
        engineRoot()
            .appending(path: "target")
            .appending(path: "release")
            .appending(path: "libbrush_training_bridge.dylib")
    }

    nonisolated private static func bundleDylibURL() -> URL? {
        if let frameworksURL = Bundle.main.privateFrameworksURL {
            let candidate = frameworksURL.appending(path: "libbrush_training_bridge.dylib")
            if FileManager.default.fileExists(atPath: candidate.path()) {
                return candidate
            }
        }

        if let resourcesURL = Bundle.main.resourceURL {
            let candidate = resourcesURL.appending(path: "libbrush_training_bridge.dylib")
            if FileManager.default.fileExists(atPath: candidate.path()) {
                return candidate
            }
        }

        return nil
    }
}