/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Locates and verifies the COLMAP binary bundled with the app.
*/

import Foundation
import os

private let logger = Logger(subsystem: ObjectCaptureReconstructionApp.subsystem,
                            category: "COLMAPManager")

/// Manages the lifecycle of the COLMAP binary bundled in the app's Resources.
@MainActor @Observable
class COLMAPManager {

    // MARK: - State

    enum Status: Equatable {
        case installed(version: String)
        case error(String)
    }

    private(set) var status: Status = .error("Not checked yet")

    // MARK: - Paths

    /// Full path to the bundled COLMAP executable inside colmap-bundle/.
    var binaryURL: URL {
        if let resourceURL = Bundle.main.resourceURL {
            let url = resourceURL
                .appendingPathComponent("colmap-bundle")
                .appendingPathComponent("bin")
                .appendingPathComponent("colmap")
            logger.info("COLMAP binary path: \(url.path)")
            return url
        }
        let url = Bundle.main.bundleURL
            .appendingPathComponent("Contents")
            .appendingPathComponent("Resources")
            .appendingPathComponent("colmap-bundle")
            .appendingPathComponent("bin")
            .appendingPathComponent("colmap")
        logger.info("Fallback colmap path: \(url.path)")
        return url
    }

    /// Path to the bundled dylib directory (colmap-bundle/lib/).
    var libraryDirectoryURL: URL {
        binaryURL
            .deletingLastPathComponent()  // bin/
            .deletingLastPathComponent()  // colmap-bundle/
            .appendingPathComponent("lib")
    }

    var isInstalled: Bool {
        let path = binaryURL.path
        let exists = FileManager.default.fileExists(atPath: path)
        let executable = FileManager.default.isExecutableFile(atPath: path)
        logger.info("COLMAP check — path: \(path), exists: \(exists), executable: \(executable)")
        return executable
    }

    // MARK: - Init

    init() {
        refreshStatus()
    }

    // MARK: - Public API

    /// Ensure the bundled COLMAP binary is available.
    func ensureAvailable() async throws {
        let url = binaryURL
        logger.info("ensureAvailable — binaryURL: \(url.path), isInstalled: \(self.isInstalled)")
        logger.info("Bundle.main.bundlePath: \(Bundle.main.bundlePath)")
        if isInstalled {
            return
        }
        let msg = "Bundled COLMAP binary not found at \(url.path). The app may be corrupted — try reinstalling."
        logger.error("\(msg)")
        status = .error(msg)
        throw COLMAPError.binaryNotFound
    }

    // MARK: - Status

    func refreshStatus() {
        guard isInstalled else {
            status = .error("COLMAP binary not found in app bundle.")
            return
        }
        status = .installed(version: "bundled")
        logger.log("COLMAP binary found at \(self.binaryURL.path)")
    }
}

// MARK: - Errors

enum COLMAPError: LocalizedError {
    case binaryNotFound
    case executionFailed(exitCode: Int32, stderr: String)
    case processTerminated
    case invalidOutput(String)

    var errorDescription: String? {
        switch self {
        case .binaryNotFound:
            return "COLMAP binary not found in app bundle."
        case .executionFailed(let code, let stderr):
            return "COLMAP exited with code \(code): \(stderr)"
        case .processTerminated:
            return "COLMAP process was terminated."
        case .invalidOutput(let msg):
            return "Invalid COLMAP output: \(msg)"
        }
    }
}
