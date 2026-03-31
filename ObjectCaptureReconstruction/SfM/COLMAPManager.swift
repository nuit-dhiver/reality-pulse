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

    /// Full path to the bundled COLMAP executable.
    var binaryURL: URL {
        Bundle.main.url(forResource: "colmap", withExtension: nil)
            ?? Bundle.main.bundleURL.appending(path: "Contents/Resources/colmap")
    }

    var isInstalled: Bool {
        FileManager.default.isExecutableFile(atPath: binaryURL.path())
    }

    // MARK: - Init

    init() {
        refreshStatus()
    }

    // MARK: - Public API

    /// Ensure the bundled COLMAP binary is available.
    func ensureAvailable() async throws {
        if isInstalled {
            logger.log("COLMAP binary available at \(self.binaryURL.path())")
            return
        }
        let msg = "Bundled COLMAP binary not found at \(binaryURL.path()). The app may be corrupted — try reinstalling."
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
        logger.log("COLMAP binary found at \(self.binaryURL.path())")
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
