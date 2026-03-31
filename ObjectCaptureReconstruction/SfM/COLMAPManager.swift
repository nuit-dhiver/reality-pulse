/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Downloads, caches, and verifies the COLMAP binary for Apple Silicon.
*/

import Foundation
import CryptoKit
import os

private let logger = Logger(subsystem: ObjectCaptureReconstructionApp.subsystem,
                            category: "COLMAPManager")

/// Manages the lifecycle of the COLMAP binary: download, cache, verify, and locate.
@MainActor @Observable
class COLMAPManager {

    // MARK: - Configuration

    /// URL to fetch the pre-compiled COLMAP binary for Apple Silicon.
    static let downloadURL = URL(string: "https://github.com/nuit-dhiver/colmap/releases/download/Beta/colmap")!

    /// Expected SHA-256 hex digest of the downloaded archive. Set to `nil` to skip verification.
    static var expectedSHA256: String?

    // MARK: - State

    enum Status: Equatable {
        case notInstalled
        case downloading(progress: Double)
        case installed(version: String)
        case error(String)
    }

    private(set) var status: Status = .notInstalled

    // MARK: - Paths

    private static let colmapDirectoryName = "colmap"
    private static let binaryName = "colmap"

    private let storeDirectory: URL

    /// Full path to the cached COLMAP executable.
    var binaryURL: URL {
        storeDirectory.appending(path: Self.binaryName)
    }

    var isInstalled: Bool {
        FileManager.default.isExecutableFile(atPath: binaryURL.path())
    }

    // MARK: - Init

    init() {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        storeDirectory = appSupport
            .appending(path: "RealityPulse")
            .appending(path: Self.colmapDirectoryName)

        try? FileManager.default.createDirectory(at: storeDirectory, withIntermediateDirectories: true)
        refreshStatus()
    }

    // MARK: - Public API

    /// Ensure COLMAP is available, downloading if necessary.
    func ensureAvailable() async throws {
        if isInstalled {
            logger.log("COLMAP binary already installed at \(self.binaryURL.path())")
            return
        }
        try await download()
    }

    /// Download and install the COLMAP binary.
    func download() async throws {
        logger.log("Downloading COLMAP from \(Self.downloadURL)")
        status = .downloading(progress: 0)

        // Wipe the entire store directory and recreate it to ensure a clean slate.
        try? FileManager.default.removeItem(at: storeDirectory)
        try FileManager.default.createDirectory(at: storeDirectory, withIntermediateDirectories: true)

        do {
            let (tempURL, response) = try await downloadWithProgress(from: Self.downloadURL)

            defer {
                try? FileManager.default.removeItem(at: tempURL)
            }

            // Validate HTTP response.
            if let httpResponse = response as? HTTPURLResponse,
               !(200..<300).contains(httpResponse.statusCode) {
                let msg = "Download failed: HTTP \(httpResponse.statusCode)"
                logger.error("\(msg)")
                status = .error(msg)
                throw COLMAPError.invalidOutput(msg)
            }

            // Verify checksum if configured.
            if let expected = Self.expectedSHA256 {
                let actual = try sha256(of: tempURL)
                guard actual == expected.lowercased() else {
                    let msg = "SHA-256 mismatch: expected \(expected), got \(actual)"
                    logger.error("\(msg)")
                    status = .error(msg)
                    throw COLMAPError.checksumMismatch(expected: expected, actual: actual)
                }
                logger.log("SHA-256 verified.")
            }

            // Extract archive.
            try await extractArchive(at: tempURL)

            // Mark executable.
            if FileManager.default.fileExists(atPath: binaryURL.path()) {
                try FileManager.default.setAttributes(
                    [.posixPermissions: 0o755],
                    ofItemAtPath: binaryURL.path()
                )
            }

            // Verify the binary actually exists and is executable.
            guard isInstalled else {
                let msg = "COLMAP binary not found after extraction. The download URL may be incorrect."
                logger.error("\(msg)")
                status = .error(msg)
                throw COLMAPError.binaryNotFound
            }

            refreshStatus()
            logger.log("COLMAP installed successfully.")
        } catch let error as COLMAPError {
            throw error
        } catch {
            let msg = "Download failed: \(error.localizedDescription)"
            logger.error("\(msg)")
            status = .error(msg)
            throw error
        }
    }

    /// Remove the cached COLMAP binary.
    func uninstall() throws {
        if FileManager.default.fileExists(atPath: storeDirectory.path()) {
            try FileManager.default.removeItem(at: storeDirectory)
            try FileManager.default.createDirectory(at: storeDirectory, withIntermediateDirectories: true)
        }
        status = .notInstalled
        logger.log("COLMAP uninstalled.")
    }

    // MARK: - Status

    func refreshStatus() {
        guard isInstalled else {
            status = .notInstalled
            return
        }

        // Try to read version.
        let versionFile = storeDirectory.appending(path: "version.txt")
        if let data = try? Data(contentsOf: versionFile),
           let version = String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines) {
            status = .installed(version: version)
        } else {
            status = .installed(version: "unknown")
        }
    }

    // MARK: - Download helpers

    private func downloadWithProgress(from url: URL) async throws -> (URL, URLResponse) {
        let request = URLRequest(url: url)
        let (asyncBytes, response) = try await URLSession.shared.bytes(for: request)

        let totalBytes = response.expectedContentLength
        var receivedBytes: Int64 = 0

        // Use a neutral temp filename (the download may be a plain binary, not an archive).
        let tempURL = FileManager.default.temporaryDirectory
            .appending(path: "colmap-download-\(UUID().uuidString)")
        FileManager.default.createFile(atPath: tempURL.path(), contents: nil)

        let fileHandle = try FileHandle(forWritingTo: tempURL)
        defer { try? fileHandle.close() }

        var buffer = Data()
        let bufferSize = 65_536

        for try await byte in asyncBytes {
            buffer.append(byte)
            receivedBytes += 1

            if buffer.count >= bufferSize {
                fileHandle.write(buffer)
                buffer.removeAll(keepingCapacity: true)

                if totalBytes > 0 {
                    let fraction = Double(receivedBytes) / Double(totalBytes)
                    await MainActor.run { status = .downloading(progress: fraction) }
                }
            }
        }

        if !buffer.isEmpty {
            fileHandle.write(buffer)
        }

        await MainActor.run { status = .downloading(progress: 1.0) }
        return (tempURL, response)
    }

    private func extractArchive(at archiveURL: URL) async throws {
        // Extract to a temporary directory first, then move the binary out.
        let extractDir = FileManager.default.temporaryDirectory
            .appending(path: "colmap-extract-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: extractDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: extractDir) }

        let process = Process()
        process.executableURL = URL(filePath: "/usr/bin/tar")
        process.arguments = ["-xzf", archiveURL.path(), "-C", extractDir.path()]
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice

        try process.run()
        process.waitUntilExit()

        if process.terminationStatus == 0 {
            // Find the colmap binary in the extracted contents.
            if let foundBinary = findBinary(named: Self.binaryName, in: extractDir) {
                try? FileManager.default.removeItem(at: binaryURL)
                try FileManager.default.moveItem(at: foundBinary, to: binaryURL)
            } else {
                throw COLMAPError.invalidOutput("Archive extracted but no '\(Self.binaryName)' binary found inside.")
            }
        } else {
            // Not a tar archive — treat the raw download as the binary itself.
            try? FileManager.default.removeItem(at: binaryURL)
            try FileManager.default.copyItem(at: archiveURL, to: binaryURL)
        }

        // Try to extract version info.
        await saveVersionInfo()
    }

    /// Recursively search for a binary by name in a directory.
    private func findBinary(named name: String, in directory: URL) -> URL? {
        guard let enumerator = FileManager.default.enumerator(
            at: directory,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else { return nil }

        for case let fileURL as URL in enumerator {
            if fileURL.lastPathComponent == name {
                let isFile = (try? fileURL.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) ?? false
                if isFile { return fileURL }
            }
        }
        return nil
    }

    private func saveVersionInfo() async {
        let process = Process()
        process.executableURL = binaryURL
        process.arguments = ["--version"]

        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = FileHandle.nullDevice

        do {
            try process.run()
            process.waitUntilExit()

            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            if let output = String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines) {
                let versionFile = storeDirectory.appending(path: "version.txt")
                try output.write(to: versionFile, atomically: true, encoding: .utf8)
            }
        } catch {
            logger.warning("Could not determine COLMAP version: \(error)")
        }
    }

    // MARK: - Checksum

    private func sha256(of fileURL: URL) throws -> String {
        let data = try Data(contentsOf: fileURL)
        let digest = SHA256.hash(data: data)
        return digest.map { String(format: "%02x", $0) }.joined()
    }
}

// MARK: - Errors

enum COLMAPError: LocalizedError {
    case checksumMismatch(expected: String, actual: String)
    case binaryNotFound
    case executionFailed(exitCode: Int32, stderr: String)
    case processTerminated
    case invalidOutput(String)

    var errorDescription: String? {
        switch self {
        case .checksumMismatch(let expected, let actual):
            return "COLMAP binary checksum mismatch (expected: \(expected), got: \(actual))"
        case .binaryNotFound:
            return "COLMAP binary not found. Please download it first."
        case .executionFailed(let code, let stderr):
            return "COLMAP exited with code \(code): \(stderr)"
        case .processTerminated:
            return "COLMAP process was terminated."
        case .invalidOutput(let msg):
            return "Invalid COLMAP output: \(msg)"
        }
    }
}
