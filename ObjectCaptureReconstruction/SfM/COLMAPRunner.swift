/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Orchestrates COLMAP subprocess execution for Structure from Motion processing.
*/

import Foundation
import os

private let logger = Logger(subsystem: ObjectCaptureReconstructionApp.subsystem,
                            category: "COLMAPRunner")

/// Orchestrates a complete COLMAP SfM pipeline by running subcommands sequentially.
actor COLMAPRunner {

    /// Phases of the SfM pipeline, reported for progress tracking.
    enum Phase: String, CaseIterable, Sendable {
        case featureExtraction = "Feature Extraction"
        case featureMatching = "Feature Matching"
        case sparseReconstruction = "Sparse Reconstruction"
        case complete = "Complete"
    }

    /// Progress update emitted during processing.
    struct ProgressUpdate: Sendable {
        let phase: Phase
        let fraction: Double
        let message: String
    }

    private let colmapBinaryURL: URL
    private var currentProcess: Process?
    private var isCancelled = false

    init(colmapBinaryURL: URL) {
        self.colmapBinaryURL = colmapBinaryURL
    }

    // MARK: - Public API

    /// Run the full SfM pipeline: feature extraction → matching → mapping.
    /// Returns the path to the sparse reconstruction output directory.
    func run(
        configuration: SfMConfiguration,
        imageFolder: URL,
        outputFolder: URL,
        progressHandler: @Sendable @escaping (ProgressUpdate) async -> Void
    ) async throws -> URL {
        isCancelled = false

        // Ensure the full output directory tree exists.
        try FileManager.default.createDirectory(
            at: outputFolder,
            withIntermediateDirectories: true
        )

        let outputExists = FileManager.default.fileExists(atPath: outputFolder.path)
        logger.info("Output folder: \(outputFolder.path), exists: \(outputExists)")

        let databasePath = outputFolder.appending(path: "database.db").path
        let sparsePath = outputFolder.appending(path: "sparse")

        try FileManager.default.createDirectory(
            at: sparsePath,
            withIntermediateDirectories: true
        )

        // Phase 1: Feature Extraction
        await progressHandler(ProgressUpdate(
            phase: .featureExtraction, fraction: 0, message: "Starting feature extraction..."
        ))

        try await runCOLMAP(
            args: configuration.featureExtractorArgs(
                databasePath: databasePath,
                imagePath: imageFolder.path
            ),
            phase: .featureExtraction,
            progressHandler: progressHandler
        )

        try checkCancellation()

        // Phase 2: Feature Matching
        await progressHandler(ProgressUpdate(
            phase: .featureMatching, fraction: 0, message: "Starting feature matching..."
        ))

        try await runCOLMAP(
            args: configuration.matcherArgs(databasePath: databasePath),
            phase: .featureMatching,
            progressHandler: progressHandler
        )

        try checkCancellation()

        // Phase 3: Sparse Reconstruction (Mapper)
        await progressHandler(ProgressUpdate(
            phase: .sparseReconstruction, fraction: 0, message: "Starting sparse reconstruction..."
        ))

        try await runCOLMAP(
            args: configuration.mapperArgs(
                databasePath: databasePath,
                imagePath: imageFolder.path,
                outputPath: sparsePath.path
            ),
            phase: .sparseReconstruction,
            progressHandler: progressHandler
        )

        await progressHandler(ProgressUpdate(
            phase: .complete, fraction: 1.0, message: "SfM pipeline complete"
        ))

        // COLMAP mapper creates numbered subdirectories (0, 1, 2...) for each model.
        // Return the first (largest) reconstruction.
        let sparseSubdirs = try FileManager.default.contentsOfDirectory(
            at: sparsePath,
            includingPropertiesForKeys: nil
        ).filter { $0.hasDirectoryPath }.sorted { $0.lastPathComponent < $1.lastPathComponent }

        guard let firstModel = sparseSubdirs.first else {
            throw COLMAPError.invalidOutput("No sparse reconstruction models found in output directory.")
        }

        logger.log("SfM complete. Primary model at: \(firstModel.path)")
        return firstModel
    }

    /// Cancel the current COLMAP process.
    func cancel() {
        isCancelled = true
        currentProcess?.terminate()
        currentProcess = nil
        logger.log("COLMAP process cancelled.")
    }

    // MARK: - Private

    private func runCOLMAP(
        args: [String],
        phase: Phase,
        progressHandler: @Sendable @escaping (ProgressUpdate) async -> Void
    ) async throws {
        let process = Process()
        process.executableURL = colmapBinaryURL
        process.arguments = args

        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe

        currentProcess = process

        logger.log("Running: colmap \(args.joined(separator: " "))")

        guard FileManager.default.isExecutableFile(atPath: colmapBinaryURL.path) else {
            throw COLMAPError.binaryNotFound
        }

        try process.run()

        // Read stdout asynchronously for progress parsing.
        let progressTask = Task {
            await parseProgress(
                from: stdoutPipe.fileHandleForReading,
                phase: phase,
                handler: progressHandler
            )
        }

        // Wait for process to finish.
        process.waitUntilExit()
        currentProcess = nil

        progressTask.cancel()

        let exitCode = process.terminationStatus
        if exitCode != 0 {
            let stderrData = stderrPipe.fileHandleForReading.readDataToEndOfFile()
            let stderrString = String(data: stderrData, encoding: .utf8) ?? "Unknown error"
            logger.error("COLMAP exited with code \(exitCode): \(stderrString)")

            if process.terminationReason == .uncaughtSignal {
                throw COLMAPError.processTerminated
            }
            throw COLMAPError.executionFailed(exitCode: exitCode, stderr: stderrString)
        }

        // Report phase completion.
        await progressHandler(ProgressUpdate(phase: phase, fraction: 1.0, message: "\(phase.rawValue) complete"))
    }

    private func parseProgress(
        from fileHandle: FileHandle,
        phase: Phase,
        handler: @Sendable @escaping (ProgressUpdate) async -> Void
    ) async {
        var buffer = Data()

        while !Task.isCancelled {
            let chunk = fileHandle.availableData
            if chunk.isEmpty { break }
            buffer.append(chunk)

            // Process complete lines.
            while let newlineIndex = buffer.firstIndex(of: UInt8(ascii: "\n")) {
                let lineData = buffer[buffer.startIndex..<newlineIndex]
                buffer = Data(buffer[buffer.index(after: newlineIndex)...])

                guard let line = String(data: lineData, encoding: .utf8)?.trimmingCharacters(in: .whitespaces) else {
                    continue
                }

                let fraction = Self.estimateFraction(from: line, phase: phase)
                await handler(ProgressUpdate(phase: phase, fraction: fraction, message: line))
            }
        }
    }

    /// Heuristic progress estimation from COLMAP log output.
    private static func estimateFraction(from line: String, phase: Phase) -> Double {
        // COLMAP often outputs lines like "Processing image X / Y"
        if let match = line.range(of: #"(\d+)\s*/\s*(\d+)"#, options: .regularExpression) {
            let numbers = line[match].components(separatedBy: "/").compactMap { Double($0.trimmingCharacters(in: .whitespaces)) }
            if numbers.count == 2, numbers[1] > 0 {
                return min(numbers[0] / numbers[1], 1.0)
            }
        }

        // Percentage-based output.
        if let match = line.range(of: #"(\d+(?:\.\d+)?)\s*%"#, options: .regularExpression) {
            let pctStr = line[match].replacingOccurrences(of: "%", with: "").trimmingCharacters(in: .whitespaces)
            if let pct = Double(pctStr) {
                return min(pct / 100.0, 1.0)
            }
        }

        return 0
    }

    private func checkCancellation() throws {
        if isCancelled {
            throw CancellationError()
        }
    }
}
