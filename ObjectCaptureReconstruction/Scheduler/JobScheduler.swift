/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Sequential job scheduler that processes reconstruction jobs one at a time,
respecting optional time-window constraints.
*/

import Foundation
import RealityKit
import UserNotifications
import os

private let logger = Logger(subsystem: ObjectCaptureReconstructionApp.subsystem,
                            category: "JobScheduler")

/// Processes reconstruction jobs sequentially. Before each job, the scheduler
/// checks the configured time window and sleeps until it opens if necessary.
@MainActor @Observable
class JobScheduler {

    // MARK: - Published state

    private(set) var jobs: [ReconstructionJob] = []
    private(set) var sfmJobs: [SfMJob] = []
    private(set) var gaussianSplatJobs: [GaussianSplatTrainingJob] = []
    var scheduleConfig: ScheduleConfig = ScheduleConfig()

    private(set) var isRunning = false
    private(set) var isPaused = false
    private(set) var currentJobId: UUID?
    private(set) var currentProgress: Double = 0
    private(set) var estimatedTimeRemaining: TimeInterval?
    private(set) var currentSfMPhase: String = ""
    private(set) var currentTrainingPhase: String = ""

    // MARK: - Internal

    private var processingTask: Task<Void, Never>?
    private var currentSession: PhotogrammetrySession?
    private var currentCOLMAPRunner: COLMAPRunner?
    private var currentTrainingRunner: GaussianSplatTrainingRunner?
    let colmapManager = COLMAPManager()
    private let store = JobStore()

    // MARK: - Persistence helpers

    func persist() {
        store.saveJobs(jobs)
        store.saveSfMJobs(sfmJobs)
        store.saveGaussianSplatJobs(gaussianSplatJobs)
        store.saveSchedule(scheduleConfig)
    }

    func loadFromDisk() {
        jobs = store.loadJobs()
        sfmJobs = store.loadSfMJobs()
        gaussianSplatJobs = store.loadGaussianSplatJobs()
        scheduleConfig = store.loadSchedule()
    }

    // MARK: - Queue management

    func addJob(_ job: ReconstructionJob) {
        jobs.append(job)
        persist()
    }

    func removeJob(_ job: ReconstructionJob) {
        jobs.removeAll { $0.id == job.id }
        persist()
    }

    func removeJobs(at offsets: IndexSet) {
        jobs.remove(atOffsets: offsets)
        persist()
    }

    func moveJob(from source: IndexSet, to destination: Int) {
        jobs.move(fromOffsets: source, toOffset: destination)
        persist()
    }

    func retryJob(_ job: ReconstructionJob) {
        guard let index = jobs.firstIndex(where: { $0.id == job.id }) else { return }
        jobs[index].status = .pending
        jobs[index].progress = 0
        jobs[index].errorMessage = nil
        persist()
    }

    func updateJob(_ job: ReconstructionJob) {
        guard let index = jobs.firstIndex(where: { $0.id == job.id }) else { return }
        jobs[index] = job
        persist()
    }

    var pendingJobCount: Int {
        jobs.filter { $0.status == .pending }.count
            + sfmJobs.filter { $0.status == .pending }.count
            + gaussianSplatJobs.filter { $0.status == .pending }.count
    }

    var completedJobCount: Int {
        jobs.filter { $0.status == .completed }.count
            + sfmJobs.filter { $0.status == .completed }.count
            + gaussianSplatJobs.filter { $0.status == .completed }.count
    }

    var totalJobCount: Int {
        jobs.count + sfmJobs.count + gaussianSplatJobs.count
    }

    // MARK: - SfM Queue management

    func addSfMJob(_ job: SfMJob) {
        sfmJobs.append(job)
        persist()
    }

    func removeSfMJob(_ job: SfMJob) {
        sfmJobs.removeAll { $0.id == job.id }
        persist()
    }

    func retrySfMJob(_ job: SfMJob) {
        guard let index = sfmJobs.firstIndex(where: { $0.id == job.id }) else { return }
        sfmJobs[index].status = .pending
        sfmJobs[index].progress = 0
        sfmJobs[index].errorMessage = nil
        sfmJobs[index].currentPhase = ""
        persist()
    }

    // MARK: - Gaussian Splat Queue management

    func addGaussianSplatJob(_ job: GaussianSplatTrainingJob) {
        gaussianSplatJobs.append(job)
        persist()
    }

    func removeGaussianSplatJob(_ job: GaussianSplatTrainingJob) {
        gaussianSplatJobs.removeAll { $0.id == job.id }
        persist()
    }

    func retryGaussianSplatJob(_ job: GaussianSplatTrainingJob) {
        guard let index = gaussianSplatJobs.firstIndex(where: { $0.id == job.id }) else { return }
        gaussianSplatJobs[index].status = .pending
        gaussianSplatJobs[index].progress = 0
        gaussianSplatJobs[index].currentPhase = ""
        gaussianSplatJobs[index].errorMessage = nil
        persist()
    }

    // MARK: - Scheduler control

    func start() {
        guard !isRunning else { return }
        isRunning = true
        isPaused = false
        logger.log("Scheduler started.")
        processingTask = Task { await processQueue() }
    }

    /// Whether any pending job requires COLMAP.
    var hasPendingSfMWork: Bool {
        let hasSfmJobs = sfmJobs.contains { $0.status == .pending }
        let hasReconWithSfM = jobs.contains { $0.status == .pending && $0.runSfMFirst }
        let hasTrainingWithSfM = gaussianSplatJobs.contains {
            $0.status == .pending && $0.inputMode == .runCOLMAPInApp
        }
        return hasSfmJobs || hasReconWithSfM || hasTrainingWithSfM
    }

    func pause() {
        isPaused = true
        logger.log("Scheduler paused.")
    }

    func resume() {
        guard isPaused else { return }
        isPaused = false
        logger.log("Scheduler resumed.")
    }

    func cancel() {
        logger.log("Scheduler cancelled.")
        processingTask?.cancel()
        currentSession?.cancel()
        currentSession = nil

        // Cancel any running COLMAP process.
        if let runner = currentCOLMAPRunner {
            Task { await runner.cancel() }
            currentCOLMAPRunner = nil
        }
        if let runner = currentTrainingRunner {
            Task { await runner.cancel() }
            currentTrainingRunner = nil
        }

        isRunning = false
        isPaused = false

        // Mark the running job as cancelled.
        if let id = currentJobId {
            if let idx = jobs.firstIndex(where: { $0.id == id }) {
                jobs[idx].status = .cancelled
            }
            if let idx = sfmJobs.firstIndex(where: { $0.id == id }) {
                sfmJobs[idx].status = .cancelled
            }
            if let idx = gaussianSplatJobs.firstIndex(where: { $0.id == id }) {
                gaussianSplatJobs[idx].status = .cancelled
            }
        }

        currentJobId = nil
        currentProgress = 0
        estimatedTimeRemaining = nil
        currentSfMPhase = ""
        currentTrainingPhase = ""
        persist()
    }

    // MARK: - Processing loop

    private func processQueue() async {
        defer {
            isRunning = false
            currentJobId = nil
            currentProgress = 0
            estimatedTimeRemaining = nil
            currentSfMPhase = ""
            currentTrainingPhase = ""

            let succeeded = jobs.filter { $0.status == .completed }.count
                + sfmJobs.filter { $0.status == .completed }.count
                + gaussianSplatJobs.filter { $0.status == .completed }.count
            let failed = jobs.filter { $0.status == .failed }.count
                + sfmJobs.filter { $0.status == .failed }.count
                + gaussianSplatJobs.filter { $0.status == .failed }.count
            sendNotification(
                title: "Queue Complete",
                body: "\(succeeded) succeeded, \(failed) failed"
            )

            persist()
            logger.log("Scheduler finished.")
        }

        while !Task.isCancelled {
            // Wait while paused.
            while isPaused && !Task.isCancelled {
                try? await Task.sleep(for: .seconds(1))
            }
            if Task.isCancelled { break }

            // Wait for the allowed time window.
            if !scheduleConfig.isWithinAllowedWindow() {
                if let nextOpen = scheduleConfig.nextWindowOpen() {
                    let delay = nextOpen.timeIntervalSinceNow
                    if delay > 0 {
                        logger.log("Outside time window. Sleeping \(Int(delay))s until \(nextOpen).")
                        try? await Task.sleep(for: .seconds(delay))
                        continue
                    }
                }
            }
            if Task.isCancelled { break }

            // Pick the next pending job from either queue (SfM jobs first, then reconstruction).
            if let sfmIndex = sfmJobs.firstIndex(where: { $0.status == .pending }) {
                await processSfMJob(at: sfmIndex)
            } else if let trainingIndex = gaussianSplatJobs.firstIndex(where: { $0.status == .pending }) {
                await processGaussianSplatJob(at: trainingIndex)
            } else if let reconIndex = jobs.firstIndex(where: { $0.status == .pending }) {
                await processJob(at: reconIndex)
            } else {
                logger.log("No more pending jobs.")
                break
            }
        }
    }

    private func processJob(at index: Int) async {
        let jobId = jobs[index].id
        currentJobId = jobId
        currentProgress = 0
        estimatedTimeRemaining = nil

        jobs[index].status = .running
        jobs[index].progress = 0
        persist()

        let jobName = jobs[index].modelName
        logger.log("Starting job: \(jobName) (\(jobId))")

        // Resolve bookmarks for sandbox access.
        let (imageURL, modelURL) = jobs[index].resolveBookmarks()

        guard let imageURL, let modelURL else {
            jobs[index].status = .failed
            jobs[index].errorMessage = "Cannot access saved folders. Please remove and re-add the job."
            sendNotification(title: "Job Failed", body: jobName)
            persist()
            return
        }

        let imageAccess = imageURL.startAccessingSecurityScopedResource()
        let modelAccess = modelURL.startAccessingSecurityScopedResource()

        defer {
            if imageAccess { imageURL.stopAccessingSecurityScopedResource() }
            if modelAccess { modelURL.stopAccessingSecurityScopedResource() }
        }

        let config = jobs[index].sessionConfiguration.toSessionConfiguration()
        let requests = jobs[index].createReconstructionRequests()

        guard !requests.isEmpty else {
            jobs[index].status = .failed
            jobs[index].errorMessage = "No detail levels selected."
            sendNotification(title: "Job Failed", body: jobName)
            persist()
            return
        }

        do {
            // Run SfM pre-step if enabled.
            if jobs[index].runSfMFirst, let sfmConfig = jobs[index].sfmConfiguration {
                logger.log("Running SfM pre-step for job \(jobId)")
                currentSfMPhase = "SfM: Preparing"

                try await colmapManager.ensureAvailable()

                let sfmOutputDir = jobs[index].modelFolder.appending(path: "\(jobName)-sfm")
                let runner = COLMAPRunner(colmapBinaryURL: colmapManager.binaryURL,
                                         libraryDirectoryURL: colmapManager.libraryDirectoryURL)
                currentCOLMAPRunner = runner

                let sparseDir = try await runner.run(
                    configuration: sfmConfig.toSfMConfiguration(),
                    imageFolder: jobs[index].imageFolder,
                    outputFolder: sfmOutputDir
                ) { [weak self] update in
                    await MainActor.run {
                        self?.currentSfMPhase = "SfM: \(update.phase.rawValue)"
                    }
                }

                currentCOLMAPRunner = nil
                currentSfMPhase = ""

                // Parse and re-export results to the model folder.
                let result = try COLMAPBinaryParser.parse(directory: sparseDir)
                let exportDir = jobs[index].modelFolder.appending(path: "\(jobName)-colmap")
                try COLMAPBinaryExporter.export(result, to: exportDir)

                logger.log("SfM pre-step complete: \(result.imageCount) images, \(result.pointCount) points")
            }

            let session = try await createSession(
                imageFolder: jobs[index].imageFolder,
                configuration: config
            )
            currentSession = session
            try session.process(requests: requests)

            // Consume session outputs.
            for try await output in session.outputs {
                if Task.isCancelled { break }
                // Re-check time window between outputs.
                if !scheduleConfig.isWithinAllowedWindow() {
                    logger.log("Time window closed during processing. Pausing until window reopens.")
                    while !scheduleConfig.isWithinAllowedWindow() && !Task.isCancelled {
                        try await Task.sleep(for: .seconds(30))
                    }
                }

                switch output {
                case .requestProgress(_, let fraction):
                    currentProgress = fraction
                    if let idx = jobs.firstIndex(where: { $0.id == jobId }) {
                        jobs[idx].progress = fraction
                    }

                case .requestProgressInfo(_, let info):
                    estimatedTimeRemaining = info.estimatedRemainingTime

                case .requestComplete(_, _):
                    logger.log("Request completed for job \(jobId).")

                case .requestError(_, let error):
                    logger.warning("Request error for job \(jobId): \(error)")

                case .processingComplete:
                    logger.log("Processing complete for job \(jobId).")

                default:
                    continue
                }
            }

            currentSession = nil

            if Task.isCancelled {
                if let idx = jobs.firstIndex(where: { $0.id == jobId }) {
                    jobs[idx].status = .cancelled
                }
            } else if let idx = jobs.firstIndex(where: { $0.id == jobId }) {
                jobs[idx].status = .completed
                jobs[idx].progress = 1.0
            }

        } catch {
            logger.warning("Job \(jobId) failed: \(error)")
            currentSession = nil
            if let idx = jobs.firstIndex(where: { $0.id == jobId }) {
                jobs[idx].status = .failed
                jobs[idx].errorMessage = "\(error)"
            }
            sendNotification(title: "Job Failed", body: jobName)
        }

        persist()
    }

    // MARK: - SfM Job Processing

    private func processGaussianSplatJob(at index: Int) async {
        let jobId = gaussianSplatJobs[index].id
        currentJobId = jobId
        currentProgress = 0
        estimatedTimeRemaining = nil
        currentTrainingPhase = ""

        gaussianSplatJobs[index].status = .running
        gaussianSplatJobs[index].progress = 0
        gaussianSplatJobs[index].currentPhase = "Preparing"
        persist()

        let jobName = gaussianSplatJobs[index].jobName
        let inputMode = gaussianSplatJobs[index].inputMode
        let trainingConfiguration = gaussianSplatJobs[index].trainingConfiguration
        let sfmConfiguration = gaussianSplatJobs[index].sfmConfiguration.toSfMConfiguration()
        logger.log("Starting Gaussian Splat job: \(jobName) (\(jobId))")

        let (imageURL, outputURL, importedCOLMAPURL) = gaussianSplatJobs[index].resolveBookmarks()

        guard let imageURL, let outputURL else {
            gaussianSplatJobs[index].status = .failed
            gaussianSplatJobs[index].errorMessage = "Cannot access saved folders. Please remove and re-add the job."
            gaussianSplatJobs[index].currentPhase = ""
            sendNotification(title: "Gaussian Splat Job Failed", body: jobName)
            persist()
            return
        }

        if inputMode == .useExistingCOLMAP && importedCOLMAPURL == nil {
            gaussianSplatJobs[index].status = .failed
            gaussianSplatJobs[index].errorMessage = "Cannot access saved COLMAP folder. Please remove and re-add the job."
            gaussianSplatJobs[index].currentPhase = ""
            sendNotification(title: "Gaussian Splat Job Failed", body: jobName)
            persist()
            return
        }

        let imageAccess = imageURL.startAccessingSecurityScopedResource()
        let outputAccess = outputURL.startAccessingSecurityScopedResource()
        let importedCOLMAPAccess = importedCOLMAPURL?.startAccessingSecurityScopedResource() ?? false

        defer {
            if imageAccess { imageURL.stopAccessingSecurityScopedResource() }
            if outputAccess { outputURL.stopAccessingSecurityScopedResource() }
            if importedCOLMAPAccess, let importedCOLMAPURL {
                importedCOLMAPURL.stopAccessingSecurityScopedResource()
            }
        }

        do {
            let trainingOutputDirectory = gaussianSplatJobs[index].trainingOutputDirectory
            if FileManager.default.fileExists(atPath: trainingOutputDirectory.path()) {
                try FileManager.default.removeItem(at: trainingOutputDirectory)
            }
            try FileManager.default.createDirectory(at: trainingOutputDirectory, withIntermediateDirectories: true)

            let colmapModelDirectory: URL
            if inputMode == .runCOLMAPInApp {
                currentSfMPhase = "SfM: Preparing"
                gaussianSplatJobs[index].currentPhase = currentSfMPhase

                try await colmapManager.ensureAvailable()

                let sfmOutputDirectory = trainingOutputDirectory.appending(path: "colmap-work")
                let runner = COLMAPRunner(
                    colmapBinaryURL: colmapManager.binaryURL,
                    libraryDirectoryURL: colmapManager.libraryDirectoryURL
                )
                currentCOLMAPRunner = runner

                colmapModelDirectory = try await runner.run(
                    configuration: sfmConfiguration,
                    imageFolder: gaussianSplatJobs[index].imageFolder,
                    outputFolder: sfmOutputDirectory
                ) { [weak self] update in
                    await MainActor.run {
                        guard let self else { return }
                        let overallProgress = update.fraction * 0.25
                        self.currentSfMPhase = "SfM: \(update.phase.rawValue)"
                        self.currentProgress = overallProgress
                        if let currentIndex = self.gaussianSplatJobs.firstIndex(where: { $0.id == jobId }) {
                            self.gaussianSplatJobs[currentIndex].progress = overallProgress
                            self.gaussianSplatJobs[currentIndex].currentPhase = self.currentSfMPhase
                        }
                    }
                }

                currentCOLMAPRunner = nil
                currentSfMPhase = ""
            } else {
                colmapModelDirectory = importedCOLMAPURL ?? outputURL
            }

            currentTrainingPhase = "Preparing Dataset"
            gaussianSplatJobs[index].currentPhase = currentTrainingPhase
            if inputMode != .runCOLMAPInApp {
                currentProgress = 0.05
                gaussianSplatJobs[index].progress = 0.05
            }

            let preparedDataset = try GaussianSplatDatasetPreparer().prepareDataset(
                imageFolder: imageURL,
                colmapDirectory: colmapModelDirectory,
                workingDirectory: trainingOutputDirectory
            )

            let trainingRunner = GaussianSplatTrainingRunner()
            currentTrainingRunner = trainingRunner

            let result = try await trainingRunner.run(
                datasetRoot: preparedDataset.datasetRoot,
                exportsDirectory: preparedDataset.exportsDirectory,
                configuration: trainingConfiguration
            ) { [weak self] update in
                await MainActor.run {
                    guard let self else { return }
                    let offset = inputMode == .runCOLMAPInApp ? 0.25 : 0.0
                    let scale = inputMode == .runCOLMAPInApp ? 0.75 : 1.0
                    let overallProgress = min(offset + update.fraction * scale, 1.0)

                    self.currentProgress = overallProgress
                    self.currentTrainingPhase = update.phase

                    if let currentIndex = self.gaussianSplatJobs.firstIndex(where: { $0.id == jobId }) {
                        self.gaussianSplatJobs[currentIndex].progress = overallProgress
                        self.gaussianSplatJobs[currentIndex].currentPhase = update.phase
                    }
                }
            }

            currentTrainingRunner = nil
            currentTrainingPhase = ""

            if Task.isCancelled {
                if let currentIndex = gaussianSplatJobs.firstIndex(where: { $0.id == jobId }) {
                    gaussianSplatJobs[currentIndex].status = .cancelled
                }
            } else if let currentIndex = gaussianSplatJobs.firstIndex(where: { $0.id == jobId }) {
                gaussianSplatJobs[currentIndex].status = .completed
                gaussianSplatJobs[currentIndex].progress = 1
                gaussianSplatJobs[currentIndex].currentPhase = ""
                gaussianSplatJobs[currentIndex].resultSummary = GaussianSplatTrainingResultSummary(
                    exportedPLYCount: result.exportedPLYCount,
                    totalIterations: result.totalIterations
                )
            }

            logger.log("Gaussian Splat job complete with \(result.exportedPLYCount) export(s)")
        } catch is CancellationError {
            currentCOLMAPRunner = nil
            currentTrainingRunner = nil
            currentSfMPhase = ""
            currentTrainingPhase = ""
            if let currentIndex = gaussianSplatJobs.firstIndex(where: { $0.id == jobId }) {
                gaussianSplatJobs[currentIndex].status = .cancelled
                gaussianSplatJobs[currentIndex].currentPhase = ""
            }
        } catch {
            logger.warning("Gaussian Splat job \(jobId) failed: \(error)")
            currentCOLMAPRunner = nil
            currentTrainingRunner = nil
            currentSfMPhase = ""
            currentTrainingPhase = ""
            if let currentIndex = gaussianSplatJobs.firstIndex(where: { $0.id == jobId }) {
                gaussianSplatJobs[currentIndex].status = .failed
                gaussianSplatJobs[currentIndex].errorMessage = "\(error)"
                gaussianSplatJobs[currentIndex].currentPhase = ""
            }
            sendNotification(title: "Gaussian Splat Job Failed", body: jobName)
        }

        currentSfMPhase = ""
        currentTrainingPhase = ""
        persist()
    }

    private func processSfMJob(at index: Int) async {
        let jobId = sfmJobs[index].id
        currentJobId = jobId
        currentProgress = 0
        estimatedTimeRemaining = nil
        currentSfMPhase = ""

        sfmJobs[index].status = .running
        sfmJobs[index].progress = 0
        persist()

        let jobName = sfmJobs[index].jobName
        logger.log("Starting SfM job: \(jobName) (\(jobId))")

        // Resolve bookmarks for sandbox access.
        let (imageURL, outputURL) = sfmJobs[index].resolveBookmarks()

        guard let imageURL, let outputURL else {
            sfmJobs[index].status = .failed
            sfmJobs[index].errorMessage = "Cannot access saved folders. Please remove and re-add the job."
            sendNotification(title: "SfM Job Failed", body: jobName)
            persist()
            return
        }

        let imageAccess = imageURL.startAccessingSecurityScopedResource()
        let outputAccess = outputURL.startAccessingSecurityScopedResource()

        defer {
            if imageAccess { imageURL.stopAccessingSecurityScopedResource() }
            if outputAccess { outputURL.stopAccessingSecurityScopedResource() }
        }

        do {
            try await colmapManager.ensureAvailable()

            let sfmOutputDir = sfmJobs[index].colmapOutputDirectory
            let config = sfmJobs[index].sfmConfiguration.toSfMConfiguration()

            let runner = COLMAPRunner(colmapBinaryURL: colmapManager.binaryURL,
                                     libraryDirectoryURL: colmapManager.libraryDirectoryURL)
            currentCOLMAPRunner = runner

            let sparseDir = try await runner.run(
                configuration: config,
                imageFolder: sfmJobs[index].imageFolder,
                outputFolder: sfmOutputDir
            ) { [weak self] update in
                await MainActor.run {
                    guard let self else { return }
                    self.currentSfMPhase = update.phase.rawValue
                    self.currentProgress = update.fraction

                    // Map phase progress to overall progress.
                    let phaseWeights: [COLMAPRunner.Phase: (offset: Double, weight: Double)] = [
                        .featureExtraction: (0.0, 0.3),
                        .featureMatching: (0.3, 0.3),
                        .sparseReconstruction: (0.6, 0.4),
                        .complete: (1.0, 0.0)
                    ]
                    if let pw = phaseWeights[update.phase] {
                        let overall = pw.offset + pw.weight * update.fraction
                        if let idx = self.sfmJobs.firstIndex(where: { $0.id == jobId }) {
                            self.sfmJobs[idx].progress = overall
                            self.sfmJobs[idx].currentPhase = update.phase.rawValue
                        }
                    }
                }
            }

            currentCOLMAPRunner = nil

            // Parse COLMAP output.
            let result = try COLMAPBinaryParser.parse(directory: sparseDir)

            // Export to the user's output folder in COLMAP binary format.
            let exportDir = sfmJobs[index].colmapOutputDirectory.appending(path: "export")
            try COLMAPBinaryExporter.export(result, to: exportDir)

            // Count total images from the input folder for the summary.
            let totalImages = ImageHelper.getListOfURLs(from: sfmJobs[index].imageFolder).count

            // Store result summary.
            let meanError = result.points3D.isEmpty ? nil :
                result.points3D.map(\.reprojectionError).reduce(0, +) / Double(result.points3D.count)

            if Task.isCancelled {
                if let idx = sfmJobs.firstIndex(where: { $0.id == jobId }) {
                    sfmJobs[idx].status = .cancelled
                }
            } else if let idx = sfmJobs.firstIndex(where: { $0.id == jobId }) {
                sfmJobs[idx].status = .completed
                sfmJobs[idx].progress = 1.0
                sfmJobs[idx].resultSummary = SfMResultSummary(
                    registeredImages: result.imageCount,
                    totalImages: totalImages,
                    sparsePoints: result.pointCount,
                    cameras: result.cameraCount,
                    meanReprojectionError: meanError
                )
            }

            logger.log("SfM job complete: \(result.imageCount) registered, \(result.pointCount) points")

        } catch is CancellationError {
            currentCOLMAPRunner = nil
            if let idx = sfmJobs.firstIndex(where: { $0.id == jobId }) {
                sfmJobs[idx].status = .cancelled
            }
        } catch {
            logger.warning("SfM job \(jobId) failed: \(error)")
            currentCOLMAPRunner = nil
            if let idx = sfmJobs.firstIndex(where: { $0.id == jobId }) {
                sfmJobs[idx].status = .failed
                sfmJobs[idx].errorMessage = "\(error)"
            }
            sendNotification(title: "SfM Job Failed", body: jobName)
        }

        currentSfMPhase = ""
        persist()
    }

    // MARK: - Session creation (nonisolated to avoid blocking main actor)

    private nonisolated func createSession(
        imageFolder: URL,
        configuration: PhotogrammetrySession.Configuration
    ) async throws -> PhotogrammetrySession {
        logger.log("Creating PhotogrammetrySession for \(imageFolder.lastPathComponent)")
        return try PhotogrammetrySession(input: imageFolder, configuration: configuration)
    }

    // MARK: - Notifications

    private var notificationAuthorized = false

    private func sendNotification(title: String, body: String) {
        let center = UNUserNotificationCenter.current()

        Task {
            if !notificationAuthorized {
                let granted = (try? await center.requestAuthorization(options: [.alert, .sound])) ?? false
                notificationAuthorized = granted
                if !granted {
                    logger.warning("Notification permission not granted.")
                    return
                }
            }

            let content = UNMutableNotificationContent()
            content.title = title
            content.body = body
            content.sound = .default

            let request = UNNotificationRequest(
                identifier: UUID().uuidString,
                content: content,
                trigger: nil
            )

            do {
                try await center.add(request)
            } catch {
                logger.warning("Failed to schedule notification: \(error)")
            }
        }
    }
}
