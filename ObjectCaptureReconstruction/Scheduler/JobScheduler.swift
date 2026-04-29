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
    var scheduleConfig: ScheduleConfig = ScheduleConfig()

    private(set) var isRunning = false
    private(set) var isPaused = false
    private(set) var isPauseRequested = false
    private(set) var currentJobId: UUID?
    private(set) var currentProgress: Double = 0
    private(set) var estimatedTimeRemaining: TimeInterval?

    // MARK: - Internal

    private var processingTask: Task<Void, Never>?
    private var currentSession: PhotogrammetrySession?
    private let store = JobStore()

    // MARK: - Persistence helpers

    func persist() {
        store.saveJobs(jobs)
        store.saveSchedule(scheduleConfig)
    }

    func loadFromDisk() {
        jobs = store.loadJobs()
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
    }

    var completedJobCount: Int {
        jobs.filter { $0.status == .completed }.count
    }

    // MARK: - Scheduler control

    func start() {
        guard !isRunning else { return }
        isRunning = true
        isPaused = false
        isPauseRequested = false
        logger.log("Scheduler started.")
        processingTask = Task { await processQueue() }
    }

    func pause() {
        guard isRunning else { return }
        guard !isPaused && !isPauseRequested else { return }

        if currentJobId != nil {
            isPauseRequested = true
            logger.log("Pause requested. Current job will continue until completion because PhotogrammetrySession does not support pausing.")
        } else {
            isPaused = true
            logger.log("Scheduler paused between jobs.")
        }
    }

    func resume() {
        guard isPaused || isPauseRequested else { return }

        let hadPendingPauseRequest = isPauseRequested
        isPauseRequested = false

        if isPaused {
            isPaused = false
            logger.log("Scheduler resumed.")
        } else if hadPendingPauseRequest {
            logger.log("Pending pause request cleared.")
        }
    }

    func cancel() {
        logger.log("Scheduler cancelled.")
        let cancellingJobId = currentJobId
        processingTask?.cancel()
        currentSession?.cancel()
        currentSession = nil
        isRunning = false
        isPaused = false
        isPauseRequested = false
        currentJobId = nil
        currentProgress = 0
        estimatedTimeRemaining = nil

        // Mark the running job as cancelled.
        if let id = cancellingJobId, let idx = jobs.firstIndex(where: { $0.id == id }) {
            jobs[idx].status = .cancelled
        }
        persist()
    }

    // MARK: - Processing loop

    private func processQueue() async {
        defer {
            isRunning = false
            currentJobId = nil
            currentProgress = 0
            estimatedTimeRemaining = nil

            let succeeded = jobs.filter { $0.status == .completed }.count
            let failed = jobs.filter { $0.status == .failed }.count
            sendNotification(
                title: "Queue Complete",
                body: "\(succeeded) succeeded, \(failed) failed"
            )

            persist()
            logger.log("Scheduler finished.")
        }

        while !Task.isCancelled {
            activatePauseIfNeeded()

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

            activatePauseIfNeeded()
            if isPaused { continue }

            // Pick the next pending job.
            guard let index = jobs.firstIndex(where: { $0.status == .pending }) else {
                logger.log("No more pending jobs.")
                break
            }

            await processJob(at: index)
        }
    }

    private func processJob(at index: Int) async {
        let jobId = jobs[index].id
        currentJobId = jobId
        currentProgress = 0
        estimatedTimeRemaining = nil
        var loggedWindowClosureDuringActiveJob = false

        jobs[index].status = .running
        jobs[index].progress = 0
        persist()

        let jobName = jobs[index].modelName
        logger.log("Starting job: \(jobName) (\(jobId))")

        // Resolve bookmarks for sandbox access.
        let (imageURL, modelURL) = jobs[index].resolveBookmarks()
        let imageAccess = imageURL?.startAccessingSecurityScopedResource() ?? false
        let modelAccess = modelURL?.startAccessingSecurityScopedResource() ?? false

        defer {
            if imageAccess { imageURL?.stopAccessingSecurityScopedResource() }
            if modelAccess { modelURL?.stopAccessingSecurityScopedResource() }
        }

        let config = jobs[index].sessionConfiguration.toSessionConfiguration()
        let exportRequests = jobs[index].createModelExportRequests()
        let requests = exportRequests.map(\.photogrammetryRequest)

        guard !requests.isEmpty else {
            jobs[index].status = .failed
            jobs[index].errorMessage = "No detail levels selected."
            sendNotification(title: "Job Failed", body: jobName)
            persist()
            return
        }

        do {
            for exportRequest in exportRequests {
                if let intermediateDirectory = exportRequest.intermediateDirectory {
                    try FileManager.default.createDirectory(
                        at: intermediateDirectory,
                        withIntermediateDirectories: true
                    )
                }
            }

            let session = try await createSession(
                imageFolder: jobs[index].imageFolder,
                configuration: config
            )
            currentSession = session
            try session.process(requests: requests)
            var didReceiveProcessingComplete = false

            // Consume session outputs.
            outputLoop: for try await output in session.outputs {
                if Task.isCancelled { break }
                if !scheduleConfig.isWithinAllowedWindow() && !loggedWindowClosureDuringActiveJob {
                    logger.log("Allowed time window closed during processing. Continuing the active job because PhotogrammetrySession does not support pausing.")
                    loggedWindowClosureDuringActiveJob = true
                }

                switch output {
                case .requestProgress(_, let fraction):
                    currentProgress = fraction
                    if let idx = jobs.firstIndex(where: { $0.id == jobId }) {
                        jobs[idx].progress = fraction
                    }

                case .requestProgressInfo(_, let info):
                    estimatedTimeRemaining = info.estimatedRemainingTime

                case .requestComplete(let request, _):
                    logger.log("Request completed for job \(jobId).")
                    try await finishExport(for: request, in: exportRequests)

                case .requestError(_, let error):
                    logger.warning("Request error for job \(jobId): \(error)")

                case .processingComplete:
                    logger.log("Processing complete for job \(jobId).")
                    didReceiveProcessingComplete = true
                    currentProgress = 1.0
                    estimatedTimeRemaining = nil
                    if let idx = jobs.firstIndex(where: { $0.id == jobId }) {
                        jobs[idx].progress = 1.0
                    }
                    break outputLoop

                default:
                    continue
                }
            }

            currentSession = nil

            if Task.isCancelled {
                if let idx = jobs.firstIndex(where: { $0.id == jobId }) {
                    jobs[idx].status = .cancelled
                }
            } else if !didReceiveProcessingComplete {
                logger.warning("Session output ended before processingComplete for job \(jobId).")
                if let idx = jobs.firstIndex(where: { $0.id == jobId }) {
                    jobs[idx].status = .failed
                    jobs[idx].errorMessage = "Processing ended before completion signal."
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

        currentJobId = nil
        persist()
    }

    private func activatePauseIfNeeded() {
        guard isPauseRequested, currentJobId == nil else { return }
        isPauseRequested = false
        isPaused = true
        logger.log("Scheduler paused between jobs.")
    }

    private nonisolated func finishExport(
        for request: PhotogrammetrySession.Request,
        in exportRequests: [ModelExportRequest]
    ) async throws {
        guard let sourceURL = request.modelFileURL,
              let exportRequest = exportRequests.first(where: { $0.outputURL == sourceURL }),
              let intermediateDirectory = exportRequest.intermediateDirectory else {
            return
        }

        for target in exportRequest.targets where target.format != .usdz {
            try OBJToGLTFConverter.convertOBJFolder(
                at: intermediateDirectory,
                to: target.url,
                format: target.format
            )
            logger.log("Exported \(target.format.displayName) model to \(target.url.lastPathComponent)")
        }

        try? FileManager.default.removeItem(at: intermediateDirectory)
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

private extension PhotogrammetrySession.Request {
    var modelFileURL: URL? {
        if case .modelFile(let url, _, _) = self {
            return url
        }
        return nil
    }
}
