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
    var scheduleConfig: ScheduleConfig = ScheduleConfig() {
        didSet {
            updateSleepPreventionActivity()
            if !isLoadingPersistedState {
                store.saveSchedule(scheduleConfig)
            }
        }
    }

    private(set) var isRunning = false
    private(set) var isPaused = false
    private(set) var isPauseRequested = false
    private(set) var currentJobId: UUID?
    private(set) var currentProgress: Double = 0
    private(set) var estimatedTimeRemaining: TimeInterval?

    // MARK: - Internal

    private var processingTask: Task<Void, Never>?
    private var currentSession: PhotogrammetrySession?
    private var sleepPreventionActivity: NSObjectProtocol?
    private var isLoadingPersistedState = false
    private let store: JobStore

    init(store: JobStore) {
        self.store = store
    }

    var persistenceErrorMessage: String? {
        store.lastErrorMessage
    }

    // MARK: - Persistence helpers

    func persist() {
        store.saveJobs(jobs)
        store.saveSchedule(scheduleConfig)
    }

    func performLaunchRecovery() {
        store.performLaunchRecovery()
    }

    func loadFromDisk() {
        jobs = store.loadJobs()
        isLoadingPersistedState = true
        scheduleConfig = store.loadSchedule()
        isLoadingPersistedState = false
    }

    // MARK: - Queue management

    func addJob(_ job: ReconstructionJob) {
        jobs.append(job)
        persist()
    }

    func removeJob(_ job: ReconstructionJob) {
        jobs.removeAll { $0.id == job.id }
        store.deleteJob(id: job.id)
        store.saveJobs(jobs)
    }

    func removeJobs(at offsets: IndexSet) {
        let removedIds = offsets.map { jobs[$0].id }
        jobs.remove(atOffsets: offsets)
        for id in removedIds {
            store.deleteJob(id: id)
        }
        store.saveJobs(jobs)
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
        store.saveJob(jobs[index])
    }

    func updateJob(_ job: ReconstructionJob) {
        guard let index = jobs.firstIndex(where: { $0.id == job.id }) else { return }
        jobs[index] = job
        store.saveJob(job)
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
        updateSleepPreventionActivity()
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
        updateSleepPreventionActivity()

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
            updateSleepPreventionActivity()

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

    // MARK: - Sleep prevention

    private var shouldPreventSleep: Bool {
        isRunning && scheduleConfig.preventSleepWhileQueueActive
    }

    private func updateSleepPreventionActivity() {
        if shouldPreventSleep {
            beginSleepPreventionActivityIfNeeded()
        } else {
            endSleepPreventionActivityIfNeeded()
        }
    }

    private func beginSleepPreventionActivityIfNeeded() {
        guard sleepPreventionActivity == nil else { return }
        sleepPreventionActivity = ProcessInfo.processInfo.beginActivity(
            options: [.idleSystemSleepDisabled],
            reason: "Reality Pulse queue is active"
        )
        logger.log("Sleep prevention activity started.")
    }

    private func endSleepPreventionActivityIfNeeded() {
        guard let activity = sleepPreventionActivity else { return }
        ProcessInfo.processInfo.endActivity(activity)
        sleepPreventionActivity = nil
        logger.log("Sleep prevention activity ended.")
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

        jobs[index].progress = jobs[index].completedOutputFraction()
        persist()

        let config = jobs[index].sessionConfiguration.toSessionConfiguration()
        let totalOutputCount = jobs[index].requestedOutputCount
        let requests: [PhotogrammetrySession.Request]

        do {
            requests = try prepareRequestsForProcessing(at: index)
        } catch {
            jobs[index].status = .failed
            jobs[index].errorMessage = "\(error)"
            sendNotification(title: "Job Failed", body: jobName)
            persist()
            return
        }

        guard totalOutputCount > 0 else {
            jobs[index].status = .failed
            jobs[index].errorMessage = "No detail levels selected."
            sendNotification(title: "Job Failed", body: jobName)
            persist()
            return
        }

        guard !requests.isEmpty else {
            jobs[index].status = .completed
            jobs[index].progress = 1.0
            sendNotification(title: "Job Complete", body: jobName)
            persist()
            return
        }

        do {
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
                    if let idx = jobs.firstIndex(where: { $0.id == jobId }) {
                        let completedCount = jobs[idx].completedOutputCount()
                        let overallProgress = (
                            Double(completedCount) + fraction
                        ) / Double(totalOutputCount)
                        currentProgress = overallProgress
                        jobs[idx].progress = overallProgress
                    }

                case .requestProgressInfo(_, let info):
                    estimatedTimeRemaining = info.estimatedRemainingTime

                case .requestComplete(let request, _):
                    logger.log("Request completed for job \(jobId).")
                    if let idx = jobs.firstIndex(where: { $0.id == jobId }) {
                        markCompletedOutput(for: request, on: &jobs[idx])
                        jobs[idx].progress = jobs[idx].completedOutputFraction()
                        currentProgress = jobs[idx].progress
                        store.saveJob(jobs[idx])
                    }

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

    func prepareRequestsForProcessing(at index: Int) throws -> [PhotogrammetrySession.Request] {
        for level in jobs[index].requestedDetailLevels {
            if jobs[index].hasCompletedOutputFile(for: level) {
                continue
            }

            let url = jobs[index].outputURL(for: level)
            if FileManager.default.fileExists(atPath: url.path) {
                try FileManager.default.removeItem(at: url)
                logger.log("Removed stale existing output before retry: \(url.lastPathComponent)")
            }
        }

        return jobs[index].createReconstructionRequests(skippingCompletedOutputs: true)
    }

    private func markCompletedOutput(
        for request: PhotogrammetrySession.Request,
        on job: inout ReconstructionJob
    ) {
        guard case .modelFile(let url, _, _) = request else { return }
        job.markOutputCompleted(at: url)
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
