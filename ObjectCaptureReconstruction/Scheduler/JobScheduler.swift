/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Sequential job scheduler that processes reconstruction jobs one at a time,
respecting optional time-window constraints.
*/

import Foundation
import RealityKit
import UserNotifications
import WatermarkCore
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
            // All detail levels are already on disk (e.g. a retry after every
            // USDZ was written) — finalize must still run, or derived exports
            // and provenance stamping silently never happen for this job.
            jobs[index].status = .completed
            jobs[index].progress = 1.0
            let notificationBody = await finalizeCompletedJob(for: jobs[index]) ?? jobName
            sendNotification(title: "Job Complete", body: notificationBody)
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
                let notificationBody = await finalizeCompletedJob(for: jobs[idx]) ?? jobName
                sendNotification(title: "Job Complete", body: notificationBody)
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

    /// Post-reconstruction finalize, in a deliberate order: derived formats
    /// are converted first (from the still-pristine USDZ, so they never
    /// inherit its marks), then the USDZ itself is texture-stamped last.
    /// Returns the notification body; stamping problems are surfaced there
    /// because the queue row only shows errors for failed jobs.
    private func finalizeCompletedJob(for job: ReconstructionJob) async -> String? {
        let (sharedKey, keyWarning) = resolveSharedKey(for: job)
        let exportSummary = await exportAdditionalFormats(for: job, sharedKey: sharedKey)
        let stampWarning = await stampCompletedUSDZOutputs(for: job, sharedKey: sharedKey)

        let warnings = [keyWarning, stampWarning].compactMap { $0 }
        switch (exportSummary, warnings.isEmpty) {
        case (nil, true):
            return nil
        case (let summary?, true):
            return summary
        case (nil, false):
            return "\(job.modelName) — \(warnings.joined(separator: "; "))"
        case (let summary?, false):
            return "\(summary); \(warnings.joined(separator: "; "))"
        }
    }

    /// Resolve the job's selected library key. A missing key (deleted since
    /// the job was created) falls back to fresh per-copy keys — strictly more
    /// traceable, but the user asked for the label, so it is surfaced.
    private func resolveSharedKey(
        for job: ReconstructionJob
    ) -> (shared: SharedWatermarkKey?, warning: String?) {
        guard job.isWatermarkEnabled, let keyId = job.watermarkKeyId else { return (nil, nil) }
        guard let resolved = store.watermarkKey(id: keyId) else {
            logger.warning("Saved watermark key \(keyId) is missing; falling back to per-copy keys.")
            return (nil, "saved watermark key is missing, used fresh per-copy keys instead")
        }
        return (SharedWatermarkKey(key: resolved.key, label: resolved.label), nil)
    }

    /// Returns a user-facing warning when any stamp failed, nil when all
    /// stamps succeeded or watermarking is off.
    private func stampCompletedUSDZOutputs(
        for job: ReconstructionJob,
        sharedKey: SharedWatermarkKey?
    ) async -> String? {
        guard job.isWatermarkEnabled else { return nil }

        var attempted = 0
        var failed = 0
        for level in job.requestedDetailLevels where job.hasCompletedOutputFile(for: level) {
            let usdzURL = job.outputURL(for: level)
            let recordedSHA = store.latestExportRecord(
                jobId: job.id, detailLevel: level.rawValue, format: "usdz"
            )?.fileSHA256
            let stamp = WatermarkStamp.next(sharedKey: sharedKey)

            do {
                // Stage off the main actor, persist the record, then swap the
                // file in — never the other way around: a marked file without
                // a persisted key would be untraceable, and a later retry
                // would stamp a second layer over it.
                guard let staged = try await stageStampIfNeeded(
                    usdzURL: usdzURL, recordedSHA: recordedSHA, stamp: stamp
                ) else { continue }

                attempted += 1
                let record = WatermarkRecord(
                    jobId: job.id,
                    format: "usdz",
                    detailLevel: level.rawValue,
                    filename: usdzURL.lastPathComponent,
                    filePath: usdzURL.path,
                    key: stamp.key,
                    keyLabel: stamp.keyLabel,
                    channels: [WatermarkRecord.Channel.texture],
                    geometry: nil,
                    texture: WatermarkRecord.TextureChannelInfo(
                        parameters: stamp.textureParameters,
                        images: staged.stampedImages
                    ),
                    fileSHA256: staged.fileSHA256
                )
                guard store.saveExportRecords([record]) else {
                    USDZStamper.discard(staged)
                    failed += 1
                    continue
                }

                do {
                    try USDZStamper.commit(staged, to: usdzURL)
                } catch {
                    // The record is saved but the file was never replaced.
                    // Roll the record back, otherwise it would describe bytes
                    // that do not exist and defeat the idempotence check on
                    // the next retry.
                    USDZStamper.discard(staged)
                    store.deleteExportRecords(recordIds: [record.recordId])
                    failed += 1
                    logger.warning("USDZ provenance commit failed for \(usdzURL.lastPathComponent, privacy: .public): \(error.localizedDescription)")
                }
            } catch {
                attempted += 1
                failed += 1
                logger.warning("USDZ provenance stamp failed for \(usdzURL.lastPathComponent, privacy: .public): \(error.localizedDescription)")
            }
        }

        guard failed > 0 else { return nil }
        return "provenance stamping failed for \(failed) of \(attempted) USDZ file(s)"
    }

    private func exportAdditionalFormats(
        for job: ReconstructionJob,
        sharedKey: SharedWatermarkKey?
    ) async -> String? {
        guard !job.exportFormats.isEmpty else { return nil }

        // Idempotence: a slot whose file still matches its newest record is
        // left alone, so re-finalizing (e.g. a retry where every USDZ already
        // existed) cannot silently re-key files that are already traceable.
        var recordedHashes: [ModelExportService.ExportSlot: String] = [:]
        if job.isWatermarkEnabled {
            for level in job.requestedDetailLevels {
                for format in job.exportFormats {
                    guard let record = store.latestExportRecord(
                        jobId: job.id, detailLevel: level.rawValue, format: format.fileExtension
                    ) else { continue }
                    recordedHashes[.init(detailLevel: level, format: format)] = record.fileSHA256
                }
            }
        }

        do {
            let exportedFiles = try await runExports(
                job: job,
                formats: job.exportFormats,
                embedWatermark: job.isWatermarkEnabled,
                sharedKey: sharedKey,
                recordedHashes: recordedHashes
            )
            guard !exportedFiles.isEmpty else { return nil }

            let markedFiles = exportedFiles.filter { $0.record != nil }
            guard store.saveExportRecords(markedFiles.compactMap(\.record)) else {
                // Marked files whose keys were never persisted are
                // untraceable — remove them rather than ship them silently.
                for file in markedFiles {
                    try? FileManager.default.removeItem(at: file.url)
                }
                return "\(job.modelName) — additional export failed: provenance records could not be saved"
            }

            let freshCount = exportedFiles.filter { !$0.isUpToDate }.count
            guard freshCount > 0 else { return nil }
            return "\(job.modelName) — exported \(freshCount) additional file(s)"
        } catch {
            logger.warning("Additional format export failed for \(job.modelName): \(error.localizedDescription)")
            return "\(job.modelName) — reconstruction complete, but additional export failed"
        }
    }

    // MARK: - Finalize work (nonisolated to avoid blocking main actor)

    /// Format conversion, watermark embedding, and hashing are all CPU- and
    /// I/O-heavy (splat generation alone samples a million points), so they
    /// run off the main actor like `createSession` does.
    private nonisolated func runExports(
        job: ReconstructionJob,
        formats: Set<ModelExportFormat>,
        embedWatermark: Bool,
        sharedKey: SharedWatermarkKey?,
        recordedHashes: [ModelExportService.ExportSlot: String]
    ) async throws -> [ModelExportService.ExportedFile] {
        try ModelExportService.exportCompletedOutputs(
            for: job,
            formats: formats,
            embedWatermark: embedWatermark,
            sharedKey: sharedKey,
            recordedHashes: recordedHashes
        )
    }

    /// Returns nil when the file already matches its recorded hash (nothing to
    /// do), otherwise a staged copy awaiting record persistence and commit.
    private nonisolated func stageStampIfNeeded(
        usdzURL: URL,
        recordedSHA: String?,
        stamp: WatermarkStamp
    ) async throws -> USDZStamper.StagedStamp? {
        if let recordedSHA,
           let currentSHA = try? WatermarkingService.sha256Hex(of: usdzURL),
           currentSHA == recordedSHA {
            return nil
        }
        return try USDZStamper.stage(usdzURL: usdzURL, stamp: stamp)
    }

    /// Persist provenance records produced by an export outside the scheduler
    /// path (the dashboard's manual "Export As" flow). Returns false when the
    /// records could not be saved — the caller must treat the marked files as
    /// untraceable.
    @discardableResult
    func saveExportRecords(_ records: [WatermarkRecord]) -> Bool {
        store.saveExportRecords(records)
    }

    func exportRecords(jobId: UUID) -> [WatermarkRecord] {
        store.exportRecords(jobId: jobId)
    }

    // MARK: - Watermark key library

    func watermarkKeys() -> [WatermarkKeyInfo] {
        store.watermarkKeys()
    }

    /// Returns nil when the label is blank or already in use.
    func createWatermarkKey(label: String) -> WatermarkKeyInfo? {
        store.createWatermarkKey(label: label)
    }

    /// Resolve a job's selected library key for an export outside the
    /// scheduler path (the dashboard's manual "Export As" flow).
    func sharedWatermarkKey(for job: ReconstructionJob) -> SharedWatermarkKey? {
        resolveSharedKey(for: job).shared
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
