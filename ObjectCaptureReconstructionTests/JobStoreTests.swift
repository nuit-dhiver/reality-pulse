import SwiftData
import XCTest
@testable import Object_Capture_Reconstruction

@MainActor
final class JobStoreTests: XCTestCase {
    func testJobsReloadInQueueOrder() throws {
        let container = try JobStore.makeModelContainer(inMemory: true)
        let first = makeJob(modelName: "First")
        let second = makeJob(modelName: "Second")

        JobStore(modelContainer: container).saveJobs([first, second])

        let reloadedJobs = JobStore(modelContainer: container).loadJobs()
        XCTAssertEqual(reloadedJobs.map(\.id), [first.id, second.id])
        XCTAssertEqual(reloadedJobs.map(\.modelName), ["First", "Second"])
    }

    func testHistoryStatusesSurviveReload() throws {
        let container = try JobStore.makeModelContainer(inMemory: true)
        let completed = makeJob(modelName: "Completed", status: .completed)
        let failed = makeJob(modelName: "Failed", status: .failed, errorMessage: "No images")
        let cancelled = makeJob(modelName: "Cancelled", status: .cancelled)
        let interrupted = makeJob(modelName: "Interrupted", status: .interrupted)

        JobStore(modelContainer: container).saveJobs([completed, failed, cancelled, interrupted])

        let statuses = JobStore(modelContainer: container).loadJobs().map(\.status)
        XCTAssertEqual(statuses, [.completed, .failed, .cancelled, .interrupted])
    }

    func testScheduleSurvivesReload() throws {
        let container = try JobStore.makeModelContainer(inMemory: true)
        let delayedStart = Date(timeIntervalSince1970: 1_800_000_000)
        let schedule = ScheduleConfig(
            delayedStart: delayedStart,
            allowedWindowStart: 22,
            allowedWindowEnd: 6,
            preventSleepWhileQueueActive: true
        )

        JobStore(modelContainer: container).saveSchedule(schedule)

        XCTAssertEqual(JobStore(modelContainer: container).loadSchedule(), schedule)
    }

    func testCompletedOutputsSurviveReload() throws {
        let container = try JobStore.makeModelContainer(inMemory: true)
        var job = makeJob(modelName: "CompletedOutput")
        job.markOutputCompleted(at: job.outputURL(for: .medium))

        JobStore(modelContainer: container).saveJobs([job])

        XCTAssertEqual(
            JobStore(modelContainer: container).loadJobs().first?.completedOutputFilenames,
            [job.outputFilename(for: .medium)]
        )
    }

    func testLaunchRecoveryMarksRunningJobsInterrupted() throws {
        let container = try JobStore.makeModelContainer(inMemory: true)
        let running = makeJob(modelName: "Running", status: .running, progress: 0.42)
        let store = JobStore(modelContainer: container)
        store.saveJobs([running])

        store.performLaunchRecovery()

        let recovered = JobStore(modelContainer: container).loadJobs()
        XCTAssertEqual(recovered.first?.status, .interrupted)
        XCTAssertEqual(recovered.first?.progress, 0)
        XCTAssertEqual(recovered.first?.errorMessage, "Interrupted while the app was not running.")
    }

    func testRetryingInterruptedJobResetsToPending() throws {
        let store = try JobStore(inMemory: true)
        let scheduler = JobScheduler(store: store)
        let interrupted = makeJob(modelName: "Interrupted", status: .interrupted, errorMessage: "Stopped")
        scheduler.addJob(interrupted)

        scheduler.retryJob(interrupted)

        XCTAssertEqual(scheduler.jobs.first?.status, .pending)
        XCTAssertNil(scheduler.jobs.first?.errorMessage)
        XCTAssertEqual(store.loadJobs().first?.status, .pending)
    }

    func testRetryingInterruptedJobPreservesCompletedOutputs() throws {
        let store = try JobStore(inMemory: true)
        let scheduler = JobScheduler(store: store)
        var interrupted = makeJob(modelName: "Interrupted", status: .interrupted, errorMessage: "Stopped")
        interrupted.markOutputCompleted(at: interrupted.outputURL(for: .medium))
        scheduler.addJob(interrupted)

        scheduler.retryJob(interrupted)

        XCTAssertEqual(
            scheduler.jobs.first?.completedOutputFilenames,
            [interrupted.outputFilename(for: .medium)]
        )
        XCTAssertEqual(
            store.loadJobs().first?.completedOutputFilenames,
            [interrupted.outputFilename(for: .medium)]
        )
    }

    func testPrepareRequestsSkipsCompletedOutputsAndDeletesStaleFiles() throws {
        let store = try JobStore(inMemory: true)
        let scheduler = JobScheduler(store: store)
        let outputDirectory = try makeTemporaryDirectory()
        var additionalDetailLevels = CodableDetailLevelOptions()
        additionalDetailLevels.isSelected = true
        additionalDetailLevels.reduced = true
        var job = makeJob(
            modelName: "Multi",
            modelFolder: outputDirectory,
            primaryDetailLevel: .preview,
            additionalDetailLevels: additionalDetailLevels
        )

        let completedURL = job.outputURL(for: .preview)
        let staleURL = job.outputURL(for: .reduced)
        try Data("done".utf8).write(to: completedURL)
        try Data("stale".utf8).write(to: staleURL)
        job.markOutputCompleted(at: completedURL)
        scheduler.addJob(job)

        let requests = try scheduler.prepareRequestsForProcessing(at: 0)

        XCTAssertEqual(requests.count, 1)
        XCTAssertTrue(FileManager.default.fileExists(atPath: completedURL.path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: staleURL.path))
    }

    func testLegacyJSONMigrationImportsJobsAndScheduleOnce() throws {
        let container = try JobStore.makeModelContainer(inMemory: true)
        let legacyDirectory = try makeTemporaryDirectory()
        let legacyJob = makeJob(modelName: "Legacy")
        let legacySchedule = ScheduleConfig(
            delayedStart: Date(timeIntervalSince1970: 1_800_000_001),
            allowedWindowStart: 9,
            allowedWindowEnd: 17,
            preventSleepWhileQueueActive: true
        )

        try JSONEncoder().encode([legacyJob]).write(
            to: legacyDirectory.appending(path: "jobs.json")
        )
        try JSONEncoder().encode(legacySchedule).write(
            to: legacyDirectory.appending(path: "schedule.json")
        )

        let firstStore = JobStore(
            modelContainer: container,
            legacyStoreDirectory: legacyDirectory
        )
        firstStore.performLaunchRecovery()

        XCTAssertEqual(firstStore.loadJobs().map(\.id), [legacyJob.id])
        XCTAssertEqual(firstStore.loadSchedule(), legacySchedule)

        let secondStore = JobStore(
            modelContainer: container,
            legacyStoreDirectory: legacyDirectory
        )
        secondStore.performLaunchRecovery()

        XCTAssertEqual(secondStore.loadJobs().map(\.id), [legacyJob.id])
    }

    private func makeJob(
        modelName: String,
        modelFolder: URL? = nil,
        primaryDetailLevel: CodableDetailLevel = .medium,
        additionalDetailLevels: CodableDetailLevelOptions = CodableDetailLevelOptions(),
        status: JobStatus = .pending,
        progress: Double = 0,
        errorMessage: String? = nil
    ) -> ReconstructionJob {
        ReconstructionJob(
            imageFolder: URL(fileURLWithPath: "/tmp/\(modelName)-images"),
            modelFolder: modelFolder ?? URL(fileURLWithPath: "/tmp/\(modelName)-models"),
            modelName: modelName,
            primaryDetailLevel: primaryDetailLevel,
            additionalDetailLevels: additionalDetailLevels,
            status: status,
            progress: progress,
            errorMessage: errorMessage
        )
    }

    private func makeTemporaryDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appending(
            path: UUID().uuidString,
            directoryHint: .isDirectory
        )
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        return directory
    }
}
