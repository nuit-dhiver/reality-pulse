/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
SwiftData persistence layer for saving and loading the job queue and schedule.
*/

import Foundation
import SwiftData
import WatermarkCore
import os

private let logger = Logger(subsystem: ObjectCaptureReconstructionApp.subsystem,
                            category: "JobStore")

/// Saves and loads jobs and schedule settings through SwiftData.
@MainActor
class JobStore {
    private static let scheduleSettingsId = "main"
    private static let legacyMigrationId = "legacy-json-v1"

    private let modelContext: ModelContext
    private let legacyJobsFileURL: URL
    private let legacyScheduleFileURL: URL

    private(set) var lastErrorMessage: String?

    init(modelContainer: ModelContainer, legacyStoreDirectory: URL? = nil) {
        modelContext = ModelContext(modelContainer)

        let storeDir: URL
        if let legacyStoreDirectory {
            storeDir = legacyStoreDirectory
        } else {
            let appSupport = FileManager.default.urls(
                for: .applicationSupportDirectory,
                in: .userDomainMask
            ).first!
            storeDir = appSupport.appending(path: "ObjectCaptureReconstruction")
        }

        try? FileManager.default.createDirectory(at: storeDir, withIntermediateDirectories: true)
        legacyJobsFileURL = storeDir.appending(path: "jobs.json")
        legacyScheduleFileURL = storeDir.appending(path: "schedule.json")
    }

    convenience init(inMemory: Bool) throws {
        try self.init(modelContainer: Self.makeModelContainer(inMemory: inMemory))
    }

    static func makeModelContainer(inMemory: Bool = false) throws -> ModelContainer {
        let schema = Schema([
            PersistentJob.self,
            PersistentScheduleSettings.self,
            PersistentMigrationState.self,
            PersistentExportRecord.self,
            PersistentWatermarkKey.self
        ])
        let configuration = ModelConfiguration(
            schema: schema,
            isStoredInMemoryOnly: inMemory
        )
        return try ModelContainer(for: schema, configurations: [configuration])
    }

    // MARK: - Launch recovery

    func performLaunchRecovery() {
        do {
            try migrateLegacyJSONIfNeeded()
            try markRunningJobsInterrupted()
            try saveIfNeeded()
            lastErrorMessage = nil
        } catch {
            recordError("Failed to recover persisted jobs: \(error.localizedDescription)", error: error)
        }
    }

    // MARK: - Jobs

    func saveJob(_ job: ReconstructionJob) {
        do {
            let queueOrder = try queueOrderForSavedJob(id: job.id)
            try upsert(job, queueOrder: queueOrder)
            try saveIfNeeded()
            lastErrorMessage = nil
        } catch {
            recordError("Failed to save job: \(error.localizedDescription)", error: error)
        }
    }

    func saveJobs(_ jobs: [ReconstructionJob]) {
        do {
            for (queueOrder, job) in jobs.enumerated() {
                try upsert(job, queueOrder: queueOrder)
            }
            try saveIfNeeded()
            logger.log("Saved \(jobs.count) job(s) to SwiftData.")
            lastErrorMessage = nil
        } catch {
            recordError("Failed to save jobs: \(error.localizedDescription)", error: error)
        }
    }

    func deleteJob(id: UUID) {
        do {
            if let job = try persistentJob(id: id) {
                modelContext.delete(job)
                try saveIfNeeded()
                lastErrorMessage = nil
            }
        } catch {
            recordError("Failed to delete job: \(error.localizedDescription)", error: error)
        }
    }

    func loadJobs() -> [ReconstructionJob] {
        do {
            var descriptor = FetchDescriptor<PersistentJob>(
                sortBy: [
                    SortDescriptor(\.queueOrder),
                    SortDescriptor(\.createdAt)
                ]
            )
            descriptor.includePendingChanges = true

            let persistedJobs = try modelContext.fetch(descriptor)
            var jobs = try persistedJobs.map { try $0.toJob() }

            for index in jobs.indices {
                _ = jobs[index].resolveBookmarks()
            }

            logger.log("Loaded \(jobs.count) job(s) from SwiftData.")
            return jobs
        } catch {
            recordError("Failed to load jobs: \(error.localizedDescription)", error: error)
            return []
        }
    }

    // MARK: - Provenance export records

    /// Export records are append-only provenance history: they are never
    /// updated in place and intentionally survive job deletion.
    ///
    /// Returns false when persistence failed — callers must treat the
    /// affected files as unmarked, because a marked file whose key was never
    /// recorded is untraceable.
    @discardableResult
    func saveExportRecords(_ records: [WatermarkRecord]) -> Bool {
        guard !records.isEmpty else { return true }
        do {
            for record in records {
                modelContext.insert(try PersistentExportRecord(record: record))
            }
            try saveIfNeeded()
            logger.log("Saved \(records.count) provenance export record(s).")
            lastErrorMessage = nil
            return true
        } catch {
            modelContext.rollback()
            recordError("Failed to save export records: \(error.localizedDescription)", error: error)
            return false
        }
    }

    /// Remove records again — used to roll back when the file a record
    /// describes could not be put in place after the record was saved.
    func deleteExportRecords(recordIds: [UUID]) {
        guard !recordIds.isEmpty else { return }
        do {
            let ids = Set(recordIds)
            var descriptor = FetchDescriptor<PersistentExportRecord>(
                predicate: #Predicate { ids.contains($0.recordId) }
            )
            descriptor.includePendingChanges = true
            for record in try modelContext.fetch(descriptor) {
                modelContext.delete(record)
            }
            try saveIfNeeded()
        } catch {
            recordError("Failed to roll back export records: \(error.localizedDescription)", error: error)
        }
    }

    func exportRecords(jobId: UUID) -> [WatermarkRecord] {
        do {
            var descriptor = FetchDescriptor<PersistentExportRecord>(
                predicate: #Predicate { $0.jobId == jobId },
                sortBy: [SortDescriptor(\.createdAt)]
            )
            descriptor.includePendingChanges = true
            return try modelContext.fetch(descriptor).compactMap { try? $0.toRecord() }
        } catch {
            recordError("Failed to load export records: \(error.localizedDescription)", error: error)
            return []
        }
    }

    /// Most recent record for one exported file, used to make USDZ stamping
    /// idempotent across job retries.
    func latestExportRecord(jobId: UUID, detailLevel: String, format: String) -> WatermarkRecord? {
        do {
            var descriptor = FetchDescriptor<PersistentExportRecord>(
                predicate: #Predicate {
                    $0.jobId == jobId
                        && $0.detailLevelRawValue == detailLevel
                        && $0.formatRawValue == format
                },
                sortBy: [SortDescriptor(\.createdAt, order: .reverse)]
            )
            descriptor.fetchLimit = 1
            descriptor.includePendingChanges = true
            return try modelContext.fetch(descriptor).first.flatMap { try? $0.toRecord() }
        } catch {
            recordError("Failed to load export record: \(error.localizedDescription)", error: error)
            return nil
        }
    }

    // MARK: - Watermark key library

    /// Saved keys, most recently used first. Returns metadata only — key
    /// material never leaves the store except through `watermarkKey(id:)`.
    func watermarkKeys() -> [WatermarkKeyInfo] {
        do {
            var descriptor = FetchDescriptor<PersistentWatermarkKey>(
                sortBy: [SortDescriptor(\.createdAt, order: .reverse)]
            )
            descriptor.includePendingChanges = true
            return try modelContext.fetch(descriptor).map(\.info)
        } catch {
            recordError("Failed to load watermark keys: \(error.localizedDescription)", error: error)
            return []
        }
    }

    /// Create and save a new labeled key. Returns nil when the label is blank
    /// or already taken.
    func createWatermarkKey(label: String) -> WatermarkKeyInfo? {
        let trimmed = label.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }

        do {
            var descriptor = FetchDescriptor<PersistentWatermarkKey>(
                predicate: #Predicate { $0.label == trimmed }
            )
            descriptor.fetchLimit = 1
            descriptor.includePendingChanges = true
            guard try modelContext.fetch(descriptor).isEmpty else { return nil }

            let saved = PersistentWatermarkKey(label: trimmed, key: .random())
            modelContext.insert(saved)
            try saveIfNeeded()
            logger.log("Created watermark key '\(trimmed, privacy: .public)'.")
            lastErrorMessage = nil
            return saved.info
        } catch {
            modelContext.rollback()
            recordError("Failed to create watermark key: \(error.localizedDescription)", error: error)
            return nil
        }
    }

    /// Resolve a saved key for embedding, marking it as used.
    func watermarkKey(id: UUID) -> (key: WatermarkKey, label: String)? {
        do {
            var descriptor = FetchDescriptor<PersistentWatermarkKey>(
                predicate: #Predicate { $0.id == id }
            )
            descriptor.fetchLimit = 1
            descriptor.includePendingChanges = true
            guard let saved = try modelContext.fetch(descriptor).first,
                  let key = saved.watermarkKey else { return nil }
            saved.lastUsedAt = Date()
            try saveIfNeeded()
            return (key, saved.label)
        } catch {
            recordError("Failed to load watermark key: \(error.localizedDescription)", error: error)
            return nil
        }
    }

    /// Saved keys are provenance history: deleting one makes every file marked
    /// with it unverifiable, so this is only for keys created by mistake.
    func deleteWatermarkKey(id: UUID) {
        do {
            var descriptor = FetchDescriptor<PersistentWatermarkKey>(
                predicate: #Predicate { $0.id == id }
            )
            descriptor.fetchLimit = 1
            if let saved = try modelContext.fetch(descriptor).first {
                modelContext.delete(saved)
                try saveIfNeeded()
            }
        } catch {
            recordError("Failed to delete watermark key: \(error.localizedDescription)", error: error)
        }
    }

    // MARK: - Schedule

    func saveSchedule(_ config: ScheduleConfig) {
        do {
            if let settings = try scheduleSettings() {
                settings.update(from: config)
            } else {
                modelContext.insert(PersistentScheduleSettings(
                    id: Self.scheduleSettingsId,
                    config: config
                ))
            }
            try saveIfNeeded()
            lastErrorMessage = nil
        } catch {
            recordError("Failed to save schedule: \(error.localizedDescription)", error: error)
        }
    }

    func loadSchedule() -> ScheduleConfig {
        do {
            let config = try scheduleSettings()?.scheduleConfig ?? ScheduleConfig()
            return config
        } catch {
            recordError("Failed to load schedule: \(error.localizedDescription)", error: error)
            return ScheduleConfig()
        }
    }

    // MARK: - Private helpers

    private func upsert(_ job: ReconstructionJob, queueOrder: Int) throws {
        if let existingJob = try persistentJob(id: job.id) {
            try existingJob.update(from: job, queueOrder: queueOrder)
        } else {
            modelContext.insert(try PersistentJob(job: job, queueOrder: queueOrder))
        }
    }

    private func persistentJob(id: UUID) throws -> PersistentJob? {
        var descriptor = FetchDescriptor<PersistentJob>(
            predicate: #Predicate { $0.id == id }
        )
        descriptor.fetchLimit = 1
        return try modelContext.fetch(descriptor).first
    }

    private func scheduleSettings() throws -> PersistentScheduleSettings? {
        let settingsId = Self.scheduleSettingsId
        var descriptor = FetchDescriptor<PersistentScheduleSettings>(
            predicate: #Predicate { $0.id == settingsId }
        )
        descriptor.fetchLimit = 1
        return try modelContext.fetch(descriptor).first
    }

    private func queueOrderForSavedJob(id: UUID) throws -> Int {
        if let existingJob = try persistentJob(id: id) {
            return existingJob.queueOrder
        }

        var descriptor = FetchDescriptor<PersistentJob>(
            sortBy: [SortDescriptor(\.queueOrder, order: .reverse)]
        )
        descriptor.fetchLimit = 1
        return (try modelContext.fetch(descriptor).first?.queueOrder ?? -1) + 1
    }

    private func markRunningJobsInterrupted() throws {
        let runningStatus = JobStatus.running.rawValue
        let descriptor = FetchDescriptor<PersistentJob>(
            predicate: #Predicate { $0.statusRawValue == runningStatus }
        )
        let runningJobs = try modelContext.fetch(descriptor)
        for job in runningJobs {
            job.statusRawValue = JobStatus.interrupted.rawValue
            job.progress = 0
            job.errorMessage = "Interrupted while the app was not running."
            job.updatedAt = Date()
        }

        if !runningJobs.isEmpty {
            logger.log("Marked \(runningJobs.count) interrupted job(s) during launch recovery.")
        }
    }

    private func migrateLegacyJSONIfNeeded() throws {
        let migrationId = Self.legacyMigrationId
        var migrationDescriptor = FetchDescriptor<PersistentMigrationState>(
            predicate: #Predicate { $0.id == migrationId }
        )
        migrationDescriptor.fetchLimit = 1

        guard try modelContext.fetch(migrationDescriptor).isEmpty else { return }

        let hasLegacyJobs = FileManager.default.fileExists(atPath: legacyJobsFileURL.path)
        let hasLegacySchedule = FileManager.default.fileExists(atPath: legacyScheduleFileURL.path)
        guard hasLegacyJobs || hasLegacySchedule else { return }

        if hasLegacyJobs, try persistentJobCount() == 0 {
            let data = try Data(contentsOf: legacyJobsFileURL)
            let jobs = try JSONDecoder().decode([ReconstructionJob].self, from: data)
            for (queueOrder, job) in jobs.enumerated() {
                try upsert(job, queueOrder: queueOrder)
            }
            logger.log("Migrated \(jobs.count) legacy JSON job(s) into SwiftData.")
        }

        if hasLegacySchedule, try scheduleSettings() == nil {
            let data = try Data(contentsOf: legacyScheduleFileURL)
            let schedule = try JSONDecoder().decode(ScheduleConfig.self, from: data)
            modelContext.insert(PersistentScheduleSettings(
                id: Self.scheduleSettingsId,
                config: schedule
            ))
            logger.log("Migrated legacy JSON schedule into SwiftData.")
        }

        modelContext.insert(PersistentMigrationState(id: migrationId))
        try saveIfNeeded()
    }

    private func persistentJobCount() throws -> Int {
        let descriptor = FetchDescriptor<PersistentJob>()
        return try modelContext.fetchCount(descriptor)
    }

    private func saveIfNeeded() throws {
        if modelContext.hasChanges {
            try modelContext.save()
        }
    }

    private func recordError(_ message: String, error: Error) {
        lastErrorMessage = message
        logger.warning("\(message, privacy: .public) \(String(describing: error), privacy: .public)")
    }
}
