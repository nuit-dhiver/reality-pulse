/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Persistence layer for saving and loading the job queue and schedule to disk.
*/

import Foundation
import os

private let logger = Logger(subsystem: ObjectCaptureReconstructionApp.subsystem,
                            category: "JobStore")

/// Saves and loads `[ReconstructionJob]` and `ScheduleConfig` as JSON files
/// in the app's Application Support directory.
@MainActor
class JobStore {

    private let jobsFileURL: URL
    private let scheduleFileURL: URL
    private let sfmJobsFileURL: URL

    init() {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let storeDir = appSupport.appending(path: "RealityPulse")

        // Ensure the directory exists.
        try? FileManager.default.createDirectory(at: storeDir, withIntermediateDirectories: true)

        jobsFileURL = storeDir.appending(path: "jobs.json")
        scheduleFileURL = storeDir.appending(path: "schedule.json")
        sfmJobsFileURL = storeDir.appending(path: "sfm_jobs.json")
    }

    // MARK: - Jobs

    func saveJobs(_ jobs: [ReconstructionJob]) {
        do {
            let data = try JSONEncoder().encode(jobs)
            try data.write(to: jobsFileURL, options: .atomic)
            logger.log("Saved \(jobs.count) job(s) to disk.")
        } catch {
            logger.warning("Failed to save jobs: \(error)")
        }
    }

    func loadJobs() -> [ReconstructionJob] {
        guard FileManager.default.fileExists(atPath: jobsFileURL.path()) else { return [] }
        do {
            let data = try Data(contentsOf: jobsFileURL)
            var jobs = try JSONDecoder().decode([ReconstructionJob].self, from: data)

            // Resolve security-scoped bookmarks and reset any stale running state.
            for index in jobs.indices {
                _ = jobs[index].resolveBookmarks()
                if jobs[index].status == .running {
                    jobs[index].status = .pending
                    jobs[index].progress = 0
                }
            }

            logger.log("Loaded \(jobs.count) job(s) from disk.")
            return jobs
        } catch {
            logger.warning("Failed to load jobs: \(error)")
            return []
        }
    }

    // MARK: - Schedule

    func saveSchedule(_ config: ScheduleConfig) {
        do {
            let data = try JSONEncoder().encode(config)
            try data.write(to: scheduleFileURL, options: .atomic)
        } catch {
            logger.warning("Failed to save schedule: \(error)")
        }
    }

    func loadSchedule() -> ScheduleConfig {
        guard FileManager.default.fileExists(atPath: scheduleFileURL.path()) else { return ScheduleConfig() }
        do {
            let data = try Data(contentsOf: scheduleFileURL)
            return try JSONDecoder().decode(ScheduleConfig.self, from: data)
        } catch {
            logger.warning("Failed to load schedule: \(error)")
            return ScheduleConfig()
        }
    }

    // MARK: - SfM Jobs

    func saveSfMJobs(_ jobs: [SfMJob]) {
        do {
            let data = try JSONEncoder().encode(jobs)
            try data.write(to: sfmJobsFileURL, options: .atomic)
            logger.log("Saved \(jobs.count) SfM job(s) to disk.")
        } catch {
            logger.warning("Failed to save SfM jobs: \(error)")
        }
    }

    func loadSfMJobs() -> [SfMJob] {
        guard FileManager.default.fileExists(atPath: sfmJobsFileURL.path()) else { return [] }
        do {
            let data = try Data(contentsOf: sfmJobsFileURL)
            var jobs = try JSONDecoder().decode([SfMJob].self, from: data)

            for index in jobs.indices {
                _ = jobs[index].resolveBookmarks()
                if jobs[index].status == .running {
                    jobs[index].status = .pending
                    jobs[index].progress = 0
                }
            }

            logger.log("Loaded \(jobs.count) SfM job(s) from disk.")
            return jobs
        } catch {
            logger.warning("Failed to load SfM jobs: \(error)")
            return []
        }
    }
}
