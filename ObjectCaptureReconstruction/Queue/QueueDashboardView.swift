/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Main dashboard showing the job queue, scheduler controls, and overall progress.
*/

import SwiftUI
import os

private let logger = Logger(subsystem: ObjectCaptureReconstructionApp.subsystem,
                            category: "QueueDashboardView")

struct QueueDashboardView: View {
    @Environment(AppDataModel.self) private var appDataModel: AppDataModel

    var body: some View {
        VStack(spacing: 0) {
            // Header with queue controls
            QueueHeaderView()

            Divider()

            // Job list
            if appDataModel.scheduler.jobs.isEmpty && appDataModel.scheduler.sfmJobs.isEmpty {
                emptyState
            } else {
                jobList
            }

            Divider()

            // Footer with add/schedule buttons
            QueueFooterView()
        }
    }

    private var emptyState: some View {
        VStack(spacing: 12) {
            Spacer()
            Image(systemName: "cube.transparent")
                .resizable()
                .aspectRatio(contentMode: .fit)
                .frame(width: 60)
                .foregroundStyle(.tertiary)
                .fontWeight(.ultraLight)

            Text("No jobs in queue")
                .foregroundStyle(.secondary)

            Text("Add a job to get started")
                .font(.caption)
                .foregroundStyle(.tertiary)
            Spacer()
        }
        .frame(maxWidth: .infinity)
    }

    private var jobList: some View {
        List {
            // SfM jobs section
            if !appDataModel.scheduler.sfmJobs.isEmpty {
                Section("SfM Jobs") {
                    ForEach(appDataModel.scheduler.sfmJobs) { job in
                        SfMJobRowView(job: job)
                            .contextMenu {
                                if job.status == .failed {
                                    Button("Retry") {
                                        appDataModel.scheduler.retrySfMJob(job)
                                    }
                                }

                                if job.status == .pending || job.status == .failed || job.status == .cancelled {
                                    Button("Remove", role: .destructive) {
                                        appDataModel.scheduler.removeSfMJob(job)
                                    }
                                }

                                if job.status == .completed {
                                    Button("Open in Finder") {
                                        NSWorkspace.shared.selectFile(nil, inFileViewerRootedAtPath: job.colmapOutputDirectory.path())
                                    }
                                }
                            }
                    }
                }
            }

            // Reconstruction jobs section
            if !appDataModel.scheduler.jobs.isEmpty {
                Section("Reconstruction Jobs") {
                    ForEach(appDataModel.scheduler.jobs) { job in
                        JobRowView(job: job)
                            .contextMenu {
                                if job.status == .pending {
                                    Button("Edit") {
                                        appDataModel.editingJob = job
                                        appDataModel.showingJobSetup = true
                                    }
                                }

                                if job.status == .failed {
                                    Button("Retry") {
                                        appDataModel.scheduler.retryJob(job)
                                    }
                                }

                                if job.status == .pending || job.status == .failed || job.status == .cancelled {
                                    Button("Remove", role: .destructive) {
                                        appDataModel.scheduler.removeJob(job)
                                    }
                                }
                            }
                    }
                    .onMove { source, destination in
                        appDataModel.scheduler.moveJob(from: source, to: destination)
                    }
                }
            }
        }
        .listStyle(.inset(alternatesRowBackgrounds: true))
    }
}

// MARK: - Header

private struct QueueHeaderView: View {
    @Environment(AppDataModel.self) private var appDataModel: AppDataModel

    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text("Job Queue")
                    .font(.headline)

                Text(statusSummary)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Spacer()

            if appDataModel.scheduler.isRunning {
                if appDataModel.scheduler.isPaused {
                    Button("Resume") {
                        appDataModel.scheduler.resume()
                    }
                } else {
                    Button("Pause") {
                        appDataModel.scheduler.pause()
                    }
                }

                Button("Stop") {
                    appDataModel.scheduler.cancel()
                }
                .foregroundStyle(.red)
            } else if appDataModel.scheduler.pendingJobCount > 0 {
                Button("Start") {
                    appDataModel.scheduler.start()
                }
                .buttonStyle(.borderedProminent)
            }
        }
        .padding()
    }

    private var statusSummary: String {
        let scheduler = appDataModel.scheduler
        let total = scheduler.totalJobCount
        let completed = scheduler.completedJobCount
        let pending = scheduler.pendingJobCount

        if scheduler.isRunning {
            if scheduler.isPaused {
                return "Paused — \(completed)/\(total) complete"
            }
            if !scheduler.currentSfMPhase.isEmpty {
                return "\(scheduler.currentSfMPhase) — \(completed)/\(total) complete"
            }
            return "Processing — \(completed)/\(total) complete"
        }

        if total == 0 { return "Empty" }
        if pending == 0 && completed == total { return "All \(total) jobs complete" }
        return "\(pending) pending, \(completed) complete"
    }
}

// MARK: - Footer

private struct QueueFooterView: View {
    @Environment(AppDataModel.self) private var appDataModel: AppDataModel

    var body: some View {
        HStack {
            Button {
                appDataModel.showingScheduleSettings = true
            } label: {
                Label("Schedule", systemImage: "clock")
            }

            if appDataModel.scheduler.scheduleConfig.delayedStart != nil ||
               appDataModel.scheduler.scheduleConfig.hasAllowedWindow {
                Image(systemName: "checkmark.circle.fill")
                    .foregroundStyle(.green)
                    .font(.caption)
            }

            Spacer()

            Button {
                appDataModel.showingSfMJobSetup = true
            } label: {
                Label("Add SfM Job", systemImage: "viewfinder")
            }

            Button {
                appDataModel.editingJob = nil
                appDataModel.showingJobSetup = true
            } label: {
                Label("Add Job", systemImage: "plus")
            }
            .buttonStyle(.borderedProminent)
        }
        .padding()
    }
}
