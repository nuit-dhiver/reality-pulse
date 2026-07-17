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
    @State private var isExporting = false
    @State private var exportErrorMessage: String?

    var body: some View {
        VStack(spacing: 0) {
            // Header with queue controls
            QueueHeaderView()

            Divider()

            // Job list
            if appDataModel.scheduler.jobs.isEmpty {
                emptyState
            } else {
                jobList
            }

            Divider()

            // Footer with add/schedule buttons
            QueueFooterView()
        }
        .alert("Export Failed", isPresented: Binding(
            get: { exportErrorMessage != nil },
            set: { if !$0 { exportErrorMessage = nil } }
        )) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(exportErrorMessage ?? "")
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
            ForEach(appDataModel.scheduler.jobs) { job in
                JobRowView(job: job)
                    .contextMenu {
                        Button("Show in Finder") {
                            showInFinder(job)
                        }

                        if hasCompletedOutputs(job) {
                            Menu("Export As") {
                                Button("glTF (.gltf)") {
                                    exportJob(job, format: .gltf)
                                }
                                .disabled(isExporting)

                                Button("glb (.glb)") {
                                    exportJob(job, format: .glb)
                                }
                                .disabled(isExporting)
                            }

                            // Internal tooling, hidden unless enabled via
                            // `defaults write <bundle-id> RPWatermarkRecordExport -bool YES`.
                            // Record JSONs contain the per-copy secret keys.
                            if UserDefaults.standard.bool(forKey: "RPWatermarkRecordExport") {
                                Button("Export Provenance Records…") {
                                    exportProvenanceRecords(for: job)
                                }
                            }
                        }

                        Divider()

                        if job.status == .pending {
                            Button("Edit") {
                                appDataModel.editingJob = job
                                appDataModel.showingJobSetup = true
                            }
                        }

                        if job.status == .failed || job.status == .interrupted {
                            Button("Retry") {
                                appDataModel.scheduler.retryJob(job)
                            }
                        }

                        if job.status != .running {
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
        .listStyle(.inset(alternatesRowBackgrounds: true))
    }

    private func showInFinder(_ job: ReconstructionJob) {
        var job = job
        let (_, modelURL) = job.resolveBookmarks()
        let folderURL = modelURL ?? job.modelFolder
        let didAccess = folderURL.startAccessingSecurityScopedResource()

        defer {
            if didAccess {
                folderURL.stopAccessingSecurityScopedResource()
            }
        }

        let outputURLs = job.requestedDetailLevels
            .map { job.outputURL(for: $0) }
            .filter { FileManager.default.fileExists(atPath: $0.path) }

        if outputURLs.isEmpty {
            NSWorkspace.shared.selectFile(nil, inFileViewerRootedAtPath: folderURL.path)
        } else {
            NSWorkspace.shared.activateFileViewerSelecting(outputURLs)
        }
    }

    private func hasCompletedOutputs(_ job: ReconstructionJob) -> Bool {
        job.requestedDetailLevels.contains {
            job.hasCompletedOutputFile(for: $0)
        }
    }

    private func exportJob(_ job: ReconstructionJob, format: ModelExportFormat) {
        guard !isExporting else { return }
        isExporting = true

        Task {
            var workingJob = job
            let (_, modelURL) = workingJob.resolveBookmarks()
            let folderURL = modelURL ?? workingJob.modelFolder
            let didAccess = folderURL.startAccessingSecurityScopedResource()

            defer {
                if didAccess {
                    folderURL.stopAccessingSecurityScopedResource()
                }
                isExporting = false
            }

            do {
                let exportedFiles = try ModelExportService.exportCompletedOutputs(
                    for: workingJob,
                    formats: [format],
                    embedWatermark: workingJob.isWatermarkEnabled
                )
                guard !exportedFiles.isEmpty else {
                    exportErrorMessage = "No completed USDZ outputs were available to export."
                    return
                }
                appDataModel.scheduler.saveExportRecords(exportedFiles.compactMap(\.record))
                NSWorkspace.shared.activateFileViewerSelecting(exportedFiles.map(\.url))
            } catch {
                exportErrorMessage = error.localizedDescription
            }
        }
    }

    /// Internal: dump this job's provenance records as JSON for the offline
    /// `watermark-verify` tool. Each file contains a per-copy secret key.
    private func exportProvenanceRecords(for job: ReconstructionJob) {
        let records = appDataModel.scheduler.exportRecords(jobId: job.id)
        guard !records.isEmpty else {
            exportErrorMessage = "No provenance records exist for this job."
            return
        }

        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.canCreateDirectories = true
        panel.prompt = "Export Records"
        panel.message = "Choose a folder for the record files. They contain secret watermark keys — keep them private."
        guard panel.runModal() == .OK, let directory = panel.url else { return }

        do {
            for record in records {
                let url = directory.appending(path: "\(record.filename).wmrecord.json")
                try record.jsonData().write(to: url, options: [.atomic])
            }
            NSWorkspace.shared.activateFileViewerSelecting([directory])
        } catch {
            exportErrorMessage = "Failed to write records: \(error.localizedDescription)"
        }
    }
}

// MARK: - Header

private struct QueueHeaderView: View {
    @Environment(AppDataModel.self) private var appDataModel: AppDataModel
    
    private var scheduler: JobScheduler {
        appDataModel.scheduler
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                Text("Job Queue")
                    .font(.headline)

                Text(statusSummary)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                }

                Spacer()

                if scheduler.isRunning {
                    if scheduler.isPaused || scheduler.isPauseRequested {
                        Button("Resume") {
                            scheduler.resume()
                        }
                    } else {
                        Button(pauseButtonTitle) {
                            scheduler.pause()
                        }
                    }

                    Button("Stop") {
                        scheduler.cancel()
                    }
                    .foregroundStyle(.red)
                } else if scheduler.pendingJobCount > 0 {
                    Button("Start") {
                        scheduler.start()
                    }
                    .buttonStyle(.borderedProminent)
                }
            }

            if let pauseExplanation {
                Text(pauseExplanation)
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
        }
        .padding()
    }

    private var statusSummary: String {
        let total = scheduler.jobs.count
        let completed = scheduler.completedJobCount
        let pending = scheduler.pendingJobCount
        let needsAttention = scheduler.jobs.filter {
            $0.status == .failed || $0.status == .interrupted
        }.count

        if scheduler.isRunning {
            if scheduler.isPaused {
                return "Queue paused between jobs — \(completed)/\(total) complete"
            }
            if scheduler.isPauseRequested {
                return "Pause requested — current job will finish before stopping"
            }
            return "Processing — \(completed)/\(total) complete"
        }

        if total == 0 { return "Empty" }
        if pending == 0 && completed == total { return "All \(total) jobs complete" }
        if needsAttention > 0 {
            return "\(pending) pending, \(completed) complete, \(needsAttention) need attention"
        }
        return "\(pending) pending, \(completed) complete"
    }

    private var pauseButtonTitle: String {
        scheduler.currentJobId == nil ? "Pause" : "Pause After Job"
    }

    private var pauseExplanation: String? {
        if scheduler.isPauseRequested {
            return "Apple Object Capture does not support pausing an in-progress reconstruction. The queue will stop before the next job starts."
        }

        if scheduler.isRunning && scheduler.currentJobId != nil {
            return "Pausing takes effect between jobs only. If the schedule window closes mid-job, the active reconstruction continues until it completes or is cancelled."
        }

        return nil
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
