/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
A row in the job queue list showing SfM job status, phase, and progress.
*/

import SwiftUI

struct SfMJobRowView: View {
    let job: SfMJob
    @Environment(AppDataModel.self) private var appDataModel: AppDataModel

    var body: some View {
        HStack(spacing: 10) {
            statusIcon
                .frame(width: 20)

            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 4) {
                    Image(systemName: "viewfinder")
                        .font(.caption)
                        .foregroundStyle(.blue)
                    Text(job.jobName)
                        .fontWeight(.medium)
                }

                HStack(spacing: 4) {
                    Image(systemName: "folder")
                        .font(.caption2)
                    Text(job.imageFolder.lastPathComponent)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }

                if let summary = job.resultSummary {
                    Text("\(summary.registeredImages)/\(summary.totalImages) images, \(summary.sparsePoints) points")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                } else {
                    Text("SfM • \(job.sfmConfiguration.matcherType.rawValue.capitalized) matching")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
            }

            Spacer()

            if job.status == .running {
                VStack(alignment: .trailing, spacing: 2) {
                    ProgressView(value: job.progress)
                        .frame(width: 80)

                    Text(job.currentPhase.isEmpty ? "\(Int(job.progress * 100))%" : job.currentPhase)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            } else if job.status == .failed {
                Text(job.errorMessage ?? "Failed")
                    .font(.caption2)
                    .foregroundStyle(.red)
                    .lineLimit(1)
                    .frame(maxWidth: 120)
                    .help(job.errorMessage ?? "Failed")
            }
        }
        .padding(.vertical, 4)
    }

    @ViewBuilder
    private var statusIcon: some View {
        switch job.status {
        case .pending:
            Image(systemName: "clock")
                .foregroundStyle(.secondary)
        case .running:
            ProgressView()
                .scaleEffect(0.6)
        case .completed:
            Image(systemName: "checkmark.circle.fill")
                .foregroundStyle(.green)
        case .failed:
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(.red)
        case .cancelled:
            Image(systemName: "xmark.circle")
                .foregroundStyle(.secondary)
        }
    }
}
