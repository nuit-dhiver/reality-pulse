/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
A single row in the job queue list showing status, folder, detail levels, and progress.
*/

import SwiftUI

struct JobRowView: View {
    let job: ReconstructionJob
    @Environment(AppDataModel.self) private var appDataModel: AppDataModel

    var body: some View {
        HStack(spacing: 10) {
            statusIcon
                .frame(width: 20)

            VStack(alignment: .leading, spacing: 2) {
                Text(job.modelName)
                    .fontWeight(.medium)

                HStack(spacing: 4) {
                    Image(systemName: "folder")
                        .font(.caption2)
                    Text(job.imageFolder.lastPathComponent)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }

                Text(detailLevelSummary)
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }

            Spacer()

            if job.status == .running {
                VStack(alignment: .trailing, spacing: 2) {
                    ProgressView(value: job.progress)
                        .frame(width: 80)

                    Text("\(Int(job.progress * 100))%")
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

    private var detailLevelSummary: String {
        let levels = job.allRequestedDetailLevels
            .map { $0.rawValue.capitalized }
            .sorted()
            .joined(separator: ", ")
        return levels.isEmpty ? "No detail levels" : levels
    }
}
