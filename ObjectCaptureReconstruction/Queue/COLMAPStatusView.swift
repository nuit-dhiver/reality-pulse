/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Banner view showing COLMAP installation status with download/install action.
*/

import SwiftUI
import os

private let logger = Logger(subsystem: ObjectCaptureReconstructionApp.subsystem,
                            category: "COLMAPStatusView")

/// Compact banner that shows COLMAP status and provides download/install controls.
struct COLMAPStatusView: View {
    @Environment(AppDataModel.self) private var appDataModel: AppDataModel
    @State private var isDownloading = false
    @State private var downloadError: String?

    private var manager: COLMAPManager { appDataModel.scheduler.colmapManager }

    var body: some View {
        Group {
            switch manager.status {
            case .notInstalled:
                notInstalledBanner
            case .downloading(let progress):
                downloadingBanner(progress: progress)
            case .installed(let version):
                installedBanner(version: version)
            case .error(let message):
                errorBanner(message: message)
            }
        }
    }

    // MARK: - States

    private var notInstalledBanner: some View {
        HStack(spacing: 8) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(.orange)

            Text("COLMAP is not installed")
                .font(.caption)
                .foregroundStyle(.secondary)

            Spacer()

            Button("Install COLMAP") {
                installCOLMAP()
            }
            .font(.caption)
            .buttonStyle(.borderedProminent)
            .controlSize(.small)
            .disabled(isDownloading)
        }
        .padding(.horizontal)
        .padding(.vertical, 6)
        .background(.orange.opacity(0.08))
    }

    private func downloadingBanner(progress: Double) -> some View {
        HStack(spacing: 8) {
            ProgressView()
                .controlSize(.small)

            Text("Downloading COLMAP…")
                .font(.caption)
                .foregroundStyle(.secondary)

            ProgressView(value: progress)
                .frame(maxWidth: 120)

            Text("\(Int(progress * 100))%")
                .font(.caption)
                .monospacedDigit()
                .foregroundStyle(.secondary)

            Spacer()
        }
        .padding(.horizontal)
        .padding(.vertical, 6)
        .background(.blue.opacity(0.05))
    }

    private func installedBanner(version: String) -> some View {
        HStack(spacing: 8) {
            Image(systemName: "checkmark.circle.fill")
                .foregroundStyle(.green)

            Text("COLMAP installed (\(version))")
                .font(.caption)
                .foregroundStyle(.secondary)

            Spacer()
        }
        .padding(.horizontal)
        .padding(.vertical, 6)
    }

    private func errorBanner(message: String) -> some View {
        HStack(spacing: 8) {
            Image(systemName: "xmark.circle.fill")
                .foregroundStyle(.red)

            Text(message)
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(2)

            Spacer()

            Button("Retry") {
                installCOLMAP()
            }
            .font(.caption)
            .controlSize(.small)
        }
        .padding(.horizontal)
        .padding(.vertical, 6)
        .background(.red.opacity(0.05))
    }

    // MARK: - Actions

    private func installCOLMAP() {
        isDownloading = true
        downloadError = nil

        Task {
            do {
                try await manager.download()
                logger.log("COLMAP download complete.")
            } catch {
                logger.error("COLMAP download failed: \(error)")
                downloadError = error.localizedDescription
            }
            isDownloading = false
        }
    }
}
