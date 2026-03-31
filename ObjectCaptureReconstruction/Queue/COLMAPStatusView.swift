/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Compact banner showing bundled COLMAP status.
*/

import SwiftUI
import os

private let logger = Logger(subsystem: ObjectCaptureReconstructionApp.subsystem,
                            category: "COLMAPStatusView")

/// Compact banner that shows COLMAP status.
struct COLMAPStatusView: View {
    @Environment(AppDataModel.self) private var appDataModel: AppDataModel

    private var manager: COLMAPManager { appDataModel.scheduler.colmapManager }

    var body: some View {
        Group {
            switch manager.status {
            case .installed(let version):
                installedBanner(version: version)
            case .error(let message):
                errorBanner(message: message)
            }
        }
    }

    // MARK: - States

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
        }
        .padding(.horizontal)
        .padding(.vertical, 6)
        .background(.red.opacity(0.05))
    }
}
