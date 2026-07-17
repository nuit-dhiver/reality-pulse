/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Opt-in provenance watermarking for exported models.
*/

import SwiftUI

struct WatermarkOptionView: View {
    @Environment(JobDraft.self) private var draft: JobDraft

    var body: some View {
        @Bindable var draft = draft

        GroupBox {
            VStack(alignment: .leading, spacing: 8) {
                Label("Provenance", systemImage: "checkmark.seal")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                Toggle("Embed provenance watermark", isOn: $draft.embedWatermark)

                Text("Imperceptibly marks each exported file with a per-copy secret key so copies found elsewhere can be traced to this export. Keys stay on this Mac.")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}
