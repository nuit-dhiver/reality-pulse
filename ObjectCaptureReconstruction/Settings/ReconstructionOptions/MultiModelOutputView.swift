/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Option to generate additional 3D models at other detail levels alongside
the primary quality selection. Each checked level produces a separate
USDZ file.
*/

import SwiftUI
import RealityKit

struct MultiModelOutputView: View {
    @Environment(JobDraft.self) private var draft: JobDraft

    var body: some View {
        @Bindable var draft = draft

        VStack(alignment: .leading, spacing: 6) {
            Toggle("Generate additional models", isOn: $draft.detailLevelOptionsUnderAdvancedMenu.isSelected)

            if draft.detailLevelOptionsUnderAdvancedMenu.isSelected {
                GroupBox {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Also output models at these quality levels:")
                            .font(.caption)
                            .foregroundStyle(.secondary)

                        Toggle("Preview", isOn: $draft.detailLevelOptionsUnderAdvancedMenu.preview)
                        Toggle("Reduced", isOn: $draft.detailLevelOptionsUnderAdvancedMenu.reduced)
                        Toggle("Medium", isOn: $draft.detailLevelOptionsUnderAdvancedMenu.medium)
                        Toggle("Full", isOn: $draft.detailLevelOptionsUnderAdvancedMenu.full)
                        Toggle("Raw", isOn: $draft.detailLevelOptionsUnderAdvancedMenu.raw)
                    }
                    .toggleStyle(.checkbox)
                    .padding(.vertical, 2)
                }
                .padding(.leading, 4)
            }
        }
    }
}
