/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Option to generate additional 3D models at other detail levels alongside
the primary quality selection. Each checked level produces a separate
USDZ file. The primary quality level is shown but disabled since it is
always included.
*/

import SwiftUI
import RealityKit

struct MultiModelOutputView: View {
    @Environment(JobDraft.self) private var draft: JobDraft

    var body: some View {
        @Bindable var draft = draft

        let primary = CodableDetailLevel(from: draft.detailLevelOptionUnderQualityMenu)

        VStack(alignment: .leading, spacing: 6) {
            Toggle("Generate additional models", isOn: $draft.detailLevelOptionsUnderAdvancedMenu.isSelected)

            if draft.detailLevelOptionsUnderAdvancedMenu.isSelected {
                GroupBox {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Also output models at these quality levels:")
                            .font(.caption)
                            .foregroundStyle(.secondary)

                        levelToggle("Preview", level: .preview, primary: primary,
                                    isOn: $draft.detailLevelOptionsUnderAdvancedMenu.preview)
                        levelToggle("Reduced", level: .reduced, primary: primary,
                                    isOn: $draft.detailLevelOptionsUnderAdvancedMenu.reduced)
                        levelToggle("Medium", level: .medium, primary: primary,
                                    isOn: $draft.detailLevelOptionsUnderAdvancedMenu.medium)
                        levelToggle("Full", level: .full, primary: primary,
                                    isOn: $draft.detailLevelOptionsUnderAdvancedMenu.full)
                        levelToggle("Raw", level: .raw, primary: primary,
                                    isOn: $draft.detailLevelOptionsUnderAdvancedMenu.raw)
                    }
                    .toggleStyle(.checkbox)
                    .padding(.vertical, 2)
                }
                .padding(.leading, 4)
            }
        }
        .onChange(of: draft.detailLevelOptionUnderQualityMenu) {
            // Uncheck the new primary level if it was selected as an additional level.
            let newPrimary = CodableDetailLevel(from: draft.detailLevelOptionUnderQualityMenu)
            clearAdditionalLevel(newPrimary, in: &draft.detailLevelOptionsUnderAdvancedMenu)
        }
    }

    @ViewBuilder
    private func levelToggle(_ label: String, level: CodableDetailLevel, primary: CodableDetailLevel,
                             isOn: Binding<Bool>) -> some View {
        if level == primary {
            Toggle(label + " (primary)", isOn: .constant(false))
                .disabled(true)
                .foregroundStyle(.secondary)
        } else {
            Toggle(label, isOn: isOn)
        }
    }

    private func clearAdditionalLevel(_ level: CodableDetailLevel,
                                      in options: inout CodableDetailLevelOptions) {
        switch level {
        case .preview: options.preview = false
        case .reduced: options.reduced = false
        case .medium:  options.medium  = false
        case .full:    options.full    = false
        case .raw:     options.raw     = false
        case .custom:  break
        }
    }
}
