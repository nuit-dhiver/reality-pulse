/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Choose the level of detail for the created model. The menu only offers the
levels Object Capture supports on this platform: every level on macOS, and
reduced detail on iPhone and iPad. When "Custom" is selected, inline controls
for polygon count, texture maps, format, and resolution appear directly below
the picker.
*/

import SwiftUI
import RealityKit

struct QualityView: View {
    @Environment(JobDraft.self) private var draft: JobDraft

    var body: some View {
        @Bindable var draft = draft

        VStack(alignment: .leading, spacing: 8) {
            if ReconstructionCapability.supportsMultipleDetailLevels {
                Picker("Quality:", selection: $draft.detailLevelOptionUnderQualityMenu) {
                    ForEach(ReconstructionCapability.supportedDetailLevels, id: \.self) { level in
                        Text(menuTitle(for: level))
                            .tag(level)
                    }
                }
                .pickerStyle(.menu)
            } else {
                LabeledContent("Quality:") {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(menuTitle(for: draft.detailLevelOptionUnderQualityMenu))

                        Text("On-device reconstruction on iPhone and iPad supports this detail level only.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }

            #if os(macOS)
            if draft.detailLevelOptionUnderQualityMenu == .custom {
                GroupBox("Custom Detail Settings") {
                    Form {
                        PolygonCountView()
                        TextureMapsView()
                        TextureFormatView()
                        TextureResolutionView()
                    }
                }
                .padding(.leading, 4)
            }
            #endif
        }
        .onAppear {
            // A job saved on a Mac can arrive with a level this platform cannot
            // produce, so fall back to a level it can.
            if !draft.detailLevelOptionUnderQualityMenu.isSupportedOnThisPlatform {
                draft.detailLevelOptionUnderQualityMenu = ReconstructionCapability.defaultDetailLevel
            }
        }
        .onChange(of: draft.detailLevelOptionUnderQualityMenu) {
            resetCustomDetailSpecificationIfNeeded(draft)
        }
    }

    /// Clears the custom detail settings when the job leaves the custom level.
    /// The custom detail specification only exists on macOS.
    private func resetCustomDetailSpecificationIfNeeded(_ draft: JobDraft) {
        #if os(macOS)
        guard draft.detailLevelOptionUnderQualityMenu != .custom else { return }
        draft.sessionConfiguration.customDetailSpecification =
            PhotogrammetrySession.Configuration.CustomDetailSpecification()
        #endif
    }

    private func menuTitle(for level: CodableDetailLevel) -> String {
        switch level {
        case .preview:
            return "Preview — Fastest, low detail"
        case .reduced:
            return "Reduced — Faster, moderate detail"
        case .medium:
            return "Medium — Balanced speed & detail"
        case .full:
            return "Full — Slower, high detail"
        case .raw:
            return "Raw — Slowest, maximum detail"
        case .custom:
            return "Custom — Set polygon & texture limits"
        }
    }
}
