/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Choose the level of detail for the created model. When "Custom" is
selected, inline controls for polygon count, texture maps, format,
and resolution appear directly below the picker.
*/

import SwiftUI
import RealityKit

struct QualityView: View {
    @Environment(JobDraft.self) private var draft: JobDraft

    var body: some View {
        @Bindable var draft = draft

        VStack(alignment: .leading, spacing: 8) {
            Picker("Quality:", selection: $draft.detailLevelOptionUnderQualityMenu) {
                Text("Preview — Fastest, low detail")
                    .tag(RealityFoundation.PhotogrammetrySession.Request.Detail.preview)

                Text("Reduced — Faster, moderate detail")
                    .tag(RealityFoundation.PhotogrammetrySession.Request.Detail.reduced)

                Text("Medium — Balanced speed & detail")
                    .tag(RealityFoundation.PhotogrammetrySession.Request.Detail.medium)

                Text("Full — Slower, high detail")
                    .tag(RealityFoundation.PhotogrammetrySession.Request.Detail.full)

                Text("Raw — Slowest, maximum detail")
                    .tag(RealityFoundation.PhotogrammetrySession.Request.Detail.raw)

                Divider()

                Text("Custom — Set polygon & texture limits")
                    .tag(RealityFoundation.PhotogrammetrySession.Request.Detail.custom)
            }
            .pickerStyle(.menu)

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
        }
        .onChange(of: draft.detailLevelOptionUnderQualityMenu) {
            if draft.detailLevelOptionUnderQualityMenu != .custom {
                draft.sessionConfiguration.customDetailSpecification = PhotogrammetrySession.Configuration.CustomDetailSpecification()
            }
        }
    }
}
