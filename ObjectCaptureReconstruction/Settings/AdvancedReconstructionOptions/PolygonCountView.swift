/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Set the upper limit on polygons in the model mesh.
*/

import SwiftUI
import RealityKit

struct PolygonCountView: View {
    @Environment(JobDraft.self) private var draft: JobDraft

    var body: some View {
        @Bindable var draft = draft

        LabeledContent("Max Polygon Count:") {
            TextField("", value: $draft.sessionConfiguration.customDetailSpecification.maximumPolygonCount, formatter: NumberFormatter())
                .textFieldStyle(.roundedBorder)
        }
        .onChange(of: draft.sessionConfiguration.customDetailSpecification.maximumPolygonCount, initial: false) {
            draft.detailLevelOptionUnderQualityMenu = .custom
        }
    }
}
