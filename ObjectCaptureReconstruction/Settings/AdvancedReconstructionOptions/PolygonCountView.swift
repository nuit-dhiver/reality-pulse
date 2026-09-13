/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Set the upper limit on polygons in the model mesh.
The custom detail specification is macOS only, so this control is left out of
the iPhone and iPad builds.
*/

import SwiftUI
import RealityKit

#if os(macOS)

struct PolygonCountView: View {
    @Environment(JobDraft.self) private var draft: JobDraft

    var body: some View {
        @Bindable var draft = draft

        LabeledContent("Max Polygon Count:") {
            TextField("", value: $draft.sessionConfiguration.customDetailSpecification.maximumPolygonCount, formatter: NumberFormatter())
                .textFieldStyle(.roundedBorder)
        }
    }
}
#endif
