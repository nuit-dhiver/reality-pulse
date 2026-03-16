/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Option to ignore the iOS bounding box during reconstruction.
*/

import SwiftUI
import RealityKit

struct IgnoreBoundingBoxView: View {
    @Environment(JobDraft.self) private var draft: JobDraft

    var body: some View {
        @Bindable var draft = draft
        
        LabeledContent("Crop:") {
            Toggle("Ignore iOS bounding box", isOn: $draft.sessionConfiguration.ignoreBoundingBox)
        }
        .disabled(!draft.boundingBoxAvailable)
    }
}
