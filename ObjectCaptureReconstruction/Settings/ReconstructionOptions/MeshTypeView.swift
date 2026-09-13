/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Select the mesh type of the created model.
Object Capture only exposes the mesh primitive on macOS, so this control is
left out of the iPhone and iPad builds.
*/

import SwiftUI
import RealityKit

#if os(macOS)

struct MeshTypeView: View {
    @Environment(JobDraft.self) private var draft: JobDraft

    var body: some View {
        @Bindable var draft = draft
        
        Picker("Mesh Type:", selection: $draft.sessionConfiguration.meshPrimitive) {
            Text("Triangular Mesh")
                .tag(PhotogrammetrySession.Configuration.MeshPrimitive.triangle)
            
            Text("Quad Mesh")
                .tag(PhotogrammetrySession.Configuration.MeshPrimitive.quad)
        }
        .pickerStyle(.menu)
    }
}
#endif
