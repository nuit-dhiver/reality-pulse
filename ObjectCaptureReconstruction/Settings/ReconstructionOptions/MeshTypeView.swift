/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Select the mesh type of the created model.
*/

import SwiftUI
import RealityKit

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
