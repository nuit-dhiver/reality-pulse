/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Choose the output texture maps to include in the output model.
TextureMapOutputs is an OptionSet so multiple maps can be selected.
*/

import SwiftUI
import RealityKit

private typealias TMO = PhotogrammetrySession.Configuration.CustomDetailSpecification.TextureMapOutputs

struct TextureMapsView: View {
    @Environment(JobDraft.self) private var draft: JobDraft

    var body: some View {
        LabeledContent("Texture Maps:") {
            VStack(alignment: .leading, spacing: 4) {
                mapToggle("Diffuse Color", map: .diffuseColor)
                mapToggle("Normal", map: .normal)
                mapToggle("Roughness", map: .roughness)
                mapToggle("Displacement", map: .displacement)
                mapToggle("Ambient Occlusion", map: .ambientOcclusion)

                Divider()

                Toggle("All", isOn: allBinding)
            }
            .toggleStyle(.checkbox)
        }
    }

    private func mapToggle(_ label: String, map: TMO) -> some View {
        Toggle(label, isOn: Binding(
            get: { draft.sessionConfiguration.customDetailSpecification.outputTextureMaps.contains(map) },
            set: { enabled in
                if enabled {
                    draft.sessionConfiguration.customDetailSpecification.outputTextureMaps.insert(map)
                } else {
                    draft.sessionConfiguration.customDetailSpecification.outputTextureMaps.remove(map)
                }
            }
        ))
    }

    private var allBinding: Binding<Bool> {
        Binding(
            get: { draft.sessionConfiguration.customDetailSpecification.outputTextureMaps == .all },
            set: { enabled in
                draft.sessionConfiguration.customDetailSpecification.outputTextureMaps = enabled ? .all : []
            }
        )
    }
}
