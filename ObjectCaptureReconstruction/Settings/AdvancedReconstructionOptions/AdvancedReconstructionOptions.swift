/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Choose customizable options on the reconstructed model and textures and process multiple detail levels.
*/

import SwiftUI
import RealityKit

struct AdvancedReconstructionOptions: View {
    @Environment(JobDraft.self) private var draft: JobDraft

    var body: some View {
        @Bindable var draft = draft

        VStack(alignment: .leading) {
            Text("Advanced Reconstruction Options")
            Divider()
            Form {
                PolygonCountView()
                
                TextureMapsView()
                
                TextureFormatView()
                
                TextureResolutionView()
                
                Divider()
                
                LabeledContent("Processing") {
                    Toggle("Process multiple detail levels", isOn: $draft.detailLevelOptionsUnderAdvancedMenu.isSelected)
                }
                
                Menu("Choose detail levels") {
                    Toggle("Preview", isOn: $draft.detailLevelOptionsUnderAdvancedMenu.preview)

                    Toggle("Reduced", isOn: $draft.detailLevelOptionsUnderAdvancedMenu.reduced)

                    Toggle("Medium", isOn: $draft.detailLevelOptionsUnderAdvancedMenu.medium)

                    Toggle("Full", isOn: $draft.detailLevelOptionsUnderAdvancedMenu.full)

                    Toggle("Raw", isOn: $draft.detailLevelOptionsUnderAdvancedMenu.raw)

                }
                .disabled(!draft.detailLevelOptionsUnderAdvancedMenu.isSelected)
            }
        }
        .padding()
    }
}
