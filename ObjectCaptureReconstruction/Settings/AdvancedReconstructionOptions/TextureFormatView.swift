/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Choose the output format to use for all textures.
*/

import SwiftUI
import RealityKit

struct TextureFormatView: View {
    @Environment(JobDraft.self) private var draft: JobDraft

    var body: some View {
        @Bindable var draft = draft

        Picker("Texture Format:", selection: $draft.sessionConfiguration.customDetailSpecification.textureFormat) {
            Text("PNG")
                .tag(PhotogrammetrySession.Configuration.CustomDetailSpecification.TextureFormat.png)
            
            Text("JPEG")
                .tag(PhotogrammetrySession.Configuration.CustomDetailSpecification.TextureFormat.jpeg(compressionQuality: 0.8))
        }
        .onChange(of: draft.sessionConfiguration.customDetailSpecification.textureFormat) {
            draft.detailLevelOptionUnderQualityMenu = .custom
        }
    }
}
