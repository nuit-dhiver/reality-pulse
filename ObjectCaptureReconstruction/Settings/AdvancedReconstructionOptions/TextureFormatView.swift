/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Choose the output format to use for all textures.
The custom detail specification is macOS only, so this control is left out of
the iPhone and iPad builds.
*/

import SwiftUI
import RealityKit

#if os(macOS)

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
    }
}
#endif
