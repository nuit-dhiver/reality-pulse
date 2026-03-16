/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Choose the maximum dimension of the reconstructed texture maps.
*/

import SwiftUI
import RealityKit

struct TextureResolutionView: View {
    @Environment(JobDraft.self) private var draft: JobDraft

    var body: some View {
        @Bindable var draft = draft
 
        Picker("Texture Resolution:", selection: $draft.sessionConfiguration.customDetailSpecification.maximumTextureDimension) {
            Text("1K")
                .tag(PhotogrammetrySession.Configuration.CustomDetailSpecification.TextureDimension.oneK)
            
            Text("2K")
                .tag(PhotogrammetrySession.Configuration.CustomDetailSpecification.TextureDimension.twoK)
            
            Text("4K")
                .tag(PhotogrammetrySession.Configuration.CustomDetailSpecification.TextureDimension.fourK)
            
            Text("8K")
                .tag(PhotogrammetrySession.Configuration.CustomDetailSpecification.TextureDimension.eightK)
            
            Text("16K")
                .tag(PhotogrammetrySession.Configuration.CustomDetailSpecification.TextureDimension.sixteenK)
        }
        .onChange(of: draft.sessionConfiguration.customDetailSpecification.maximumTextureDimension, initial: false) {
            draft.detailLevelOptionUnderQualityMenu = .custom
        }
    }
}
