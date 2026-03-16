/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Choose the level of detail for the created model.
*/

import SwiftUI
import RealityKit

struct QualityView: View {
    @Environment(JobDraft.self) private var draft: JobDraft
    @State private var showAdvancedOptions = false

    var body: some View {
        @Bindable var draft = draft
        
        HStack {
            Picker("Quality:", selection: $draft.detailLevelOptionUnderQualityMenu) {
                Text("Preview")
                    .tag(RealityFoundation.PhotogrammetrySession.Request.Detail.preview)
                
                Text("Reduced")
                    .tag(RealityFoundation.PhotogrammetrySession.Request.Detail.reduced)
                
                Text("Medium")
                    .tag(RealityFoundation.PhotogrammetrySession.Request.Detail.medium)
                
                Text("Full")
                    .tag(RealityFoundation.PhotogrammetrySession.Request.Detail.full)
                
                Text("Raw")
                    .tag(RealityFoundation.PhotogrammetrySession.Request.Detail.raw)
                
                Text("Custom")
                    .tag(RealityFoundation.PhotogrammetrySession.Request.Detail.custom)
            }
            .pickerStyle(.menu)
            
            Button {
                showAdvancedOptions = true
            } label: {
                Text("Advanced...")
            }
        }
        .popover(isPresented: $showAdvancedOptions) {
            AdvancedReconstructionOptions()
        }
        .onChange(of: draft.detailLevelOptionUnderQualityMenu) {
            if draft.detailLevelOptionUnderQualityMenu != .custom {
                draft.sessionConfiguration.customDetailSpecification = PhotogrammetrySession.Configuration.CustomDetailSpecification()
            }
        }
    }
}
