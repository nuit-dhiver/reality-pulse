/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Option to isolate the object or to include the environment in the created model.
*/

import SwiftUI
import RealityKit

struct MaskingView: View {
    @Environment(JobDraft.self) private var draft: JobDraft
    @State private var selectedMaskOption = MaskingOption.isolateFromEnvironment

    var body: some View {
        Picker("Masking:", selection: $selectedMaskOption) {
            ForEach(MaskingOption.allCases, id: \.self) { option in
                HStack {
                    Image(getImageName(for: option))
                    Text(option.rawValue)
                }
            }
        }
        .onChange(of: selectedMaskOption, initial: false) {
            switch selectedMaskOption {
            case .isolateFromEnvironment:
                draft.sessionConfiguration.isObjectMaskingEnabled = true
            case .includeEnvironment:
                draft.sessionConfiguration.isObjectMaskingEnabled = false
            }
        }
        .onAppear {
            if draft.sessionConfiguration.isObjectMaskingEnabled {
                selectedMaskOption = .isolateFromEnvironment
            } else {
                selectedMaskOption = .includeEnvironment
            }
        }
    }

    private enum MaskingOption: String, CaseIterable {
        case isolateFromEnvironment = "Isolate object from environment"
        case includeEnvironment = "Include environment around object"
    }

    private func getImageName(for maskingOption: MaskingOption) -> String {
        switch maskingOption {
        case .isolateFromEnvironment:
            return "IsolateFromEnvironment"
        case .includeEnvironment:
            return "IncludeEnvironment"
        }
    }
}
