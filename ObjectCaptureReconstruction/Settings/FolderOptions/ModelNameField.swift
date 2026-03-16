/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Choose the model name for the created USDZ file.
*/

import SwiftUI

struct ModelNameField: View {
    @Environment(JobDraft.self) private var draft: JobDraft
    @State private var modelName: String = ""

    var body: some View {
        LabeledContent("Model Name:") {
            TextField("", text: $modelName, prompt: Text("Name"))
                .textFieldStyle(.roundedBorder).padding(.leading, -9)
        }
        .onChange(of: modelName) {
            if !modelName.isEmpty {
                draft.modelName = modelName
            } else {
                draft.modelName = nil
            }
        }
        .onAppear {
            guard let name = draft.modelName else { return }
            modelName = name
        }
    }
}
