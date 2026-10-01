/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Choose the input (image folder, video, or USDZ file), model folder, and the model name.
*/

import SwiftUI
import RealityKit

struct FolderOptionsView: View {
    @Environment(JobDraft.self) private var draft: JobDraft

    var body: some View {
        @Bindable var draft = draft

        Section {
            Picker("Input Type:", selection: $draft.inputMode) {
                Text("Image Folder").tag(JobDraft.InputMode.images)
                Text("Video File").tag(JobDraft.InputMode.video)
                Text("USDZ File").tag(JobDraft.InputMode.usdz)
            }
            .pickerStyle(.segmented)
            .onChange(of: draft.inputMode) {
                // Clear input-related state when the mode is switched.
                // Any in-flight video extraction started by VideoInputView is automatically
                // cancelled when VideoInputView's onDisappear fires as a result of this
                // mode change (see VideoInputView.onDisappear for that cleanup).
                draft.imageFolder = nil
                draft.videoFile = nil
                draft.sourceModelFile = nil
                draft.boundingBoxAvailable = false
            }

            switch draft.inputMode {
            case .images:
                ImageFolderView()
            case .video:
                VideoInputView()
            case .usdz:
                USDZInputView()
            }

            ModelNameField()

            ModelFolderView()
        }
    }
}
