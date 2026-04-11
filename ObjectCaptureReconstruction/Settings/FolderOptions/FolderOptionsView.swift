/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Choose the image folder, model folder, and the model name.
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
            }
            .pickerStyle(.segmented)
            .onChange(of: draft.inputMode) {
                // Clear input-related state when the mode is switched.
                // Any in-flight video extraction is cancelled via VideoInputView's onDisappear
                // modifier, which fires when the view is removed from the hierarchy here.
                draft.imageFolder = nil
                draft.videoFile = nil
                draft.boundingBoxAvailable = false
            }

            if draft.inputMode == .images {
                ImageFolderView()
            } else {
                VideoInputView()
            }

            ModelNameField()

            ModelFolderView()
        }
    }
}
