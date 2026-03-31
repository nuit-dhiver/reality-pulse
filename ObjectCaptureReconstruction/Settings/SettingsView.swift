/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Choose the image and model folders, the model name, and the reconstruction options.
*/

import RealityKit
import SwiftUI
import os

private let logger = Logger(subsystem: ObjectCaptureReconstructionApp.subsystem,
                            category: "SettingsView")

struct SettingsView: View {
    @Environment(JobDraft.self) private var draft: JobDraft

    var body: some View {
        @Bindable var draft = draft

        VStack(spacing: 10) {
            Form {
                FolderOptionsView()
            }
            .padding(.top, 8)
            
            Divider()
            
            Form {
                ReconstructionOptionsView()
            }
            .padding(.leading, 13)

            Divider()

            Form {
                Section("Camera Pose Estimation (SfM)") {
                    Toggle("Run SfM before reconstruction", isOn: $draft.runSfMFirst)
                        .help("Run COLMAP Structure from Motion to estimate camera poses and export a sparse point cloud before reconstruction.")

                    if draft.runSfMFirst {
                        SfMOptionsView(configuration: $draft.sfmConfiguration)
                    }
                }
            }
            .padding(.leading, 13)
        }
    }
}
