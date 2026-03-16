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
    var body: some View {
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
        }
    }
}
