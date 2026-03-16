/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Top-level app structure of the view hierarchy.
*/

import SwiftUI

@main
struct ObjectCaptureReconstructionApp: App {
    static let subsystem: String = "com.example.apple-samplecode.ObjectCaptureReconstruction"

    var body: some Scene {
        Window("Reality Pulse", id: "main") {
            ContentView()
                .frame(minWidth: 700, minHeight: 500)
        }
        .defaultSize(width: 800, height: 600)
    }
}
