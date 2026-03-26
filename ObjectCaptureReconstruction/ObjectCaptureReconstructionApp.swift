/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Top-level app structure of the view hierarchy.
*/

import SwiftUI

@main
struct ObjectCaptureReconstructionApp: App {
    static let subsystem: String = "com.realitypulse.app"

    var body: some Scene {
        Window("Reality Pulse", id: "main") {
            ContentView()
                .frame(minWidth: 840, minHeight: 600)
        }
        .defaultSize(width: 960, height: 720)
    }
}
