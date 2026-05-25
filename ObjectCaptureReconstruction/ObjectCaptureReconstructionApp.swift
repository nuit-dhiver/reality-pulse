/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Top-level app structure of the view hierarchy.
*/

import SwiftUI
import SwiftData

@main
struct ObjectCaptureReconstructionApp: App {
    static let subsystem: String = "com.example.apple-samplecode.ObjectCaptureReconstruction"

    private let modelContainerResult: Result<ModelContainer, Error>

    init() {
        modelContainerResult = Result {
            try JobStore.makeModelContainer()
        }
    }

    var body: some Scene {
        Window("Reality Pulse", id: "main") {
            if case .success(let modelContainer) = modelContainerResult {
                ContentView(modelContainerResult: modelContainerResult)
                    .modelContainer(modelContainer)
                    .frame(minWidth: 840, minHeight: 600)
            } else {
                ContentView(modelContainerResult: modelContainerResult)
                    .frame(minWidth: 840, minHeight: 600)
            }
        }
        .defaultSize(width: 960, height: 720)
    }
}
