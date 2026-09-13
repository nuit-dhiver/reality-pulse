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
        #if os(macOS)
        Window("Reality Pulse", id: "main") {
            rootView
                .frame(minWidth: 840, minHeight: 600)
        }
        .defaultSize(width: 960, height: 720)
        #else
        // iPhone and iPad manage their own window size, so the app uses a
        // single scene without a minimum frame.
        WindowGroup {
            rootView
        }
        #endif
    }

    @ViewBuilder
    private var rootView: some View {
        if case .success(let modelContainer) = modelContainerResult {
            ContentView(modelContainerResult: modelContainerResult)
                .modelContainer(modelContainer)
        } else {
            ContentView(modelContainerResult: modelContainerResult)
        }
    }
}
