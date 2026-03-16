/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Choose the image folder.
*/

import SwiftUI
import os

private let logger = Logger(subsystem: ObjectCaptureReconstructionApp.subsystem,
                            category: "ImageFolderSelectionView")

struct ImageFolderSelectionView: View {
    @Binding var selectedFolder: URL?

    @Environment(JobDraft.self) private var draft: JobDraft
    @State private var showFileImporter = false

    var body: some View {
        Button {
            logger.log("Opening an interface for selecting the image folder...")
            showFileImporter.toggle()
        } label: {
            HStack {
                if let selectedFolder = selectedFolder {
                    HStack {
                        Image(nsImage: NSWorkspace.shared.icon(for: .folder))
                            .resizable()
                            .aspectRatio(contentMode: .fit)

                        Text("\(selectedFolder.lastPathComponent)")
                    }
                } else {
                    Text("Choose...")
                }
                Spacer()
            }
        }
        .padding(6)
        .fileImporter(isPresented: $showFileImporter,
                      allowedContentTypes: [.folder]) { result in
            switch result {
            case .success(let directory):
                let gotAccess = directory.startAccessingSecurityScopedResource()
                if !gotAccess { return }
                selectedFolder = directory
            case .failure(let error):
                draft.alertMessage = "\(error)"
                draft.hasError = true
            }
        }

    }
}
