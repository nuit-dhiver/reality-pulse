/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Choose an existing USDZ model to convert to the selected export formats
without running a reconstruction.
*/

import SwiftUI
import UniformTypeIdentifiers
import os

private let logger = Logger(subsystem: ObjectCaptureReconstructionApp.subsystem,
                            category: "USDZInputView")

struct USDZInputView: View {
    @Environment(JobDraft.self) private var draft: JobDraft

    @State private var showFileImporter = false
    /// Tracks whether the current `draft.sourceModelFile` URL has an active security scope.
    @State private var isAccessingSecurityScope = false

    var body: some View {
        LabeledContent("USDZ File:") {
            VStack(spacing: 6) {
                HStack {
                    Text(draft.sourceModelFile == nil ? "Drag in a USDZ file" : "Ready to convert")
                        .foregroundStyle(.secondary)
                        .font(.caption)

                    Spacer()

                    if draft.sourceModelFile != nil {
                        Button {
                            clearModel()
                        } label: {
                            Image(systemName: "xmark.circle.fill")
                                .frame(height: 15)
                        }
                        .buttonStyle(.plain)
                        .foregroundStyle(.secondary)
                    }
                }
                .padding([.leading, .trailing], 6)
                .padding(.top, 3)
                .frame(height: 20)

                Divider()
                    .padding(.top, -4)
                    .padding(.horizontal, 6)

                HStack {
                    if let sourceModelFile = draft.sourceModelFile {
                        Image(nsImage: NSWorkspace.shared.icon(forFile: sourceModelFile.path))
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .frame(width: 35)
                    } else {
                        Image(systemName: "cube")
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .frame(width: 28)
                            .foregroundStyle(.tertiary)
                    }
                }
                .frame(height: 35)

                Button {
                    logger.log("Opening an interface for selecting the USDZ file...")
                    showFileImporter.toggle()
                } label: {
                    HStack {
                        if let sourceModelFile = draft.sourceModelFile {
                            Text(sourceModelFile.lastPathComponent)
                        } else {
                            Text("Choose USDZ...")
                        }
                        Spacer()
                    }
                }
                .padding(6)
                .fileImporter(
                    isPresented: $showFileImporter,
                    allowedContentTypes: [.usdz]
                ) { result in
                    switch result {
                    case .success(let url):
                        let gotAccess = url.startAccessingSecurityScopedResource()
                        selectModel(url, releaseScopeOnClear: gotAccess)
                    case .failure(let error):
                        draft.alertMessage = "\(error)"
                        draft.hasError = true
                    }
                }
            }
            .background(Color.gray.opacity(0.1))
            .cornerRadius(10)
        }
        .frame(height: 130)
        .dropDestination(for: URL.self) { items, _ in
            guard let url = items.first, Self.isUSDZFile(url) else {
                logger.info("Dragged item is not a USDZ file.")
                return false
            }
            selectModel(url, releaseScopeOnClear: false)
            return true
        }
        .onDisappear {
            releaseSecurityScope()
        }
    }

    // MARK: - Helpers

    static func isUSDZFile(_ url: URL) -> Bool {
        UTType(filenameExtension: url.pathExtension)?.conforms(to: .usdz) == true
    }

    private func selectModel(_ url: URL, releaseScopeOnClear: Bool) {
        releaseSecurityScope()
        draft.sourceModelFile = url
        isAccessingSecurityScope = releaseScopeOnClear

        if draft.modelName?.isEmpty ?? true {
            draft.modelName = url.deletingPathExtension().lastPathComponent
        }
    }

    private func clearModel() {
        releaseSecurityScope()
        draft.sourceModelFile = nil
    }

    private func releaseSecurityScope() {
        if isAccessingSecurityScope, let url = draft.sourceModelFile {
            url.stopAccessingSecurityScopedResource()
            isAccessingSecurityScope = false
        }
    }
}
