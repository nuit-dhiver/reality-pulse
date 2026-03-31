/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Sheet for adding a standalone SfM (Structure from Motion) job to the queue.
*/

import SwiftUI
import os

private let logger = Logger(subsystem: ObjectCaptureReconstructionApp.subsystem,
                            category: "SfMJobSetupView")

/// Observable draft for editing an SfM job before adding to the queue.
@MainActor @Observable class SfMJobDraft {
    var imageFolder: URL?
    var outputFolder: URL?
    var jobName: String = ""
    var sfmConfiguration: CodableSfMConfiguration = CodableSfMConfiguration()

    var alertMessage: String = ""
    var hasError: Bool = false

    func validate() -> Bool {
        if imageFolder == nil {
            alertMessage = "Image folder is not selected"
            hasError = true
            return false
        }
        if jobName.isEmpty {
            alertMessage = "Job name is not entered"
            hasError = true
            return false
        }
        if outputFolder == nil {
            alertMessage = "Output folder is not selected"
            hasError = true
            return false
        }
        return true
    }

    func toJob() -> SfMJob? {
        guard let imageFolder, let outputFolder, !jobName.isEmpty else { return nil }
        return SfMJob(
            imageFolder: imageFolder,
            outputFolder: outputFolder,
            jobName: jobName,
            sfmConfiguration: sfmConfiguration
        )
    }
}

struct SfMJobSetupView: View {
    @Environment(AppDataModel.self) private var appDataModel: AppDataModel
    @State private var draft = SfMJobDraft()

    var body: some View {
        VStack(spacing: 0) {
            Text("New SfM Job")
                .font(.headline)
                .padding(.top)

            Divider()
                .padding(.top, 8)

            ScrollView {
                VStack(spacing: 16) {
                    folderSection
                    Divider()
                    SfMOptionsView(configuration: $draft.sfmConfiguration)
                }
                .padding()
            }

            Divider()

            HStack {
                Button("Cancel") {
                    appDataModel.showingSfMJobSetup = false
                }

                Spacer()

                Button("Add to Queue") {
                    addToQueue()
                }
                .buttonStyle(.borderedProminent)
            }
            .padding()
        }
        .frame(minWidth: 480, minHeight: 432)
        .alert(draft.alertMessage, isPresented: $draft.hasError) {
            Button("OK") {}
        }
    }

    // MARK: - Folder selection

    private var folderSection: some View {
        Form {
            LabeledContent("Job Name") {
                TextField("e.g. scene-001", text: $draft.jobName)
                    .textFieldStyle(.roundedBorder)
                    .frame(maxWidth: 200)
            }

            LabeledContent("Image Folder") {
                HStack {
                    Text(draft.imageFolder?.lastPathComponent ?? "Not selected")
                        .foregroundStyle(draft.imageFolder == nil ? .secondary : .primary)
                        .lineLimit(1)

                    Button("Choose…") {
                        chooseFolder(for: .image)
                    }
                }
            }

            LabeledContent("Output Folder") {
                HStack {
                    Text(draft.outputFolder?.lastPathComponent ?? "Not selected")
                        .foregroundStyle(draft.outputFolder == nil ? .secondary : .primary)
                        .lineLimit(1)

                    Button("Choose…") {
                        chooseFolder(for: .output)
                    }
                }
            }
        }
    }

    // MARK: - Actions

    private enum FolderTarget { case image, output }

    private func chooseFolder(for target: FolderTarget) {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.canCreateDirectories = true

        switch target {
        case .image:
            panel.message = "Select folder containing images for SfM"
        case .output:
            panel.message = "Select output folder for SfM results"
        }

        if panel.runModal() == .OK, let url = panel.url {
            switch target {
            case .image: draft.imageFolder = url
            case .output: draft.outputFolder = url
            }
        }
    }

    private func addToQueue() {
        guard draft.validate() else { return }
        guard let job = draft.toJob() else { return }

        appDataModel.scheduler.addSfMJob(job)
        appDataModel.showingSfMJobSetup = false

        logger.log("Added SfM job: \(job.jobName)")
    }
}
