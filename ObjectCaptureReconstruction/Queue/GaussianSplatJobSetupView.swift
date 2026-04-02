/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Sheet for adding a standalone Gaussian Splat training job to the queue.
*/

import SwiftUI
import os

private let logger = Logger(subsystem: ObjectCaptureReconstructionApp.subsystem,
                            category: "GaussianSplatJobSetupView")

@MainActor @Observable class GaussianSplatJobDraft {
    var imageFolder: URL?
    var outputFolder: URL?
    var importedCOLMAPFolder: URL?
    var jobName: String = ""
    var inputMode: GaussianSplatTrainingInputMode = .runCOLMAPInApp

    var trainingConfiguration: CodableGaussianSplatTrainingConfiguration = CodableGaussianSplatTrainingConfiguration()
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
        if inputMode == .useExistingCOLMAP && importedCOLMAPFolder == nil {
            alertMessage = "COLMAP folder is not selected"
            hasError = true
            return false
        }
        return true
    }

    func toJob() -> GaussianSplatTrainingJob? {
        guard let imageFolder, let outputFolder, !jobName.isEmpty else { return nil }
        return GaussianSplatTrainingJob(
            imageFolder: imageFolder,
            outputFolder: outputFolder,
            jobName: jobName,
            inputMode: inputMode,
            importedCOLMAPFolder: importedCOLMAPFolder,
            trainingConfiguration: trainingConfiguration,
            sfmConfiguration: sfmConfiguration
        )
    }
}

struct GaussianSplatJobSetupView: View {
    @Environment(AppDataModel.self) private var appDataModel: AppDataModel
    @State private var draft = GaussianSplatJobDraft()

    var body: some View {
        VStack(spacing: 0) {
            Text("New Gaussian Splat Job")
                .font(.headline)
                .padding(.top)

            Divider()
                .padding(.top, 8)

            ScrollView {
                VStack(spacing: 16) {
                    datasetSection
                    Divider()
                    trainingSection
                    if draft.inputMode == .runCOLMAPInApp {
                        Divider()
                        SfMOptionsView(configuration: $draft.sfmConfiguration)
                    }
                }
                .padding()
            }

            Divider()

            HStack {
                Button("Cancel") {
                    appDataModel.showingGaussianSplatJobSetup = false
                }

                Spacer()

                Button("Add to Queue") {
                    addToQueue()
                }
                .buttonStyle(.borderedProminent)
            }
            .padding()
        }
        .frame(minWidth: 560, minHeight: 560)
        .alert(draft.alertMessage, isPresented: $draft.hasError) {
            Button("OK") {}
        }
    }

    private var datasetSection: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 12) {
                LabeledContent("Job Name") {
                    TextField("e.g. scene-001-splats", text: $draft.jobName)
                        .textFieldStyle(.roundedBorder)
                        .frame(maxWidth: 220)
                }

                folderRow(
                    title: "Image Folder",
                    url: draft.imageFolder,
                    action: { chooseFolder(for: .image) }
                )

                folderRow(
                    title: "Output Folder",
                    url: draft.outputFolder,
                    action: { chooseFolder(for: .output) }
                )

                VStack(alignment: .leading, spacing: 8) {
                    Text("Training Input")
                        .font(.subheadline)
                        .fontWeight(.medium)

                    Picker("Training Input", selection: $draft.inputMode) {
                        ForEach(GaussianSplatTrainingInputMode.allCases, id: \.self) { mode in
                            Text(mode.displayName).tag(mode)
                        }
                    }
                    .labelsHidden()
                    .pickerStyle(.radioGroup)
                }

                if draft.inputMode == .useExistingCOLMAP {
                    folderRow(
                        title: "COLMAP Folder",
                        url: draft.importedCOLMAPFolder,
                        action: { chooseFolder(for: .importedCOLMAP) }
                    )
                }
            }
        } label: {
            Label("Dataset", systemImage: "folder")
        }
    }

    private var trainingSection: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 12) {
                stepperRow(
                    title: "Training Steps",
                    value: $draft.trainingConfiguration.totalTrainSteps,
                    range: 1_000...200_000,
                    step: 1_000
                )
                stepperRow(
                    title: "Refine Every",
                    value: $draft.trainingConfiguration.refineEvery,
                    range: 50...10_000,
                    step: 50
                )
                stepperRow(
                    title: "Max Resolution",
                    value: $draft.trainingConfiguration.maxResolution,
                    range: 256...4_096,
                    step: 64
                )
                stepperRow(
                    title: "Export Every",
                    value: $draft.trainingConfiguration.exportEvery,
                    range: 100...50_000,
                    step: 100
                )
                stepperRow(
                    title: "Eval Every",
                    value: $draft.trainingConfiguration.evalEvery,
                    range: 100...50_000,
                    step: 100
                )
                stepperRow(
                    title: "SH Degree",
                    value: $draft.trainingConfiguration.shDegree,
                    range: 0...4
                )
                stepperRow(
                    title: "LOD Levels",
                    value: $draft.trainingConfiguration.lodLevels,
                    range: 0...6
                )

                if draft.trainingConfiguration.lodLevels > 0 {
                    stepperRow(
                        title: "LOD Refine Steps",
                        value: $draft.trainingConfiguration.lodRefineSteps,
                        range: 500...25_000,
                        step: 500
                    )
                    stepperRow(
                        title: "LOD Keep %",
                        value: $draft.trainingConfiguration.lodDecimationKeep,
                        range: 1...100
                    )
                    stepperRow(
                        title: "LOD Image Scale %",
                        value: $draft.trainingConfiguration.lodImageScale,
                        range: 1...100
                    )
                }

                Toggle("Enable Rerun Logging", isOn: $draft.trainingConfiguration.rerunEnabled)

                Text("Total iterations: \(draft.trainingConfiguration.totalIterations)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        } label: {
            Label("Training", systemImage: "sparkles")
        }
    }

    private enum FolderTarget {
        case image
        case output
        case importedCOLMAP
    }

    private func folderRow(title: String, url: URL?, action: @escaping () -> Void) -> some View {
        LabeledContent(title) {
            HStack {
                Text(url?.lastPathComponent ?? "Not selected")
                    .foregroundStyle(url == nil ? .secondary : .primary)
                    .lineLimit(1)

                Button("Choose…", action: action)
            }
        }
    }

    private func stepperRow(
        title: String,
        value: Binding<Int>,
        range: ClosedRange<Int>,
        step: Int = 1
    ) -> some View {
        LabeledContent(title) {
            HStack(spacing: 8) {
                Stepper(value: value, in: range, step: step) {
                    EmptyView()
                }
                .labelsHidden()

                Text("\(value.wrappedValue)")
                    .monospacedDigit()
                    .frame(minWidth: 80, alignment: .trailing)
            }
            .frame(maxWidth: 180)
        }
    }

    private func chooseFolder(for target: FolderTarget) {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.canCreateDirectories = true

        switch target {
        case .image:
            panel.message = "Select folder containing training images"
        case .output:
            panel.message = "Select output folder for Gaussian Splat training"
        case .importedCOLMAP:
            panel.message = "Select folder containing existing COLMAP cameras/images/points data"
        }

        if panel.runModal() == .OK, let url = panel.url {
            switch target {
            case .image:
                draft.imageFolder = url
            case .output:
                draft.outputFolder = url
            case .importedCOLMAP:
                draft.importedCOLMAPFolder = url
            }
        }
    }

    private func addToQueue() {
        guard draft.validate() else { return }
        guard let job = draft.toJob() else { return }

        appDataModel.scheduler.addGaussianSplatJob(job)
        appDataModel.showingGaussianSplatJobSetup = false

        logger.log("Added Gaussian Splat job: \(job.jobName)")
    }
}