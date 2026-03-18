/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Sheet for adding or editing a reconstruction job.
*/

import SwiftUI
import os

private let logger = Logger(subsystem: ObjectCaptureReconstructionApp.subsystem,
                            category: "JobSetupView")

struct JobSetupView: View {
    @Environment(AppDataModel.self) private var appDataModel: AppDataModel
    @State private var draft: JobDraft

    private let isEditing: Bool

    init(existingJob: ReconstructionJob? = nil) {
        if let job = existingJob {
            _draft = State(initialValue: JobDraft(from: job))
            isEditing = true
        } else {
            _draft = State(initialValue: JobDraft())
            isEditing = false
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            Text(isEditing ? "Edit Job" : "New Job")
                .font(.headline)
                .padding(.top)

            Divider()
                .padding(.top, 8)

            ScrollView {
                SettingsView()
                    .padding()
            }

            Divider()
                .padding(.horizontal, -20)

            ProcessButton(isEditing: isEditing)
                .padding()
        }
        .environment(draft)
        .frame(minWidth: 480, minHeight: 432)
        .alert(draft.alertMessage, isPresented: $draft.hasError) {
            Button("OK") {}
        }
    }
}
