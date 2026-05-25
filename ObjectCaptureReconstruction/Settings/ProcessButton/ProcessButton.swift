/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Provide a button to add the configured job to the queue.
*/

import SwiftUI
import os

private let logger = Logger(subsystem: ObjectCaptureReconstructionApp.subsystem,
                            category: "ProcessButton")

struct ProcessButton: View {
    @Environment(JobDraft.self) private var draft: JobDraft
    @Environment(AppDataModel.self) private var appDataModel: AppDataModel
    @Environment(\.dismiss) private var dismiss

    var isEditing: Bool = false

    var body: some View {
        HStack {
            Button("Cancel") {
                dismiss()
            }

            Spacer()

            Button(isEditing ? "Save Changes" : "Add to Queue") {
                guard draft.validate() else { return }
                let existingJob = isEditing ? appDataModel.editingJob : nil
                guard var job = draft.toJob(
                    existingId: existingJob?.id,
                    createdAt: existingJob?.createdAt ?? Date()
                ) else { return }
                logger.log("Adding job to queue: \(job.modelName)")

                if let editingJob = existingJob {
                    job.status = editingJob.status
                    job.progress = editingJob.progress
                    job.errorMessage = editingJob.errorMessage
                    appDataModel.scheduler.updateJob(job)
                } else {
                    appDataModel.scheduler.addJob(job)
                }

                dismiss()
            }
        }
        .padding(.top, 3)
    }
}
