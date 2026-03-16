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
                guard let job = draft.toJob() else { return }
                logger.log("Adding job to queue: \(job.modelName)")

                if isEditing, let editingJob = appDataModel.editingJob {
                    var updatedJob = job
                    updatedJob.status = editingJob.status
                    appDataModel.scheduler.updateJob(updatedJob)
                } else {
                    appDataModel.scheduler.addJob(job)
                }

                dismiss()
            }
        }
        .padding(.top, 3)
    }
}
