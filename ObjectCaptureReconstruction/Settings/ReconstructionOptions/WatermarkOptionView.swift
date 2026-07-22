/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Opt-in provenance watermarking for exported models, with an optional saved key
that can be reused across jobs.
*/

import SwiftUI

struct WatermarkOptionView: View {
    @Environment(JobDraft.self) private var draft: JobDraft
    @Environment(AppDataModel.self) private var appDataModel: AppDataModel

    @State private var savedKeys: [WatermarkKeyInfo] = []
    @State private var isNamingKey = false
    @State private var newKeyLabel = ""
    @State private var keyErrorMessage: String?

    /// Sentinel for "generate a fresh key for every exported file".
    private static let perCopyKeyTag: UUID? = nil

    var body: some View {
        @Bindable var draft = draft

        GroupBox {
            VStack(alignment: .leading, spacing: 8) {
                Label("Provenance", systemImage: "checkmark.seal")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                Toggle("Embed provenance watermark", isOn: $draft.embedWatermark)

                Text("Imperceptibly marks each exported file with a secret key so copies found elsewhere can be traced back to this export. Keys stay on this Mac.")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)

                if draft.embedWatermark {
                    Divider()

                    Picker("Key", selection: $draft.watermarkKeyId) {
                        Text("New key per file").tag(Self.perCopyKeyTag)
                        if !savedKeys.isEmpty {
                            Divider()
                            ForEach(savedKeys) { key in
                                Text(key.label).tag(Optional(key.id))
                            }
                        }
                    }

                    Text(keyExplanation)
                        .font(.caption2)
                        .foregroundStyle(.tertiary)

                    Button("New Saved Key…") {
                        newKeyLabel = ""
                        isNamingKey = true
                    }
                    .controlSize(.small)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .task { savedKeys = appDataModel.scheduler.watermarkKeys() }
        .alert("New Saved Key", isPresented: $isNamingKey) {
            TextField("Label (for example, a client or collection name)", text: $newKeyLabel)
            Button("Create", action: createKey)
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("Every file marked with this key carries the same mark, so a leak traces back to the key rather than to one copy.")
        }
        .alert("Could Not Create Key", isPresented: Binding(
            get: { keyErrorMessage != nil },
            set: { if !$0 { keyErrorMessage = nil } }
        )) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(keyErrorMessage ?? "")
        }
    }

    private var keyExplanation: String {
        if draft.watermarkKeyId == nil {
            return "Each exported file gets its own key, so a leaked copy identifies exactly which export it came from."
        }
        return "All files in this job share the saved key. A leaked copy proves it is yours, but not which copy leaked."
    }

    private func createKey() {
        guard let created = appDataModel.scheduler.createWatermarkKey(label: newKeyLabel) else {
            keyErrorMessage = newKeyLabel.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                ? "Enter a label for the key."
                : "A key named “\(newKeyLabel)” already exists."
            return
        }
        savedKeys = appDataModel.scheduler.watermarkKeys()
        draft.watermarkKeyId = created.id
    }
}
