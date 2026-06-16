/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Optional glTF and glb export formats generated from each completed USDZ output.
*/

import SwiftUI

struct ExportFormatView: View {
    @Environment(JobDraft.self) private var draft: JobDraft

    var body: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 8) {
                Label("Additional Export Formats", systemImage: "square.and.arrow.up")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                Text("Converted from USDZ after reconstruction completes.")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)

                ForEach(ModelExportFormat.allCases, id: \.self) { format in
                    Toggle(isOn: binding(for: format)) {
                        Text(format.displayName)
                    }
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private func binding(for format: ModelExportFormat) -> Binding<Bool> {
        Binding(
            get: { draft.exportFormats.contains(format) },
            set: { isSelected in
                if isSelected {
                    draft.exportFormats.insert(format)
                } else {
                    draft.exportFormats.remove(format)
                }
            }
        )
    }
}
