/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Choose the model file formats to export for a reconstruction job.
*/

import SwiftUI

struct ExportFormatView: View {
    @Environment(JobDraft.self) private var draft: JobDraft

    var body: some View {
        @Bindable var draft = draft

        Section {
            ForEach(ModelExportFormat.allCases) { format in
                Toggle(format.displayName, isOn: Binding(
                    get: { draft.exportFormats.contains(format) },
                    set: { isSelected in
                        if isSelected {
                            draft.exportFormats.insert(format)
                        } else if draft.exportFormats.count > 1 {
                            draft.exportFormats.remove(format)
                        }
                    }
                ))
            }
        } header: {
            Text("Export Formats")
        }
    }
}

