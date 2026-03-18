/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Live preview of the output files that will be generated based on the
current quality and multi-model settings.
*/

import SwiftUI
import RealityKit

struct OutputPreviewView: View {
    @Environment(JobDraft.self) private var draft: JobDraft

    var body: some View {
        let filenames = outputFilenames

        if !filenames.isEmpty {
            GroupBox {
                VStack(alignment: .leading, spacing: 4) {
                    Label("Output (\(filenames.count) \(filenames.count == 1 ? "model" : "models"))",
                          systemImage: "doc.on.doc")
                        .font(.caption)
                        .foregroundStyle(.secondary)

                    ForEach(filenames, id: \.self) { name in
                        HStack(spacing: 4) {
                            Image(systemName: "cube")
                                .font(.caption2)
                                .foregroundStyle(.tertiary)
                            Text(name)
                                .font(.caption)
                                .foregroundStyle(.primary)
                        }
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
    }

    private var outputFilenames: [String] {
        let name = draft.modelName ?? "Model"
        var levels: Set<CodableDetailLevel> = [
            CodableDetailLevel(from: draft.detailLevelOptionUnderQualityMenu)
        ]
        if draft.detailLevelOptionsUnderAdvancedMenu.isSelected {
            if draft.detailLevelOptionsUnderAdvancedMenu.preview { levels.insert(.preview) }
            if draft.detailLevelOptionsUnderAdvancedMenu.reduced { levels.insert(.reduced) }
            if draft.detailLevelOptionsUnderAdvancedMenu.medium { levels.insert(.medium) }
            if draft.detailLevelOptionsUnderAdvancedMenu.full { levels.insert(.full) }
            if draft.detailLevelOptionsUnderAdvancedMenu.raw { levels.insert(.raw) }
        }
        return levels
            .sorted { $0.rawValue < $1.rawValue }
            .map { "\(name)-\($0.rawValue).usdz" }
    }
}
