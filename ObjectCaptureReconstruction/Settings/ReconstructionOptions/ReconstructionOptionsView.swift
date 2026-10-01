/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Reconstruction options laid out as distinct sections: quality, multi-model
output, output preview, mesh type, masking, and bounding box.
*/

import SwiftUI

struct ReconstructionOptionsView: View {
    @Environment(JobDraft.self) private var draft: JobDraft

    var body: some View {
        if draft.inputMode == .usdz {
            // Conversion jobs skip reconstruction, so only the export formats apply.
            ExportFormatView()
        } else {
            reconstructionOptions
        }
    }

    @ViewBuilder
    private var reconstructionOptions: some View {
        QualityView()

        Divider()

        MultiModelOutputView()

        ExportFormatView()

        OutputPreviewView()

        Divider()

        MeshTypeView()
        MaskingView()
        IgnoreBoundingBoxView()
    }
}
