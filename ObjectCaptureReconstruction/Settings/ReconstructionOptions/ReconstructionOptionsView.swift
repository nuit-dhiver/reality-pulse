/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Reconstruction options laid out as distinct sections: quality, multi-model
output, output preview, mesh type, masking, and bounding box. Options Object
Capture only offers on macOS are left out of the iPhone and iPad builds.
*/

import SwiftUI

struct ReconstructionOptionsView: View {
    var body: some View {
        QualityView()

        Divider()

        if ReconstructionCapability.supportsMultipleDetailLevels {
            MultiModelOutputView()
        }

        ExportFormatView()

        OutputPreviewView()

        Divider()

        #if os(macOS)
        MeshTypeView()
        #endif
        MaskingView()
        IgnoreBoundingBoxView()
    }
}
