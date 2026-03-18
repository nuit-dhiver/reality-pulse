/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Reconstruction options laid out as distinct sections: quality, multi-model
output, output preview, mesh type, masking, and bounding box.
*/

import SwiftUI

struct ReconstructionOptionsView: View {
    var body: some View {
        QualityView()

        Divider()

        MultiModelOutputView()

        OutputPreviewView()

        Divider()

        MeshTypeView()
        MaskingView()
        IgnoreBoundingBoxView()
    }
}
