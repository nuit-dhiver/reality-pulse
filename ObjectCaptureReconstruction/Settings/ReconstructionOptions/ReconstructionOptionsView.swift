/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Reconstruction options.
*/

import SwiftUI

struct ReconstructionOptionsView: View {
    var body: some View {
        MeshTypeView()
        QualityView()
        MaskingView()
        IgnoreBoundingBoxView()
    }
}
