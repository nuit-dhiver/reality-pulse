/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Supported model export formats for reconstruction jobs.
*/

import Foundation

enum ModelExportFormat: String, Codable, CaseIterable, Hashable, Identifiable {
    case usdz
    case gltf
    case glb

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .usdz: return "USDZ"
        case .gltf: return "glTF"
        case .glb: return "GLB"
        }
    }

    var fileExtension: String {
        switch self {
        case .usdz: return "usdz"
        case .gltf: return "gltf"
        case .glb: return "glb"
        }
    }
}

