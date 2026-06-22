/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Serializes Gaussian splats to a binary-little-endian `.ply` using the de-facto
3D Gaussian Splatting property layout (SH degree 0, no `f_rest_*`), so the file
loads in SuperSplat / gsplat / antimatter15-style viewers.
*/

import Foundation

enum SplatPLYWriter {

    /// 17 float32 properties per splat, in INRIA reference order.
    static let propertyNames = [
        "x", "y", "z",
        "nx", "ny", "nz",
        "f_dc_0", "f_dc_1", "f_dc_2",
        "opacity",
        "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3"
    ]

    static func data(for splats: [GaussianSplat]) -> Data {
        var header = "ply\n"
        header += "format binary_little_endian 1.0\n"
        header += "element vertex \(splats.count)\n"
        for name in propertyNames {
            header += "property float \(name)\n"
        }
        header += "end_header\n"

        let floatsPerSplat = propertyNames.count
        var data = Data(header.utf8)
        data.reserveCapacity(data.count + splats.count * floatsPerSplat * MemoryLayout<Float>.size)

        var scratch = [Float](repeating: 0, count: floatsPerSplat)
        for splat in splats {
            scratch[0] = splat.position.x
            scratch[1] = splat.position.y
            scratch[2] = splat.position.z
            scratch[3] = splat.normal.x
            scratch[4] = splat.normal.y
            scratch[5] = splat.normal.z
            scratch[6] = splat.shDC.x
            scratch[7] = splat.shDC.y
            scratch[8] = splat.shDC.z
            scratch[9] = splat.opacityLogit
            scratch[10] = splat.logScale.x
            scratch[11] = splat.logScale.y
            scratch[12] = splat.logScale.z
            // rotation is stored (w, x, y, z) → rot_0..3.
            scratch[13] = splat.rotation.x
            scratch[14] = splat.rotation.y
            scratch[15] = splat.rotation.z
            scratch[16] = splat.rotation.w
            scratch.withUnsafeBytes { data.append(contentsOf: $0) }
        }
        return data
    }

    static func write(_ splats: [GaussianSplat], to url: URL) throws {
        try data(for: splats).write(to: url, options: .atomic)
    }
}
