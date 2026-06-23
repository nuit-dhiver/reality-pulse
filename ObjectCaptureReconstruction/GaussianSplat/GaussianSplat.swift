/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
A single 3D Gaussian splat in the de-facto 3DGS storage convention. Values are
held exactly as written to the `.ply` (log-scale, logit-opacity, SH DC color),
ready for serialization.
*/

import Foundation
import simd

struct GaussianSplat {
    var position: SIMD3<Float>
    var normal: SIMD3<Float>
    var shDC: SIMD3<Float>        // f_dc_0..2
    var opacityLogit: Float       // inverse-sigmoid of alpha
    var logScale: SIMD3<Float>    // scale_0..2 (stored as natural log)
    var rotation: SIMD4<Float>    // quaternion (w, x, y, z), normalized

    /// First-order spherical-harmonic constant used by 3DGS to recover color:
    /// `rgb = shC0 * f_dc + 0.5`.
    static let shC0: Float = 0.282_094_791_773_878_14

    /// Build a flat, surface-aligned splat from a sampled surface point.
    /// - Parameters:
    ///   - color: base color in `[0, 1]`.
    ///   - tangentScale: in-plane radius in model units (from local point spacing).
    ///   - flatten: thickness of the disk along the normal, as a fraction of `tangentScale`.
    ///   - opacity: target alpha in `(0, 1)`.
    static func surfaceSplat(
        position: SIMD3<Float>,
        normal: SIMD3<Float>,
        color: SIMD3<Float>,
        tangentScale: Float,
        flatten: Float = 0.1,
        opacity: Float = 0.9
    ) -> GaussianSplat {
        let dc = (color - SIMD3<Float>(repeating: 0.5)) / shC0

        let safeScale = max(tangentScale, 1e-6)
        let logScale = SIMD3<Float>(
            log(safeScale),
            log(safeScale),
            log(max(safeScale * flatten, 1e-7))
        )

        let clampedOpacity = min(max(opacity, 1e-4), 1 - 1e-4)
        let opacityLogit = log(clampedOpacity / (1 - clampedOpacity))

        return GaussianSplat(
            position: position,
            normal: normal,
            shDC: dc,
            opacityLogit: opacityLogit,
            logScale: logScale,
            rotation: quaternion(alignedTo: normal)
        )
    }

    /// Quaternion `(w, x, y, z)` whose local Z axis maps to `normal`, laying the
    /// `(scale_0, scale_1)` disk in the surface tangent plane.
    static func quaternion(alignedTo normal: SIMD3<Float>) -> SIMD4<Float> {
        let n = normalizeOrDefault(normal, fallback: SIMD3<Float>(0, 0, 1))

        // Reference axis chosen to avoid being parallel to the normal.
        let reference = abs(n.z) < 0.99 ? SIMD3<Float>(0, 0, 1) : SIMD3<Float>(1, 0, 0)
        var t1 = simd_cross(reference, n)
        let t1Length = simd_length(t1)
        t1 = t1Length > 0 ? t1 / t1Length : SIMD3<Float>(1, 0, 0)
        let t2 = simd_cross(n, t1)

        // Columns [t1 | t2 | n] form a right-handed rotation (t1 × t2 == n).
        let matrix = matrix_float3x3(columns: (t1, t2, n))
        let quat = simd_quatf(matrix)
        let vector = quat.vector // (ix, iy, iz, r)

        var result = SIMD4<Float>(vector.w, vector.x, vector.y, vector.z) // (w, x, y, z)
        let length = simd_length(result)
        if length > 0 { result /= length }
        return result
    }

    private static func normalizeOrDefault(_ vector: SIMD3<Float>, fallback: SIMD3<Float>) -> SIMD3<Float> {
        let length = simd_length(vector)
        return length > 0 ? vector / length : fallback
    }
}
