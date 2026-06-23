/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Area-weighted random surface sampling over a set of meshes. Produces points
distributed uniformly across the total surface area, each carrying an
interpolated position, normal, and UV plus the material it came from.
*/

import Foundation
import simd

enum SurfaceSampler {

    struct SamplePoint {
        var position: SIMD3<Float>
        var normal: SIMD3<Float>
        var uv: SIMD2<Float>
        /// Index into the flattened material list (mesh-major, then submesh order).
        var materialIndex: Int
    }

    struct Result {
        var points: [SamplePoint]
        /// Total surface area sampled, in model units squared.
        var totalArea: Float
    }

    /// Sample `targetCount` points across all triangles of all meshes, weighted by
    /// triangle area. `seed` makes the output deterministic.
    static func sample(
        meshes: [MeshGeometryReader.Mesh],
        targetCount: Int,
        seed: UInt64 = 0x9E37_79B9_7F4A_7C15
    ) -> Result {
        // Flatten every triangle alongside a cumulative-area table for weighted
        // selection. Materials are numbered mesh-major, then submesh order, so the
        // index matches the caller's parallel list of texture samplers.
        struct Tri {
            var meshIndex: Int
            var materialIndex: Int
            var a: Int
            var b: Int
            var c: Int
        }

        var tris: [Tri] = []
        var cumulative: [Float] = []
        var runningArea: Float = 0
        var materialCursor = 0

        for (meshIndex, mesh) in meshes.enumerated() {
            for submesh in mesh.submeshes {
                let materialIndex = materialCursor
                materialCursor += 1
                for triangle in submesh.triangles {
                    guard triangle.i0 < mesh.positions.count,
                          triangle.i1 < mesh.positions.count,
                          triangle.i2 < mesh.positions.count else { continue }
                    let p0 = mesh.positions[triangle.i0]
                    let p1 = mesh.positions[triangle.i1]
                    let p2 = mesh.positions[triangle.i2]
                    let area = 0.5 * simd_length(simd_cross(p1 - p0, p2 - p0))
                    guard area.isFinite, area > 0 else { continue }
                    runningArea += area
                    tris.append(Tri(meshIndex: meshIndex, materialIndex: materialIndex,
                                    a: triangle.i0, b: triangle.i1, c: triangle.i2))
                    cumulative.append(runningArea)
                }
            }
        }

        guard !tris.isEmpty, runningArea > 0, targetCount > 0 else {
            return Result(points: [], totalArea: runningArea)
        }

        var rng = SplitMix64(seed: seed)
        var points = [SamplePoint]()
        points.reserveCapacity(targetCount)

        for _ in 0..<targetCount {
            // Pick a triangle weighted by area.
            let target = rng.nextUnitFloat() * runningArea
            let triIndex = lowerBound(cumulative, value: target)
            let tri = tris[triIndex]
            let mesh = meshes[tri.meshIndex]

            // Uniformly distributed barycentric coordinates over the triangle.
            var u = rng.nextUnitFloat()
            var v = rng.nextUnitFloat()
            if u + v > 1 { u = 1 - u; v = 1 - v }
            let w = 1 - u - v

            let p0 = mesh.positions[tri.a]
            let p1 = mesh.positions[tri.b]
            let p2 = mesh.positions[tri.c]
            let position = w * p0 + u * p1 + v * p2

            let normal: SIMD3<Float>
            if mesh.normals.count == mesh.positions.count {
                let interpolated = w * mesh.normals[tri.a] + u * mesh.normals[tri.b] + v * mesh.normals[tri.c]
                let length = simd_length(interpolated)
                normal = length > 0 ? interpolated / length : faceNormal(p0, p1, p2)
            } else {
                normal = faceNormal(p0, p1, p2)
            }

            let uv: SIMD2<Float>
            if mesh.texCoords.count == mesh.positions.count {
                uv = w * mesh.texCoords[tri.a] + u * mesh.texCoords[tri.b] + v * mesh.texCoords[tri.c]
            } else {
                uv = SIMD2<Float>(0, 0)
            }

            points.append(SamplePoint(position: position, normal: normal, uv: uv, materialIndex: tri.materialIndex))
        }

        return Result(points: points, totalArea: runningArea)
    }

    private static func faceNormal(_ p0: SIMD3<Float>, _ p1: SIMD3<Float>, _ p2: SIMD3<Float>) -> SIMD3<Float> {
        let n = simd_cross(p1 - p0, p2 - p0)
        let length = simd_length(n)
        return length > 0 ? n / length : SIMD3<Float>(0, 0, 1)
    }

    /// First index whose cumulative value is `>= value` (binary search).
    private static func lowerBound(_ array: [Float], value: Float) -> Int {
        var low = 0
        var high = array.count - 1
        while low < high {
            let mid = (low + high) / 2
            if array[mid] < value {
                low = mid + 1
            } else {
                high = mid
            }
        }
        return low
    }
}

/// Small, fast, deterministic PRNG (SplitMix64) — avoids pulling in the system
/// RNG so splat output is reproducible from a seed.
struct SplitMix64 {
    private var state: UInt64

    init(seed: UInt64) {
        state = seed
    }

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    /// Uniform `Float` in `[0, 1)` using the top 24 bits of the next draw.
    mutating func nextUnitFloat() -> Float {
        Float(next() >> 40) * (1.0 / Float(1 << 24))
    }
}
