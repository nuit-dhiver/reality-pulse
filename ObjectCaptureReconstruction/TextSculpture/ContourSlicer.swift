/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Slices meshes with evenly spaced horizontal planes and chains the cut segments
into polylines: the contour "rings" that text sculptures write along. Each
polyline vertex carries the interpolated surface normal, UV, and material.
*/

import Foundation
import simd

enum ContourSlicer {

    struct Vertex {
        var position: SIMD3<Float>
        var normal: SIMD3<Float>
        var uv: SIMD2<Float>
        /// Index into the flattened material list (mesh-major, then submesh order),
        /// matching `SurfaceSampler.SamplePoint.materialIndex`.
        var materialIndex: Int
    }

    struct Polyline {
        var vertices: [Vertex]
        /// `true` when the last vertex connects back to the first.
        var isClosed: Bool
        /// Slice index, 0 = lowest.
        var ringIndex: Int
    }

    struct Result {
        var polylines: [Polyline]
        /// Vertical distance between neighboring slices, in model units.
        var spacing: Float
    }

    private typealias Key = SIMD3<Int64>

    private struct Segment {
        var a: Vertex
        var b: Vertex
        var keyA: Key
        var keyB: Key
    }

    /// Upper bound on slices, however fine the requested spacing.
    static let maximumRingCount = 4_000

    /// Cut `meshes` with planes perpendicular to +Y, about `targetSpacing` apart
    /// and centered in equal bands between the lowest and highest vertex.
    static func slice(meshes: [MeshGeometryReader.Mesh], targetSpacing: Float) -> Result {
        var lower = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var upper = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
        for mesh in meshes {
            for position in mesh.positions {
                lower = simd_min(lower, position)
                upper = simd_max(upper, position)
            }
        }

        let height = upper.y - lower.y
        guard targetSpacing > 0, height.isFinite, height > 0 else {
            return Result(polylines: [], spacing: 0)
        }
        let ringCount = min(max(1, Int((height / targetSpacing).rounded())), maximumRingCount)
        let spacing = height / Float(ringCount)

        // Crossing points are welded on a fine grid so chains continue across UV
        // seams and per-face duplicated vertices.
        let quantum = max(simd_length(upper - lower) * 1e-5, .leastNormalMagnitude)
        func key(_ position: SIMD3<Float>) -> Key {
            Key(position / quantum, rounding: .toNearestOrAwayFromZero)
        }

        var segmentsByRing = [[Segment]](repeating: [], count: ringCount)
        var materialCursor = 0

        for mesh in meshes {
            let positions = mesh.positions
            let hasNormals = mesh.normals.count == positions.count
            let hasUVs = mesh.texCoords.count == positions.count

            // Order the edge endpoints by position so both triangles sharing an
            // edge (even through duplicated vertices) compute identical points.
            func crossing(_ first: Int, _ second: Int, at h: Float, faceNormal: SIMD3<Float>, materialIndex: Int) -> Vertex {
                let (i, j) = precedes(positions[first], positions[second]) ? (first, second) : (second, first)
                let p0 = positions[i]
                let p1 = positions[j]
                let t = (h - p0.y) / (p1.y - p0.y)

                var normal = faceNormal
                if hasNormals {
                    let blended = simd_mix(mesh.normals[i], mesh.normals[j], SIMD3<Float>(repeating: t))
                    let length = simd_length(blended)
                    if length > 0 { normal = blended / length }
                }
                let uv = hasUVs
                    ? simd_mix(mesh.texCoords[i], mesh.texCoords[j], SIMD2<Float>(repeating: t))
                    : SIMD2<Float>(0, 0)

                return Vertex(
                    position: simd_mix(p0, p1, SIMD3<Float>(repeating: t)),
                    normal: normal,
                    uv: uv,
                    materialIndex: materialIndex
                )
            }

            for submesh in mesh.submeshes {
                let materialIndex = materialCursor
                materialCursor += 1

                for triangle in submesh.triangles {
                    let i0 = triangle.i0, i1 = triangle.i1, i2 = triangle.i2
                    guard i0 < positions.count, i1 < positions.count, i2 < positions.count else { continue }
                    let y0 = positions[i0].y, y1 = positions[i1].y, y2 = positions[i2].y

                    // Only visit the slices this triangle can reach (±1 for rounding).
                    let low = (min(y0, y1, y2) - lower.y) / spacing - 0.5
                    let high = (max(y0, y1, y2) - lower.y) / spacing - 0.5
                    guard low.isFinite, high.isFinite else { continue }
                    let firstRing = max(0, Int(low.rounded(.up)) - 1)
                    let lastRing = min(ringCount - 1, Int(high.rounded(.down)) + 1)
                    guard firstRing <= lastRing else { continue }

                    let edgeCross = simd_cross(positions[i1] - positions[i0], positions[i2] - positions[i0])
                    let edgeLength = simd_length(edgeCross)
                    let faceNormal = edgeLength > 0 ? edgeCross / edgeLength : SIMD3<Float>(0, 0, 1)

                    for ring in firstRing...lastRing {
                        let h = lower.y + (Float(ring) + 0.5) * spacing
                        // `>=` puts every vertex strictly on one side, so a triangle
                        // is crossed on exactly zero or two edges.
                        let above0 = y0 >= h, above1 = y1 >= h, above2 = y2 >= h
                        if above0 == above1 && above1 == above2 { continue }

                        // Exactly two edges cross: the one not crossed is skipped.
                        let first: Vertex
                        let second: Vertex
                        if above0 == above1 {
                            first = crossing(i1, i2, at: h, faceNormal: faceNormal, materialIndex: materialIndex)
                            second = crossing(i2, i0, at: h, faceNormal: faceNormal, materialIndex: materialIndex)
                        } else if above1 == above2 {
                            first = crossing(i0, i1, at: h, faceNormal: faceNormal, materialIndex: materialIndex)
                            second = crossing(i2, i0, at: h, faceNormal: faceNormal, materialIndex: materialIndex)
                        } else {
                            first = crossing(i0, i1, at: h, faceNormal: faceNormal, materialIndex: materialIndex)
                            second = crossing(i1, i2, at: h, faceNormal: faceNormal, materialIndex: materialIndex)
                        }

                        let keyA = key(first.position)
                        let keyB = key(second.position)
                        guard keyA != keyB else { continue }
                        segmentsByRing[ring].append(Segment(a: first, b: second, keyA: keyA, keyB: keyB))
                    }
                }
            }
        }

        var polylines: [Polyline] = []
        for (ring, segments) in segmentsByRing.enumerated() {
            polylines.append(contentsOf: chain(segments, ringIndex: ring))
        }
        return Result(polylines: polylines, spacing: spacing)
    }

    /// Connect segments that share endpoints. Open chains are walked from their
    /// ends first; whatever remains forms closed loops. Chains stop at
    /// non-manifold junctions (a point shared by more than two segments).
    private static func chain(_ segments: [Segment], ringIndex: Int) -> [Polyline] {
        guard !segments.isEmpty else { return [] }

        var incident: [Key: [Int]] = [:]
        for (index, segment) in segments.enumerated() {
            incident[segment.keyA, default: []].append(index)
            incident[segment.keyB, default: []].append(index)
        }

        var used = [Bool](repeating: false, count: segments.count)
        var polylines: [Polyline] = []

        func degree(_ key: Key) -> Int { incident[key]?.count ?? 0 }

        func walk(from startKey: Key, through firstSegment: Int) {
            let first = segments[firstSegment]
            var vertices = [first.keyA == startKey ? first.a : first.b]
            var currentKey = startKey
            var segmentIndex = firstSegment
            var isClosed = false

            while true {
                used[segmentIndex] = true
                let segment = segments[segmentIndex]
                let (nextKey, nextVertex) = segment.keyA == currentKey
                    ? (segment.keyB, segment.b)
                    : (segment.keyA, segment.a)
                currentKey = nextKey
                if currentKey == startKey {
                    isClosed = true
                    break
                }
                vertices.append(nextVertex)
                guard degree(currentKey) == 2,
                      let next = incident[currentKey]?.first(where: { !used[$0] }) else { break }
                segmentIndex = next
            }

            if vertices.count >= 2 {
                polylines.append(Polyline(vertices: vertices, isClosed: isClosed && vertices.count >= 3, ringIndex: ringIndex))
            }
        }

        // Open chains (and branches at junctions) start where a point isn't shared by exactly two segments.
        for index in segments.indices where !used[index] {
            let segment = segments[index]
            if degree(segment.keyA) != 2 {
                walk(from: segment.keyA, through: index)
            } else if degree(segment.keyB) != 2 {
                walk(from: segment.keyB, through: index)
            }
        }
        // Everything left is part of a closed loop.
        for index in segments.indices where !used[index] {
            walk(from: segments[index].keyA, through: index)
        }

        return polylines
    }

    /// Strict lexicographic order on (y, x, z).
    private static func precedes(_ lhs: SIMD3<Float>, _ rhs: SIMD3<Float>) -> Bool {
        if lhs.y != rhs.y { return lhs.y < rhs.y }
        if lhs.x != rhs.x { return lhs.x < rhs.x }
        return lhs.z < rhs.z
    }
}
