/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Shared Model I/O geometry extraction used by both the glTF exporter
(`USDZToGLTFConverter`) and the Gaussian-splat generator (`SplatSampleGenerator`).
*/

import Foundation
import ModelIO
import simd

/// Loads mesh geometry from a USD/USDZ asset and exposes the low-level
/// vertex-attribute byte readers as a single source of truth.
///
/// `loadMeshes` returns world-space geometry grouped by submesh/material; the
/// individual `read*` helpers are reused by `USDZToGLTFConverter` so the raw
/// pointer arithmetic and half-precision handling live in exactly one place.
enum MeshGeometryReader {

    // MARK: - High-level model

    struct Triangle {
        var i0: Int
        var i1: Int
        var i2: Int
    }

    struct Submesh {
        var triangles: [Triangle]
        var material: MDLMaterial?
    }

    struct Mesh {
        /// World-space vertex positions.
        var positions: [SIMD3<Float>]
        /// World-space, normalized normals. Empty when the source has none.
        var normals: [SIMD3<Float>]
        /// Raw texture coordinates (no V flip). Empty when the source has none.
        var texCoords: [SIMD2<Float>]
        var submeshes: [Submesh]
    }

    enum ReaderError: Error {
        case unsupportedIndexType
    }

    /// Load every mesh in the asset, flattening node transforms into world space.
    static func loadMeshes(
        from url: URL,
        generateMissingNormals: Bool = true
    ) -> [Mesh] {
        let asset = MDLAsset(url: url)
        asset.loadTextures()

        var collected: [(mesh: MDLMesh, transform: matrix_float4x4)] = []
        for index in 0..<asset.count {
            if let object = asset.object(at: index) as? MDLObject {
                collectMeshes(from: object, parentTransform: matrix_identity_float4x4, into: &collected)
            }
        }

        var result: [Mesh] = []
        result.reserveCapacity(collected.count)

        for (mesh, transform) in collected {
            if generateMissingNormals,
               mesh.vertexAttributeData(forAttributeNamed: MDLVertexAttributeNormal) == nil {
                mesh.addNormals(withAttributeNamed: MDLVertexAttributeNormal, creaseThreshold: 0.5)
            }

            let positions = readFloat3(mesh, attribute: MDLVertexAttributePosition, transform: transform, isDirection: false)
            guard !positions.isEmpty else { continue }

            let normals = readFloat3(mesh, attribute: MDLVertexAttributeNormal, transform: transform, isDirection: true)
            let texCoords = readFloat2(mesh, attribute: MDLVertexAttributeTextureCoordinate)

            var submeshes: [Submesh] = []
            for case let submesh as MDLSubmesh in mesh.submeshes ?? [] {
                guard let triangles = try? readTriangles(submesh) else { continue }
                submeshes.append(Submesh(triangles: triangles, material: submesh.material))
            }

            result.append(Mesh(positions: positions, normals: normals, texCoords: texCoords, submeshes: submeshes))
        }

        return result
    }

    // MARK: - Mesh collection

    /// Recursively gather meshes, accumulating each node's transform into world space.
    static func collectMeshes(
        from object: MDLObject,
        parentTransform: matrix_float4x4,
        into meshes: inout [(mesh: MDLMesh, transform: matrix_float4x4)]
    ) {
        let localTransform = object.transform?.matrix ?? matrix_identity_float4x4
        let worldTransform = simd_mul(parentTransform, localTransform)

        if let mesh = object as? MDLMesh {
            meshes.append((mesh, worldTransform))
        }

        for child in object.children.objects {
            if let childObject = child as? MDLObject {
                collectMeshes(from: childObject, parentTransform: worldTransform, into: &meshes)
            }
        }
    }

    // MARK: - Vertex attribute readers

    /// Read a float3 vertex attribute. When `isDirection` is true the vector is
    /// transformed as a direction (w = 0) and renormalized; otherwise as a point.
    ///
    /// The attribute is forced to a `.float3` layout so a native half-precision
    /// format isn't misread when interpreting the raw bytes as full-width Floats.
    static func readFloat3(
        _ mesh: MDLMesh,
        attribute name: String,
        transform: matrix_float4x4,
        isDirection: Bool
    ) -> [SIMD3<Float>] {
        guard let attrData = mesh.vertexAttributeData(forAttributeNamed: name, as: .float3) else {
            return []
        }

        let count = mesh.vertexCount
        var result = [SIMD3<Float>]()
        result.reserveCapacity(count)

        let stride = attrData.stride
        let dataStart = attrData.dataStart

        for index in 0..<count {
            let pointer = dataStart.advanced(by: index * stride)
            let components = pointer.assumingMemoryBound(to: Float.self)
            let vector = SIMD3<Float>(components[0], components[1], components[2])
            if isDirection {
                let transformed = simd_mul(transform, SIMD4<Float>(vector, 0))
                let direction = SIMD3<Float>(transformed.x, transformed.y, transformed.z)
                let length = simd_length(direction)
                result.append(length > 0 ? direction / length : SIMD3<Float>(0, 0, 1))
            } else {
                let transformed = simd_mul(transform, SIMD4<Float>(vector, 1))
                result.append(SIMD3<Float>(transformed.x, transformed.y, transformed.z))
            }
        }
        return result
    }

    /// Read a float4 vertex attribute verbatim (e.g. tangents).
    static func readFloat4(
        _ mesh: MDLMesh,
        attribute name: String
    ) -> [SIMD4<Float>] {
        guard let attrData = mesh.vertexAttributeData(forAttributeNamed: name, as: .float4) else {
            return []
        }

        let count = mesh.vertexCount
        var result = [SIMD4<Float>]()
        result.reserveCapacity(count)

        let stride = attrData.stride
        let dataStart = attrData.dataStart

        for index in 0..<count {
            let pointer = dataStart.advanced(by: index * stride)
            let components = pointer.assumingMemoryBound(to: Float.self)
            result.append(SIMD4<Float>(components[0], components[1], components[2], components[3]))
        }
        return result
    }

    /// Read a float2 attribute (e.g. texture coordinates) without any axis flip.
    static func readFloat2(
        _ mesh: MDLMesh,
        attribute name: String
    ) -> [SIMD2<Float>] {
        guard let attrData = mesh.vertexAttributeData(forAttributeNamed: name, as: .float2) else {
            return []
        }

        let count = mesh.vertexCount
        var result = [SIMD2<Float>]()
        result.reserveCapacity(count)

        let stride = attrData.stride
        let dataStart = attrData.dataStart

        for index in 0..<count {
            let pointer = dataStart.advanced(by: index * stride)
            let components = pointer.assumingMemoryBound(to: Float.self)
            result.append(SIMD2<Float>(components[0], components[1]))
        }
        return result
    }

    // MARK: - Index reading

    /// Read a submesh's triangle topology as index triples.
    static func readTriangles(_ submesh: MDLSubmesh) throws -> [Triangle] {
        guard submesh.geometryType == .triangles else {
            throw ReaderError.unsupportedIndexType
        }

        let indexCount = submesh.indexCount
        guard indexCount >= 3 else { return [] }

        let map = submesh.indexBuffer.map()
        let bytes = map.bytes
        var indices = [Int]()
        indices.reserveCapacity(indexCount)

        switch submesh.indexType {
        case .uInt16:
            for i in 0..<indexCount {
                let value = bytes.advanced(by: i * MemoryLayout<UInt16>.stride)
                    .assumingMemoryBound(to: UInt16.self).pointee
                indices.append(Int(value))
            }
        case .uInt32:
            for i in 0..<indexCount {
                let value = bytes.advanced(by: i * MemoryLayout<UInt32>.stride)
                    .assumingMemoryBound(to: UInt32.self).pointee
                indices.append(Int(value))
            }
        case .uInt8:
            for i in 0..<indexCount {
                let value = bytes.advanced(by: i).assumingMemoryBound(to: UInt8.self).pointee
                indices.append(Int(value))
            }
        default:
            throw ReaderError.unsupportedIndexType
        }

        var triangles = [Triangle]()
        triangles.reserveCapacity(indexCount / 3)
        var i = 0
        while i + 2 < indexCount {
            triangles.append(Triangle(i0: indices[i], i1: indices[i + 1], i2: indices[i + 2]))
            i += 3
        }
        return triangles
    }
}
