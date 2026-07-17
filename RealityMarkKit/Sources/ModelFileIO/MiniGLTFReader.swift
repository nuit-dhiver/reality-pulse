import Foundation
import simd

public enum MiniGLTFError: Error {
    case invalidGLBHeader
    case missingJSONChunk
    case missingBuffer(Int)
    case unsupportedAccessor(String)
}

public struct MiniGLTFAsset {
    /// Concatenated contents of every unique POSITION accessor, in document order.
    public var positions: [SIMD3<Float>]
    /// Encoded payloads (PNG/JPEG bytes) of images referenced as base-color textures.
    public var baseColorImages: [Data]
}

/// Minimal read-only GLB/glTF parser: extracts vertex positions and base-color
/// images — exactly what watermark detection needs, nothing else. Mirrors the
/// subset of the glTF 2.0 schema that Reality Pulse's exporter emits, but stays
/// tolerant of foreign files (re-saved or edited copies).
public enum MiniGLTFReader {
    public static func read(url: URL) throws -> MiniGLTFAsset {
        let data = try Data(contentsOf: url)
        if data.starts(with: [0x67, 0x6C, 0x54, 0x46]) {  // "glTF"
            let (json, binary) = try parseGLBChunks(data)
            return try parse(json: json, embeddedBinary: binary, baseURL: url.deletingLastPathComponent())
        }
        return try parse(json: data, embeddedBinary: nil, baseURL: url.deletingLastPathComponent())
    }

    // MARK: - GLB container

    static func parseGLBChunks(_ data: Data) throws -> (json: Data, binary: Data?) {
        guard data.count >= 12 else { throw MiniGLTFError.invalidGLBHeader }

        var offset = 12  // magic + version + length
        var json: Data?
        var binary: Data?
        while offset + 8 <= data.count {
            let chunkLength = Int(readUInt32(data, at: offset))
            let chunkType = readUInt32(data, at: offset + 4)
            let start = offset + 8
            guard start + chunkLength <= data.count else { break }
            let chunk = data.subdata(in: start..<(start + chunkLength))
            if chunkType == 0x4E4F_534A {  // "JSON"
                json = chunk
            } else if chunkType == 0x004E_4942 {  // "BIN\0"
                binary = chunk
            }
            offset = start + chunkLength
        }

        guard let json else { throw MiniGLTFError.missingJSONChunk }
        return (json, binary)
    }

    private static func readUInt32(_ data: Data, at offset: Int) -> UInt32 {
        var value: UInt32 = 0
        withUnsafeMutableBytes(of: &value) { destination in
            data.copyBytes(to: destination, from: offset..<(offset + 4))
        }
        return UInt32(littleEndian: value)
    }

    // MARK: - Document

    private struct Document: Decodable {
        struct Accessor: Decodable {
            var bufferView: Int?
            var byteOffset: Int?
            var componentType: Int
            var count: Int
            var type: String
        }
        struct BufferView: Decodable {
            var buffer: Int
            var byteOffset: Int?
            var byteLength: Int
            var byteStride: Int?
        }
        struct Buffer: Decodable {
            var byteLength: Int
            var uri: String?
        }
        struct Mesh: Decodable {
            struct Primitive: Decodable {
                var attributes: [String: Int]
                var material: Int?
            }
            var primitives: [Primitive]
        }
        struct Material: Decodable {
            struct PBR: Decodable {
                struct TextureRef: Decodable { var index: Int }
                var baseColorTexture: TextureRef?
            }
            var pbrMetallicRoughness: PBR?
        }
        struct Texture: Decodable { var source: Int? }
        struct Image: Decodable {
            var uri: String?
            var bufferView: Int?
        }

        var accessors: [Accessor]?
        var bufferViews: [BufferView]?
        var buffers: [Buffer]?
        var meshes: [Mesh]?
        var materials: [Material]?
        var textures: [Texture]?
        var images: [Image]?
    }

    private static func parse(json: Data, embeddedBinary: Data?, baseURL: URL) throws -> MiniGLTFAsset {
        let document = try JSONDecoder().decode(Document.self, from: json)

        var bufferCache: [Int: Data] = [:]
        func bufferData(_ index: Int) throws -> Data {
            if let cached = bufferCache[index] { return cached }
            guard let buffers = document.buffers, buffers.indices.contains(index) else {
                throw MiniGLTFError.missingBuffer(index)
            }
            let data: Data
            if let uri = buffers[index].uri {
                data = try resolveURI(uri, baseURL: baseURL)
            } else if let embeddedBinary {
                data = embeddedBinary
            } else {
                throw MiniGLTFError.missingBuffer(index)
            }
            bufferCache[index] = data
            return data
        }

        func bufferViewData(_ index: Int) throws -> (data: Data, stride: Int?) {
            guard let views = document.bufferViews, views.indices.contains(index) else {
                throw MiniGLTFError.missingBuffer(index)
            }
            let view = views[index]
            let buffer = try bufferData(view.buffer)
            let start = view.byteOffset ?? 0
            guard start + view.byteLength <= buffer.count else {
                throw MiniGLTFError.missingBuffer(index)
            }
            return (buffer.subdata(in: start..<(start + view.byteLength)), view.byteStride)
        }

        // Unique POSITION accessors in first-reference order.
        var positionAccessors: [Int] = []
        var seen = Set<Int>()
        for mesh in document.meshes ?? [] {
            for primitive in mesh.primitives {
                if let accessor = primitive.attributes["POSITION"], seen.insert(accessor).inserted {
                    positionAccessors.append(accessor)
                }
            }
        }

        var positions: [SIMD3<Float>] = []
        for accessorIndex in positionAccessors {
            guard let accessors = document.accessors, accessors.indices.contains(accessorIndex) else { continue }
            let accessor = accessors[accessorIndex]
            guard accessor.componentType == 5126, accessor.type == "VEC3" else {
                throw MiniGLTFError.unsupportedAccessor("POSITION accessor must be float VEC3")
            }
            guard let viewIndex = accessor.bufferView else { continue }
            let (viewData, viewStride) = try bufferViewData(viewIndex)
            let stride = viewStride ?? 12
            let start = accessor.byteOffset ?? 0
            positions.reserveCapacity(positions.count + accessor.count)
            for element in 0..<accessor.count {
                let offset = start + element * stride
                guard offset + 12 <= viewData.count else { break }
                positions.append(SIMD3<Float>(
                    readFloat(viewData, at: offset),
                    readFloat(viewData, at: offset + 4),
                    readFloat(viewData, at: offset + 8)
                ))
            }
        }

        // Base-color images via material → texture → image.
        var imageIndices: [Int] = []
        var seenImages = Set<Int>()
        for material in document.materials ?? [] {
            guard let textureIndex = material.pbrMetallicRoughness?.baseColorTexture?.index,
                  let textures = document.textures, textures.indices.contains(textureIndex),
                  let imageIndex = textures[textureIndex].source,
                  seenImages.insert(imageIndex).inserted else { continue }
            imageIndices.append(imageIndex)
        }

        var baseColorImages: [Data] = []
        for imageIndex in imageIndices {
            guard let images = document.images, images.indices.contains(imageIndex) else { continue }
            let image = images[imageIndex]
            if let viewIndex = image.bufferView {
                if let (data, _) = try? bufferViewData(viewIndex) {
                    baseColorImages.append(data)
                }
            } else if let uri = image.uri, let data = try? resolveURI(uri, baseURL: baseURL) {
                baseColorImages.append(data)
            }
        }

        return MiniGLTFAsset(positions: positions, baseColorImages: baseColorImages)
    }

    private static func readFloat(_ data: Data, at offset: Int) -> Float {
        var bits: UInt32 = 0
        withUnsafeMutableBytes(of: &bits) { destination in
            data.copyBytes(to: destination, from: offset..<(offset + 4))
        }
        return Float(bitPattern: UInt32(littleEndian: bits))
    }

    private static func resolveURI(_ uri: String, baseURL: URL) throws -> Data {
        if uri.hasPrefix("data:") {
            guard let comma = uri.firstIndex(of: ","),
                  let data = Data(base64Encoded: String(uri[uri.index(after: comma)...])) else {
                throw MiniGLTFError.missingBuffer(-1)
            }
            return data
        }
        let decoded = uri.removingPercentEncoding ?? uri
        return try Data(contentsOf: baseURL.appending(path: decoded))
    }
}
