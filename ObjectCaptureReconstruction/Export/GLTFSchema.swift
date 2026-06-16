/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Codable glTF 2.0 schema types used when serializing exported models.
*/

import Foundation

// MARK: - Root document

struct GLTFDocument: Codable {
    var asset: GLTFAssetInfo
    var scene: Int?
    var scenes: [GLTFScene]?
    var nodes: [GLTFNode]?
    var meshes: [GLTFMesh]?
    var accessors: [GLTFAccessor]?
    var bufferViews: [GLTFBufferView]?
    var buffers: [GLTFBuffer]?
    var materials: [GLTFMaterial]?
    var textures: [GLTFTexture]?
    var images: [GLTFImage]?
    var samplers: [GLTFSampler]?
}

struct GLTFAssetInfo: Codable {
    var version: String
    var generator: String?
}

// MARK: - Scene graph

struct GLTFScene: Codable {
    var nodes: [Int]?
}

struct GLTFNode: Codable {
    var mesh: Int?
    var children: [Int]?
    var matrix: [Float]?
}

// MARK: - Geometry

struct GLTFMesh: Codable {
    var primitives: [GLTFPrimitive]
}

struct GLTFPrimitive: Codable {
    var attributes: [String: Int]
    var indices: Int?
    var material: Int?
    var mode: Int?
}

struct GLTFAccessor: Codable {
    var bufferView: Int?
    var byteOffset: Int?
    var componentType: Int
    var count: Int
    var type: String
    var min: [Float]?
    var max: [Float]?
}

struct GLTFBufferView: Codable {
    var buffer: Int
    var byteOffset: Int?
    var byteLength: Int
    var byteStride: Int?
    var target: Int?
}

struct GLTFBuffer: Codable {
    var byteLength: Int
    var uri: String?
}

// MARK: - Materials

struct GLTFMaterial: Codable {
    var name: String?
    var pbrMetallicRoughness: GLTFPBRMetallicRoughness?
    var normalTexture: GLTFNormalTextureInfo?
    var occlusionTexture: GLTFOcclusionTextureInfo?
    var doubleSided: Bool?
}

struct GLTFPBRMetallicRoughness: Codable {
    var baseColorFactor: [Float]?
    var metallicFactor: Float?
    var roughnessFactor: Float?
    var baseColorTexture: GLTFTextureInfo?
    var metallicRoughnessTexture: GLTFTextureInfo?
}

struct GLTFTextureInfo: Codable {
    var index: Int
    var texCoord: Int?
}

struct GLTFNormalTextureInfo: Codable {
    var index: Int
    var texCoord: Int?
}

struct GLTFOcclusionTextureInfo: Codable {
    var index: Int
    var texCoord: Int?
    var strength: Float?
}

// MARK: - Textures

struct GLTFTexture: Codable {
    var sampler: Int?
    var source: Int?
}

struct GLTFImage: Codable {
    var uri: String?
    var mimeType: String?
    var bufferView: Int?
}

struct GLTFSampler: Codable {
    var magFilter: Int?
    var minFilter: Int?
    var wrapS: Int?
    var wrapT: Int?
}

// MARK: - glTF constants

enum GLTFConstants {
    static let componentTypeUnsignedShort = 5123
    static let componentTypeUnsignedInt = 5125
    static let componentTypeFloat = 5126

    static let targetArrayBuffer = 34962
    static let targetElementArrayBuffer = 34963

    static let primitiveModeTriangles = 4

    static let filterLinear = 9729
    static let filterLinearMipmapLinear = 9987
    static let wrapRepeat = 10497
}

// MARK: - Binary buffer builder

final class GLTFBinaryBuilder {
    private(set) var data = Data()
    private var accessors: [GLTFAccessor] = []
    private var bufferViews: [GLTFBufferView] = []

    var accessorCount: Int { accessors.count }
    var bufferViewCount: Int { bufferViews.count }
    var allAccessors: [GLTFAccessor] { accessors }
    var allBufferViews: [GLTFBufferView] { bufferViews }

    @discardableResult
    func appendBuffer<T>(
        _ values: [T],
        target: Int? = GLTFConstants.targetArrayBuffer,
        componentType: Int = GLTFConstants.componentTypeFloat,
        type: String,
        min: [Float]? = nil,
        max: [Float]? = nil,
        byteStride: Int? = nil
    ) -> Int where T: FixedWidthInteger {
        let elementSize = MemoryLayout<T>.stride
        let byteLength = values.count * elementSize
        align(to: elementSize == 2 ? 2 : 4)

        let byteOffset = data.count
        values.withUnsafeBufferPointer { buffer in
            guard let base = buffer.baseAddress else { return }
            data.append(contentsOf: UnsafeRawBufferPointer(start: base, count: byteLength))
        }

        let bufferViewIndex = bufferViews.count
        bufferViews.append(GLTFBufferView(
            buffer: 0,
            byteOffset: byteOffset,
            byteLength: byteLength,
            byteStride: byteStride,
            target: target
        ))

        let accessorIndex = accessors.count
        accessors.append(GLTFAccessor(
            bufferView: bufferViewIndex,
            byteOffset: 0,
            componentType: componentType,
            count: values.count,
            type: type,
            min: min,
            max: max
        ))
        return accessorIndex
    }

    @discardableResult
    func appendFloatBuffer(
        _ values: [Float],
        target: Int? = GLTFConstants.targetArrayBuffer,
        componentsPerElement: Int,
        min: [Float]? = nil,
        max: [Float]? = nil
    ) -> Int {
        let byteStride = componentsPerElement * MemoryLayout<Float>.stride
        align(to: 4)

        let byteOffset = data.count
        values.withUnsafeBufferPointer { buffer in
            guard let base = buffer.baseAddress else { return }
            let byteLength = buffer.count * MemoryLayout<Float>.stride
            data.append(contentsOf: UnsafeRawBufferPointer(start: base, count: byteLength))
        }

        let elementCount = values.count / componentsPerElement
        let gltfType: String
        switch componentsPerElement {
        case 2: gltfType = "VEC2"
        case 3: gltfType = "VEC3"
        case 4: gltfType = "VEC4"
        default: gltfType = "SCALAR"
        }

        let bufferViewIndex = bufferViews.count
        bufferViews.append(GLTFBufferView(
            buffer: 0,
            byteOffset: byteOffset,
            byteLength: values.count * MemoryLayout<Float>.stride,
            byteStride: byteStride,
            target: target
        ))

        let accessorIndex = accessors.count
        accessors.append(GLTFAccessor(
            bufferView: bufferViewIndex,
            byteOffset: 0,
            componentType: GLTFConstants.componentTypeFloat,
            count: elementCount,
            type: gltfType,
            min: min,
            max: max
        ))
        return accessorIndex
    }

    @discardableResult
    func appendRawData(_ bytes: Data, target: Int? = nil) -> (bufferViewIndex: Int, byteOffset: Int) {
        align(to: 4)
        let byteOffset = data.count
        data.append(bytes)

        let bufferViewIndex = bufferViews.count
        bufferViews.append(GLTFBufferView(
            buffer: 0,
            byteOffset: byteOffset,
            byteLength: bytes.count,
            byteStride: nil,
            target: target
        ))
        return (bufferViewIndex, 0)
    }

    private func align(to boundary: Int) {
        let remainder = data.count % boundary
        if remainder != 0 {
            data.append(Data(count: boundary - remainder))
        }
    }
}

// MARK: - GLB writer

enum GLTFWriter {
    static func writeGLB(document: GLTFDocument, binaryData: Data, to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        var jsonData = try encoder.encode(document)

        // Pad JSON chunk to 4-byte alignment with spaces (0x20).
        let jsonPadding = (4 - (jsonData.count % 4)) % 4
        if jsonPadding > 0 {
            jsonData.append(Data(repeating: 0x20, count: jsonPadding))
        }

        var binData = binaryData
        let binPadding = (4 - (binData.count % 4)) % 4
        if binPadding > 0 {
            binData.append(Data(count: binPadding))
        }

        let totalLength = 12 + 8 + jsonData.count + 8 + binData.count
        var output = Data()
        output.reserveCapacity(totalLength)

        // Header
        output.append(contentsOf: [0x67, 0x6C, 0x54, 0x46]) // glTF
        output.append(contentsOf: UInt32(2).littleEndianBytes)
        output.append(contentsOf: UInt32(totalLength).littleEndianBytes)

        // JSON chunk
        output.append(contentsOf: UInt32(jsonData.count).littleEndianBytes)
        output.append(contentsOf: UInt32(0x4E4F534A).littleEndianBytes) // JSON
        output.append(jsonData)

        // BIN chunk
        output.append(contentsOf: UInt32(binData.count).littleEndianBytes)
        output.append(contentsOf: UInt32(0x004E4942).littleEndianBytes) // BIN\0
        output.append(binData)

        try output.write(to: url, options: .atomic)
    }

    static func writeGLTF(document: GLTFDocument, binaryData: Data, to url: URL) throws {
        let binFilename = url.deletingPathExtension().lastPathComponent + ".bin"
        let binURL = url.deletingLastPathComponent().appending(path: binFilename)

        try binaryData.write(to: binURL, options: .atomic)

        var doc = document
        doc.buffers = [GLTFBuffer(byteLength: binaryData.count, uri: binFilename)]

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .prettyPrinted]
        let jsonData = try encoder.encode(doc)
        try jsonData.write(to: url, options: .atomic)
    }
}

private extension UInt32 {
    var littleEndianBytes: [UInt8] {
        withUnsafeBytes(of: littleEndian) { Array($0) }
    }
}
