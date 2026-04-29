/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Small OBJ/MTL to glTF 2.0 exporter used for Object Capture glTF and GLB output.
*/

import Foundation
import simd

enum OBJToGLTFConverter {
    enum ExportError: LocalizedError {
        case missingOBJFile(URL)
        case invalidOBJ(String)
        case unsupportedFormat(ModelExportFormat)

        var errorDescription: String? {
            switch self {
            case .missingOBJFile(let url):
                return "No OBJ file was found in \(url.lastPathComponent)."
            case .invalidOBJ(let message):
                return "The OBJ output could not be converted: \(message)"
            case .unsupportedFormat(let format):
                return "\(format.displayName) is not supported by this exporter."
            }
        }
    }

    static func convertOBJFolder(
        at sourceDirectory: URL,
        to destinationURL: URL,
        format: ModelExportFormat
    ) throws {
        let objURL = try findOBJFile(in: sourceDirectory)
        let parsed = try OBJParser.parse(objURL: objURL)
        var document = try GLTFDocument(parsedOBJ: parsed, sourceDirectory: sourceDirectory)

        switch format {
        case .gltf:
            try document.writeGLTF(to: destinationURL)
        case .glb:
            try document.writeGLB(to: destinationURL)
        case .usdz:
            throw ExportError.unsupportedFormat(format)
        }
    }

    private static func findOBJFile(in directory: URL) throws -> URL {
        let contents = try FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: nil
        )
        if let obj = contents.first(where: { $0.pathExtension.lowercased() == "obj" }) {
            return obj
        }
        throw ExportError.missingOBJFile(directory)
    }
}

private struct ParsedOBJ {
    var positions: [SIMD3<Float>]
    var normals: [SIMD3<Float>]
    var texcoords: [SIMD2<Float>]
    var primitives: [OBJPrimitive]
    var materialLibraries: [String]
}

private struct OBJPrimitive {
    var materialName: String?
    var vertices: [OBJVertex]
    var indices: [UInt32]
    var vertexLookup: [OBJVertex: UInt32] = [:]
}

private struct OBJVertex: Hashable {
    var position: Int
    var texcoord: Int?
    var normal: Int?
}

private enum OBJParser {
    static func parse(objURL: URL) throws -> ParsedOBJ {
        let text = try String(contentsOf: objURL, encoding: .utf8)
        var positions: [SIMD3<Float>] = []
        var normals: [SIMD3<Float>] = []
        var texcoords: [SIMD2<Float>] = []
        var primitives: [OBJPrimitive] = []
        var materialLibraries: [String] = []
        var activeMaterial: String?
        var verticesByKey: [PrimitiveKey: UInt32] = [:]

        func ensurePrimitive() {
            let key = PrimitiveKey(materialName: activeMaterial)
            if verticesByKey[key] == nil {
                verticesByKey[key] = UInt32(primitives.count)
                primitives.append(OBJPrimitive(materialName: activeMaterial, vertices: [], indices: []))
            }
        }

        for rawLine in text.components(separatedBy: .newlines) {
            let line = rawLine.trimmingCharacters(in: .whitespaces)
            if line.isEmpty || line.hasPrefix("#") { continue }

            let parts = line.split(whereSeparator: { $0 == " " || $0 == "\t" }).map(String.init)
            guard let keyword = parts.first else { continue }

            switch keyword {
            case "v":
                if let vector = parseVector3(parts.dropFirst()) {
                    positions.append(vector)
                }
            case "vn":
                if let vector = parseVector3(parts.dropFirst()) {
                    normals.append(vector)
                }
            case "vt":
                if let uv = parseVector2(parts.dropFirst()) {
                    texcoords.append(uv)
                }
            case "mtllib":
                let library = parts.dropFirst().joined(separator: " ")
                if !library.isEmpty {
                    materialLibraries.append(library)
                }
            case "usemtl":
                activeMaterial = parts.dropFirst().joined(separator: " ")
                if activeMaterial?.isEmpty == true {
                    activeMaterial = nil
                }
                ensurePrimitive()
            case "f":
                ensurePrimitive()
                let key = PrimitiveKey(materialName: activeMaterial)
                guard let primitiveIndex = verticesByKey[key].map(Int.init) else { continue }
                let faceVertices = try parts.dropFirst().map {
                    try parseFaceVertex(
                        String($0),
                        positionCount: positions.count,
                        texcoordCount: texcoords.count,
                        normalCount: normals.count
                    )
                }
                guard faceVertices.count >= 3 else { continue }

                for i in 1..<(faceVertices.count - 1) {
                    for vertex in [faceVertices[0], faceVertices[i], faceVertices[i + 1]] {
                        let index = append(vertex: vertex, to: &primitives[primitiveIndex])
                        primitives[primitiveIndex].indices.append(index)
                    }
                }
            default:
                continue
            }
        }

        primitives.removeAll { $0.indices.isEmpty }
        guard !positions.isEmpty, !primitives.isEmpty else {
            throw OBJToGLTFConverter.ExportError.invalidOBJ("No mesh faces were found.")
        }

        return ParsedOBJ(
            positions: positions,
            normals: normals,
            texcoords: texcoords,
            primitives: primitives,
            materialLibraries: materialLibraries
        )
    }

    private static func append(vertex: OBJVertex, to primitive: inout OBJPrimitive) -> UInt32 {
        if let existing = primitive.vertexLookup[vertex] {
            return UInt32(existing)
        }
        let index = UInt32(primitive.vertices.count)
        primitive.vertices.append(vertex)
        primitive.vertexLookup[vertex] = index
        return index
    }

    private static func parseVector3(_ values: ArraySlice<String>) -> SIMD3<Float>? {
        let floats = values.prefix(3).compactMap(Float.init)
        guard floats.count == 3 else { return nil }
        return SIMD3(floats[0], floats[1], floats[2])
    }

    private static func parseVector2(_ values: ArraySlice<String>) -> SIMD2<Float>? {
        let floats = values.prefix(2).compactMap(Float.init)
        guard floats.count == 2 else { return nil }
        return SIMD2(floats[0], floats[1])
    }

    private static func parseFaceVertex(
        _ value: String,
        positionCount: Int,
        texcoordCount: Int,
        normalCount: Int
    ) throws -> OBJVertex {
        let fields = value.split(separator: "/", omittingEmptySubsequences: false)
        guard let position = resolveIndex(fields[safe: 0], count: positionCount) else {
            throw OBJToGLTFConverter.ExportError.invalidOBJ("A face references a missing position.")
        }
        return OBJVertex(
            position: position,
            texcoord: resolveIndex(fields[safe: 1], count: texcoordCount),
            normal: resolveIndex(fields[safe: 2], count: normalCount)
        )
    }

    private static func resolveIndex(_ raw: Substring?, count: Int) -> Int? {
        guard let raw, !raw.isEmpty, let value = Int(raw) else { return nil }
        let index = value > 0 ? value - 1 : count + value
        return (0..<count).contains(index) ? index : nil
    }

    private struct PrimitiveKey: Hashable {
        var materialName: String?
    }
}

private struct GLTFDocument {
    private var parsedOBJ: ParsedOBJ
    private var sourceDirectory: URL
    private var materials: [String: MTLMaterial]
    private var buffer = Data()
    private var bufferViews: [[String: Any]] = []
    private var accessors: [[String: Any]] = []
    private var images: [[String: Any]] = []
    private var textures: [[String: Any]] = []
    private var gltfMaterials: [[String: Any]] = []
    private var imageBufferViewIndices: [URL: Int] = [:]
    private var copiedImageNames: Set<String> = []

    init(parsedOBJ: ParsedOBJ, sourceDirectory: URL) throws {
        self.parsedOBJ = parsedOBJ
        self.sourceDirectory = sourceDirectory
        self.materials = try MTLParser.parse(libraries: parsedOBJ.materialLibraries, sourceDirectory: sourceDirectory)
    }

    mutating func writeGLTF(to url: URL) throws {
        let json = try buildJSON(destinationURL: url, embedImages: false)
        let data = try JSONSerialization.data(withJSONObject: json, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: url, options: .atomic)
        try buffer.write(to: url.deletingPathExtension().appendingPathExtension("bin"), options: .atomic)
    }

    mutating func writeGLB(to url: URL) throws {
        let json = try buildJSON(destinationURL: url, embedImages: true)
        var jsonData = try JSONSerialization.data(withJSONObject: json, options: [.sortedKeys])
        pad(&jsonData, byte: 0x20)

        var binaryChunk = buffer
        pad(&binaryChunk, byte: 0x00)

        var glb = Data()
        glb.appendUInt32(0x46546C67)
        glb.appendUInt32(2)
        glb.appendUInt32(UInt32(12 + 8 + jsonData.count + 8 + binaryChunk.count))
        glb.appendUInt32(UInt32(jsonData.count))
        glb.appendUInt32(0x4E4F534A)
        glb.append(jsonData)
        glb.appendUInt32(UInt32(binaryChunk.count))
        glb.appendUInt32(0x004E4942)
        glb.append(binaryChunk)
        try glb.write(to: url, options: .atomic)
    }

    private mutating func buildJSON(destinationURL: URL, embedImages: Bool) throws -> [String: Any] {
        buffer = Data()
        bufferViews = []
        accessors = []
        images = []
        textures = []
        gltfMaterials = []
        imageBufferViewIndices = [:]
        copiedImageNames = []

        let materialIndices = try buildMaterials(destinationURL: destinationURL, embedImages: embedImages)
        let meshPrimitives = parsedOBJ.primitives.map { primitive -> [String: Any] in
            let positionAccessor = addAccessor(
                data: floatData(primitive.vertices.flatMap { parsedOBJ.positions[$0.position].array }),
                componentType: 5126,
                type: "VEC3",
                count: primitive.vertices.count,
                target: 34962,
                min: bounds(for: primitive.vertices.map { parsedOBJ.positions[$0.position] }).min,
                max: bounds(for: primitive.vertices.map { parsedOBJ.positions[$0.position] }).max
            )

            var attributes: [String: Any] = ["POSITION": positionAccessor]

            if primitive.vertices.allSatisfy({ $0.normal != nil }) {
                attributes["NORMAL"] = addAccessor(
                    data: floatData(primitive.vertices.flatMap { parsedOBJ.normals[$0.normal!].array }),
                    componentType: 5126,
                    type: "VEC3",
                    count: primitive.vertices.count,
                    target: 34962
                )
            }

            if primitive.vertices.allSatisfy({ $0.texcoord != nil }) {
                attributes["TEXCOORD_0"] = addAccessor(
                    data: floatData(primitive.vertices.flatMap {
                        let uv = parsedOBJ.texcoords[$0.texcoord!]
                        return [uv.x, 1 - uv.y]
                    }),
                    componentType: 5126,
                    type: "VEC2",
                    count: primitive.vertices.count,
                    target: 34962
                )
            }

            let indexAccessor = addAccessor(
                data: indexData(primitive.indices),
                componentType: 5125,
                type: "SCALAR",
                count: primitive.indices.count,
                target: 34963
            )

            var result: [String: Any] = [
                "attributes": attributes,
                "indices": indexAccessor,
                "mode": 4
            ]

            if let materialName = primitive.materialName,
               let materialIndex = materialIndices[materialName] {
                result["material"] = materialIndex
            }

            return result
        }

        var json: [String: Any] = [
            "asset": [
                "version": "2.0",
                "generator": "Reality Pulse OBJToGLTFConverter"
            ],
            "scene": 0,
            "scenes": [["nodes": [0]]],
            "nodes": [["mesh": 0]],
            "meshes": [["primitives": meshPrimitives]],
            "buffers": [[
                "byteLength": buffer.count,
                "uri": destinationURL.deletingPathExtension().appendingPathExtension("bin").lastPathComponent
            ]],
            "bufferViews": bufferViews,
            "accessors": accessors
        ]

        if embedImages {
            json["buffers"] = [["byteLength": buffer.count]]
        }
        if !images.isEmpty { json["images"] = images }
        if !textures.isEmpty { json["textures"] = textures }
        if !gltfMaterials.isEmpty { json["materials"] = gltfMaterials }
        return json
    }

    private mutating func buildMaterials(destinationURL: URL, embedImages: Bool) throws -> [String: Int] {
        var indices: [String: Int] = [:]
        for primitive in parsedOBJ.primitives {
            guard let name = primitive.materialName, indices[name] == nil else { continue }
            let material = materials[name] ?? MTLMaterial(name: name)
            var pbr: [String: Any] = ["baseColorFactor": material.baseColorFactor]

            if let textureURL = material.diffuseTextureURL(sourceDirectory: sourceDirectory) {
                let imageIndex = try addImage(textureURL, destinationURL: destinationURL, embedImages: embedImages)
                let textureIndex = textures.count
                textures.append(["source": imageIndex])
                pbr["baseColorTexture"] = ["index": textureIndex]
            }

            var gltfMaterial: [String: Any] = [
                "name": name,
                "pbrMetallicRoughness": pbr
            ]

            if material.alpha < 1 {
                gltfMaterial["alphaMode"] = "BLEND"
            }

            indices[name] = gltfMaterials.count
            gltfMaterials.append(gltfMaterial)
        }
        return indices
    }

    private mutating func addImage(_ sourceURL: URL, destinationURL: URL, embedImages: Bool) throws -> Int {
        if embedImages {
            if let existing = imageBufferViewIndices[sourceURL] {
                return existing
            }
            let data = try Data(contentsOf: sourceURL)
            let bufferView = addBufferView(data: data, target: nil)
            let index = images.count
            images.append([
                "bufferView": bufferView,
                "mimeType": mimeType(for: sourceURL)
            ])
            imageBufferViewIndices[sourceURL] = index
            return index
        }

        let destinationName = uniqueImageName(sourceURL.lastPathComponent)
        let outputURL = destinationURL.deletingLastPathComponent().appendingPathComponent(destinationName)
        if sourceURL.standardizedFileURL != outputURL.standardizedFileURL {
            if FileManager.default.fileExists(atPath: outputURL.path()) {
                try FileManager.default.removeItem(at: outputURL)
            }
            try FileManager.default.copyItem(at: sourceURL, to: outputURL)
        }
        let index = images.count
        images.append(["uri": destinationName])
        return index
    }

    private mutating func addAccessor(
        data: Data,
        componentType: Int,
        type: String,
        count: Int,
        target: Int?,
        min: [Float]? = nil,
        max: [Float]? = nil
    ) -> Int {
        let bufferView = addBufferView(data: data, target: target)
        var accessor: [String: Any] = [
            "bufferView": bufferView,
            "componentType": componentType,
            "count": count,
            "type": type
        ]
        if let min { accessor["min"] = min }
        if let max { accessor["max"] = max }
        accessors.append(accessor)
        return accessors.count - 1
    }

    private mutating func addBufferView(data: Data, target: Int?) -> Int {
        alignBuffer()
        let offset = buffer.count
        buffer.append(data)
        pad(&buffer, byte: 0x00)

        var bufferView: [String: Any] = [
            "buffer": 0,
            "byteOffset": offset,
            "byteLength": data.count
        ]
        if let target { bufferView["target"] = target }
        bufferViews.append(bufferView)
        return bufferViews.count - 1
    }

    private mutating func alignBuffer() {
        pad(&buffer, byte: 0x00)
    }

    private mutating func uniqueImageName(_ name: String) -> String {
        if !copiedImageNames.contains(name) {
            copiedImageNames.insert(name)
            return name
        }

        let url = URL(fileURLWithPath: name)
        let base = url.deletingPathExtension().lastPathComponent
        let ext = url.pathExtension
        var candidate = name
        var suffix = 2
        while copiedImageNames.contains(candidate) {
            candidate = ext.isEmpty ? "\(base)-\(suffix)" : "\(base)-\(suffix).\(ext)"
            suffix += 1
        }
        copiedImageNames.insert(candidate)
        return candidate
    }

    private func floatData(_ values: [Float]) -> Data {
        var data = Data()
        values.forEach { data.appendFloat32($0) }
        return data
    }

    private func indexData(_ values: [UInt32]) -> Data {
        var data = Data()
        values.forEach { data.appendUInt32($0) }
        return data
    }

    private func bounds(for positions: [SIMD3<Float>]) -> (min: [Float], max: [Float]) {
        var minVector = positions[0]
        var maxVector = positions[0]
        for position in positions.dropFirst() {
            minVector = simd_min(minVector, position)
            maxVector = simd_max(maxVector, position)
        }
        return (minVector.array, maxVector.array)
    }

    private func mimeType(for url: URL) -> String {
        switch url.pathExtension.lowercased() {
        case "jpg", "jpeg": return "image/jpeg"
        case "webp": return "image/webp"
        default: return "image/png"
        }
    }
}

private struct MTLMaterial {
    var name: String
    var diffuseColor = SIMD3<Float>(1, 1, 1)
    var alpha: Float = 1
    var diffuseTexture: String?

    var baseColorFactor: [Float] {
        [diffuseColor.x, diffuseColor.y, diffuseColor.z, alpha]
    }

    func diffuseTextureURL(sourceDirectory: URL) -> URL? {
        guard let diffuseTexture else { return nil }
        let url = sourceDirectory.appendingPathComponent(diffuseTexture)
        return FileManager.default.fileExists(atPath: url.path()) ? url : nil
    }
}

private enum MTLParser {
    static func parse(libraries: [String], sourceDirectory: URL) throws -> [String: MTLMaterial] {
        var materials: [String: MTLMaterial] = [:]

        for library in libraries {
            let url = sourceDirectory.appendingPathComponent(library)
            guard FileManager.default.fileExists(atPath: url.path()) else { continue }
            let text = try String(contentsOf: url, encoding: .utf8)
            var current: MTLMaterial?

            func saveCurrent() {
                if let current {
                    materials[current.name] = current
                }
            }

            for rawLine in text.components(separatedBy: .newlines) {
                let line = rawLine.trimmingCharacters(in: .whitespaces)
                if line.isEmpty || line.hasPrefix("#") { continue }

                let parts = line.split(whereSeparator: { $0 == " " || $0 == "\t" }).map(String.init)
                guard let keyword = parts.first else { continue }

                switch keyword {
                case "newmtl":
                    saveCurrent()
                    current = MTLMaterial(name: parts.dropFirst().joined(separator: " "))
                case "Kd":
                    let floats = parts.dropFirst().prefix(3).compactMap(Float.init)
                    if floats.count == 3 {
                        current?.diffuseColor = SIMD3(floats[0], floats[1], floats[2])
                    }
                case "d":
                    if let alpha = parts.dropFirst().first.flatMap(Float.init) {
                        current?.alpha = alpha
                    }
                case "Tr":
                    if let transparency = parts.dropFirst().first.flatMap(Float.init) {
                        current?.alpha = 1 - transparency
                    }
                case "map_Kd":
                    current?.diffuseTexture = parseTexturePath(Array(parts.dropFirst()))
                default:
                    continue
                }
            }

            saveCurrent()
        }

        return materials
    }

    private static func parseTexturePath(_ fields: [String]) -> String? {
        guard !fields.isEmpty else { return nil }
        var index = 0
        while index < fields.count {
            let field = fields[index]
            if field.hasPrefix("-") {
                index += optionValueCount(for: field) + 1
            } else {
                break
            }
        }
        guard index < fields.count else { return nil }
        return fields[index...].joined(separator: " ")
    }

    private static func optionValueCount(for option: String) -> Int {
        switch option {
        case "-blendu", "-blendv", "-cc", "-clamp", "-texres":
            return 1
        case "-mm", "-o", "-s", "-t":
            return 3
        default:
            return 0
        }
    }
}

private extension Array {
    subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}

private extension SIMD3 where Scalar == Float {
    var array: [Float] { [x, y, z] }
}

private func pad(_ data: inout Data, byte: UInt8) {
    let padding = (4 - (data.count % 4)) % 4
    if padding > 0 {
        data.append(contentsOf: Array(repeating: byte, count: padding))
    }
}

private extension Data {
    mutating func appendUInt32(_ value: UInt32) {
        var littleEndian = value.littleEndian
        append(Data(bytes: &littleEndian, count: MemoryLayout<UInt32>.size))
    }

    mutating func appendFloat32(_ value: Float) {
        var bitPattern = value.bitPattern.littleEndian
        append(Data(bytes: &bitPattern, count: MemoryLayout<UInt32>.size))
    }
}
