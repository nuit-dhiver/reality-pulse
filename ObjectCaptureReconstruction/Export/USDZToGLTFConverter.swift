/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Converts Object Capture USDZ models to glTF 2.0 (.gltf) or binary glb (.glb)
using Model I/O for import and a custom glTF serializer.
*/

import Foundation
import ModelIO
import ImageIO
import UniformTypeIdentifiers
import WatermarkCore
import simd
import os

private let logger = Logger(subsystem: ObjectCaptureReconstructionApp.subsystem,
                            category: "USDZToGLTFConverter")

enum USDZToGLTFConverterError: LocalizedError {
    case failedToLoadAsset(URL)
    case noMeshesFound
    case unsupportedIndexType
    case unsupportedExportFormat(ModelExportFormat)
    case failedToEncodeTexture(String)
    case failedToWriteOutput(URL)

    var errorDescription: String? {
        switch self {
        case .failedToLoadAsset(let url):
            return "Failed to load USDZ asset at \(url.lastPathComponent)."
        case .noMeshesFound:
            return "No mesh geometry found in the USDZ file."
        case .unsupportedIndexType:
            return "Unsupported index buffer type in mesh."
        case .unsupportedExportFormat(let format):
            return "USDZToGLTFConverter cannot produce \(format.displayName)."
        case .failedToEncodeTexture(let name):
            return "Failed to encode texture \(name)."
        case .failedToWriteOutput(let url):
            return "Failed to write output to \(url.lastPathComponent)."
        }
    }
}

enum USDZToGLTFConverter {

    /// Convert a `.usdz` file to the requested glTF container format.
    /// When `watermark` is set, the vertex positions (all meshes as one point
    /// set) and base-color textures are stamped with the per-copy key before
    /// serialization; the returned outcome says which channels were embedded.
    @discardableResult
    nonisolated static func convert(
        usdzURL: URL,
        format: ModelExportFormat,
        outputURL: URL,
        watermark: WatermarkStamp? = nil
    ) throws -> WatermarkStampOutcome {
        let asset = MDLAsset(url: usdzURL)
        guard asset.count > 0 else {
            throw USDZToGLTFConverterError.failedToLoadAsset(usdzURL)
        }
        asset.loadTextures()

        var meshes: [(mesh: MDLMesh, transform: matrix_float4x4)] = []
        for index in 0..<asset.count {
            if let object = asset.object(at: index) as? MDLObject {
                MeshGeometryReader.collectMeshes(from: object, parentTransform: matrix_identity_float4x4, into: &meshes)
            }
        }

        guard !meshes.isEmpty else {
            throw USDZToGLTFConverterError.noMeshesFound
        }

        let builder = GLTFBinaryBuilder()
        var gltfMeshes: [GLTFMesh] = []
        var materials: [GLTFMaterial] = []
        var textures: [GLTFTexture] = []
        var images: [GLTFImage] = []
        var materialCache: [String: Int] = [:]
        var textureCache: [ObjectIdentifier: Int] = [:]

        let defaultSamplerIndex = 0
        let samplers = [GLTFSampler(
            magFilter: GLTFConstants.filterLinear,
            minFilter: GLTFConstants.filterLinearMipmapLinear,
            wrapS: GLTFConstants.wrapRepeat,
            wrapT: GLTFConstants.wrapRepeat
        )]

        // Pass 1: extract all vertex data so the geometry watermark can treat
        // every mesh as one point set (global centroid and bins).
        var preparedMeshes: [PreparedMesh] = []
        for (mesh, transform) in meshes {
            mesh.addNormals(withAttributeNamed: MDLVertexAttributeNormal, creaseThreshold: 0.5)
            if mesh.vertexAttributeData(forAttributeNamed: MDLVertexAttributeTextureCoordinate) != nil {
                mesh.addTangentBasis(
                    forTextureCoordinateAttributeNamed: MDLVertexAttributeTextureCoordinate,
                    tangentAttributeNamed: MDLVertexAttributeTangent,
                    bitangentAttributeNamed: MDLVertexAttributeBitangent
                )
            }

            let positions = extractFloat3(from: mesh, attribute: MDLVertexAttributePosition, transform: transform)
            guard !positions.isEmpty else { continue }

            preparedMeshes.append(PreparedMesh(
                mesh: mesh,
                positions: positions,
                normals: extractFloat3(from: mesh, attribute: MDLVertexAttributeNormal, transform: transform, isDirection: true),
                tangents: extractFloat4(from: mesh, attribute: MDLVertexAttributeTangent),
                uvs: extractTexCoords(from: mesh, attribute: MDLVertexAttributeTextureCoordinate)
            ))
        }

        var geometryInfo: WatermarkRecord.GeometryChannelInfo?
        if let watermark {
            var allPositions = preparedMeshes.flatMap(\.positions)
            let embedResult = GeometryWatermarker.embed(
                positions: &allPositions,
                key: watermark.key,
                parameters: watermark.geometryParameters
            )
            if embedResult.isEmbedded {
                var cursor = 0
                for index in preparedMeshes.indices {
                    let count = preparedMeshes[index].positions.count
                    preparedMeshes[index].positions = Array(allPositions[cursor..<(cursor + count)])
                    cursor += count
                }
                geometryInfo = WatermarkRecord.GeometryChannelInfo(
                    parameters: watermark.geometryParameters,
                    effectiveBinCount: embedResult.effectiveBinCount,
                    embeddedBits: embedResult.embeddedBits
                )
            } else {
                logger.log("Geometry watermark skipped for \(outputURL.lastPathComponent): too few vertices or degenerate shape.")
            }
        }

        // Pass 2: serialize.
        var stampedImages: [WatermarkRecord.TextureChannelInfo.Image] = []
        for prepared in preparedMeshes {
            let positions = prepared.positions

            let positionAccessor = builder.appendFloatBuffer(
                positions.flatMap { [$0.x, $0.y, $0.z] },
                componentsPerElement: 3,
                min: vectorMin(positions),
                max: vectorMax(positions)
            )
            let normalAccessor = prepared.normals.isEmpty ? nil : builder.appendFloatBuffer(
                prepared.normals.flatMap { [$0.x, $0.y, $0.z] },
                componentsPerElement: 3
            )
            // The V flip applied to UVs reverses the bitangent direction, so negate
            // the stored handedness to keep normal maps oriented correctly.
            let tangentAccessor = prepared.tangents.isEmpty ? nil : builder.appendFloatBuffer(
                prepared.tangents.flatMap { [$0.x, $0.y, $0.z, -$0.w] },
                componentsPerElement: 4
            )
            let texCoordAccessor = prepared.uvs.isEmpty ? nil : builder.appendFloatBuffer(
                prepared.uvs.flatMap { [$0.x, $0.y] },
                componentsPerElement: 2
            )

            var primitives: [GLTFPrimitive] = []

            for submesh in prepared.mesh.submeshes ?? [] {
                guard let submesh = submesh as? MDLSubmesh else { continue }

                let indexAccessor = try appendIndices(from: submesh, builder: builder)
                let materialIndex = materialIndex(
                    for: submesh.material,
                    materials: &materials,
                    materialCache: &materialCache,
                    textures: &textures,
                    images: &images,
                    textureCache: &textureCache,
                    builder: builder,
                    outputDirectory: outputURL.deletingLastPathComponent(),
                    embedImages: format == .glb,
                    defaultSamplerIndex: defaultSamplerIndex,
                    watermark: watermark,
                    stampedImages: &stampedImages
                )

                var attributes: [String: Int] = [
                    "POSITION": positionAccessor
                ]
                if let normalAccessor { attributes["NORMAL"] = normalAccessor }
                if let tangentAccessor { attributes["TANGENT"] = tangentAccessor }
                if let texCoordAccessor { attributes["TEXCOORD_0"] = texCoordAccessor }

                primitives.append(GLTFPrimitive(
                    attributes: attributes,
                    indices: indexAccessor,
                    material: materialIndex,
                    mode: GLTFConstants.primitiveModeTriangles
                ))
            }

            if !primitives.isEmpty {
                gltfMeshes.append(GLTFMesh(primitives: primitives))
            }
        }

        guard !gltfMeshes.isEmpty else {
            throw USDZToGLTFConverterError.noMeshesFound
        }

        let rootNode = GLTFNode(mesh: 0, children: nil, matrix: nil)
        let scene = GLTFScene(nodes: [0])

        var document = GLTFDocument(
            asset: GLTFAssetInfo(version: "2.0", generator: "Reality Pulse"),
            scene: 0,
            scenes: [scene],
            nodes: [rootNode],
            meshes: gltfMeshes,
            accessors: builder.allAccessors,
            bufferViews: builder.allBufferViews,
            buffers: [GLTFBuffer(byteLength: builder.data.count, uri: nil)],
            materials: materials.isEmpty ? nil : materials,
            textures: textures.isEmpty ? nil : textures,
            images: images.isEmpty ? nil : images,
            samplers: samplers
        )

        var textureInfo: WatermarkRecord.TextureChannelInfo?
        if let watermark, !stampedImages.isEmpty {
            textureInfo = WatermarkRecord.TextureChannelInfo(
                parameters: watermark.textureParameters,
                images: stampedImages
            )
        }

        switch format {
        case .glb:
            try GLTFWriter.writeGLB(document: document, binaryData: builder.data, to: outputURL)
        case .gltf:
            try GLTFWriter.writeGLTF(document: document, binaryData: builder.data, to: outputURL)
        case .gaussianSplat:
            // Splats are produced by `SplatSampleGenerator`, not this converter.
            throw USDZToGLTFConverterError.unsupportedExportFormat(format)
        }

        logger.log("Exported \(outputURL.lastPathComponent) from \(usdzURL.lastPathComponent)")
        return WatermarkStampOutcome(geometry: geometryInfo, texture: textureInfo)
    }

    /// Vertex data extracted in pass 1, serialized in pass 2 (with the
    /// geometry watermark applied to `positions` in between).
    private struct PreparedMesh {
        let mesh: MDLMesh
        var positions: [SIMD3<Float>]
        let normals: [SIMD3<Float>]
        let tangents: [SIMD4<Float>]
        let uvs: [SIMD2<Float>]
    }

    // MARK: - Vertex extraction

    // The raw byte-reading lives in `MeshGeometryReader`; these thin wrappers
    // apply the glTF-specific conventions (world transform, V flip) on top.

    private nonisolated static func extractFloat3(
        from mesh: MDLMesh,
        attribute name: String,
        transform: matrix_float4x4,
        isDirection: Bool = false
    ) -> [SIMD3<Float>] {
        MeshGeometryReader.readFloat3(mesh, attribute: name, transform: transform, isDirection: isDirection)
    }

    private nonisolated static func extractFloat4(
        from mesh: MDLMesh,
        attribute name: String
    ) -> [SIMD4<Float>] {
        MeshGeometryReader.readFloat4(mesh, attribute: name)
    }

    /// Extract UV coordinates, flipping the V axis to match glTF's top-left origin.
    private nonisolated static func extractTexCoords(
        from mesh: MDLMesh,
        attribute name: String
    ) -> [SIMD2<Float>] {
        MeshGeometryReader.readFloat2(mesh, attribute: name).map { SIMD2<Float>($0.x, 1 - $0.y) }
    }

    private nonisolated static func appendIndices(
        from submesh: MDLSubmesh,
        builder: GLTFBinaryBuilder
    ) throws -> Int {
        let indexBuffer = submesh.indexBuffer
        let indexCount = submesh.indexCount
        let indexType = submesh.geometryType

        guard indexType == .triangles else {
            throw USDZToGLTFConverterError.unsupportedIndexType
        }

        let map = indexBuffer.map()
        let bytes = map.bytes

        switch submesh.indexType {
        case .uInt16:
            var indices = [UInt16]()
            indices.reserveCapacity(indexCount)
            for index in 0..<indexCount {
                let offset = index * MemoryLayout<UInt16>.stride
                let value = bytes.advanced(by: offset).assumingMemoryBound(to: UInt16.self).pointee
                indices.append(value)
            }
            return builder.appendBuffer(
                indices,
                target: GLTFConstants.targetElementArrayBuffer,
                componentType: GLTFConstants.componentTypeUnsignedShort,
                type: "SCALAR"
            )

        case .uInt32:
            var indices = [UInt32]()
            indices.reserveCapacity(indexCount)
            for index in 0..<indexCount {
                let offset = index * MemoryLayout<UInt32>.stride
                let value = bytes.advanced(by: offset).assumingMemoryBound(to: UInt32.self).pointee
                indices.append(value)
            }
            return builder.appendBuffer(
                indices,
                target: GLTFConstants.targetElementArrayBuffer,
                componentType: GLTFConstants.componentTypeUnsignedInt,
                type: "SCALAR"
            )

        default:
            throw USDZToGLTFConverterError.unsupportedIndexType
        }
    }

    // MARK: - Materials

    private nonisolated static func materialIndex(
        for material: MDLMaterial?,
        materials: inout [GLTFMaterial],
        materialCache: inout [String: Int],
        textures: inout [GLTFTexture],
        images: inout [GLTFImage],
        textureCache: inout [ObjectIdentifier: Int],
        builder: GLTFBinaryBuilder,
        outputDirectory: URL,
        embedImages: Bool,
        defaultSamplerIndex: Int,
        watermark: WatermarkStamp? = nil,
        stampedImages: inout [WatermarkRecord.TextureChannelInfo.Image]
    ) -> Int? {
        let cacheKey = material?.name ?? "default"
        if let existing = materialCache[cacheKey] {
            return existing
        }

        var pbr = GLTFPBRMetallicRoughness(
            baseColorFactor: [1, 1, 1, 1],
            metallicFactor: 0,
            roughnessFactor: 1,
            baseColorTexture: nil,
            metallicRoughnessTexture: nil
        )
        var normalTexture: GLTFNormalTextureInfo?
        var occlusionTexture: GLTFOcclusionTextureInfo?

        if let material {
            if let baseColor = colorProperty(in: material, semantic: .baseColor) {
                pbr.baseColorFactor = baseColor
            }

            if let textureIndex = textureIndex(
                in: material,
                semantic: .baseColor,
                textures: &textures,
                images: &images,
                textureCache: &textureCache,
                builder: builder,
                outputDirectory: outputDirectory,
                embedImages: embedImages,
                defaultSamplerIndex: defaultSamplerIndex,
                repackAsMetallicRoughness: false,
                watermark: watermark,
                stampedImages: &stampedImages
            ) {
                pbr.baseColorTexture = GLTFTextureInfo(index: textureIndex)
            }

            if let textureIndex = textureIndex(
                in: material,
                semantic: .roughness,
                textures: &textures,
                images: &images,
                textureCache: &textureCache,
                builder: builder,
                outputDirectory: outputDirectory,
                embedImages: embedImages,
                defaultSamplerIndex: defaultSamplerIndex,
                repackAsMetallicRoughness: true,
                stampedImages: &stampedImages
            ) {
                pbr.metallicRoughnessTexture = GLTFTextureInfo(index: textureIndex)
            }

            if let textureIndex = textureIndex(
                in: material,
                semantic: .tangentSpaceNormal,
                textures: &textures,
                images: &images,
                textureCache: &textureCache,
                builder: builder,
                outputDirectory: outputDirectory,
                embedImages: embedImages,
                defaultSamplerIndex: defaultSamplerIndex,
                repackAsMetallicRoughness: false,
                stampedImages: &stampedImages
            ) {
                normalTexture = GLTFNormalTextureInfo(index: textureIndex)
            }

            if let textureIndex = textureIndex(
                in: material,
                semantic: .ambientOcclusion,
                textures: &textures,
                images: &images,
                textureCache: &textureCache,
                builder: builder,
                outputDirectory: outputDirectory,
                embedImages: embedImages,
                defaultSamplerIndex: defaultSamplerIndex,
                repackAsMetallicRoughness: false,
                stampedImages: &stampedImages
            ) {
                occlusionTexture = GLTFOcclusionTextureInfo(index: textureIndex, strength: 1)
            }
        }

        let gltfMaterial = GLTFMaterial(
            name: cacheKey,
            pbrMetallicRoughness: pbr,
            normalTexture: normalTexture,
            occlusionTexture: occlusionTexture,
            doubleSided: true
        )

        let index = materials.count
        materials.append(gltfMaterial)
        materialCache[cacheKey] = index
        return index
    }

    private nonisolated static func colorProperty(
        in material: MDLMaterial,
        semantic: MDLMaterialSemantic
    ) -> [Float]? {
        guard let property = material.property(with: semantic),
              property.type == .float4 else { return nil }
        let color = property.float4Value
        return [color.x, color.y, color.z, color.w]
    }

    private nonisolated static func textureIndex(
        in material: MDLMaterial,
        semantic: MDLMaterialSemantic,
        textures: inout [GLTFTexture],
        images: inout [GLTFImage],
        textureCache: inout [ObjectIdentifier: Int],
        builder: GLTFBinaryBuilder,
        outputDirectory: URL,
        embedImages: Bool,
        defaultSamplerIndex: Int,
        repackAsMetallicRoughness: Bool,
        watermark: WatermarkStamp? = nil,
        stampedImages: inout [WatermarkRecord.TextureChannelInfo.Image]
    ) -> Int? {
        guard let property = material.property(with: semantic),
              property.type == .texture,
              let texture = property.textureSamplerValue?.texture else { return nil }

        let cacheID = ObjectIdentifier(texture)
        if let existing = textureCache[cacheID] {
            return existing
        }

        guard var imageData = encodeTexture(texture, repackAsMetallicRoughness: repackAsMetallicRoughness) else {
            logger.warning("Skipping texture \(texture.name, privacy: .public)")
            return nil
        }

        // Only the base color carries the texture watermark; stamping normal,
        // AO, or roughness maps harms shading for little forensic value.
        if let watermark, semantic == .baseColor {
            let name = sanitizedTextureFilename(texture.name, semantic: semantic)
            if let stamped = WatermarkingService.stampedTexture(
                pngData: imageData, name: name, stamp: watermark
            ) {
                imageData = stamped.data
                stampedImages.append(stamped.image)
            } else {
                logger.warning("Texture watermark skipped for \(name, privacy: .public): payload not decodable.")
            }
        }

        let mimeType = "image/png"
        let imageIndex: Int

        if embedImages {
            let (bufferViewIndex, _) = builder.appendRawData(imageData)
            images.append(GLTFImage(uri: nil, mimeType: mimeType, bufferView: bufferViewIndex))
            imageIndex = images.count - 1
        } else {
            let filename = sanitizedTextureFilename(texture.name, semantic: semantic)
            let imageURL = outputDirectory.appending(path: filename)
            try? imageData.write(to: imageURL, options: [.atomic])
            images.append(GLTFImage(uri: filename, mimeType: nil, bufferView: nil))
            imageIndex = images.count - 1
        }

        textures.append(GLTFTexture(sampler: defaultSamplerIndex, source: imageIndex))
        let textureIndex = textures.count - 1
        textureCache[cacheID] = textureIndex
        return textureIndex
    }

    private nonisolated static func cgImage(from texture: MDLTexture) -> CGImage? {
        texture.imageFromTexture()?.takeUnretainedValue()
    }

    private nonisolated static func encodeTexture(
        _ texture: MDLTexture,
        repackAsMetallicRoughness: Bool
    ) -> Data? {
        if repackAsMetallicRoughness, let repacked = repackRoughnessTexture(texture) {
            return repacked
        }

        if let cgImage = cgImage(from: texture) {
            return pngData(from: cgImage)
        }

        let tempURL = FileManager.default.temporaryDirectory
            .appending(path: UUID().uuidString + ".png")
        do {
            try texture.write(to: tempURL)
            defer { try? FileManager.default.removeItem(at: tempURL) }
            return try Data(contentsOf: tempURL)
        } catch {
            return nil
        }
    }

    /// Repack a grayscale roughness map into glTF metallic-roughness layout (G = roughness, B = metallic).
    private nonisolated static func repackRoughnessTexture(_ texture: MDLTexture) -> Data? {
        guard let source = cgImage(from: texture) else { return nil }

        let width = source.width
        let height = source.height
        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return nil }

        context.draw(source, in: CGRect(x: 0, y: 0, width: width, height: height))
        guard let data = context.data else { return nil }

        let pixels = data.bindMemory(to: UInt8.self, capacity: width * height * 4)
        for index in 0..<(width * height) {
            let offset = index * 4
            let roughness = pixels[offset] // grayscale: R channel
            pixels[offset] = 0
            pixels[offset + 1] = roughness
            pixels[offset + 2] = 0
            pixels[offset + 3] = 255
        }

        guard let output = context.makeImage() else { return nil }
        return pngData(from: output)
    }

    private nonisolated static func pngData(from image: CGImage) -> Data? {
        let data = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(
            data,
            UTType.png.identifier as CFString,
            1,
            nil
        ) else { return nil }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else { return nil }
        return data as Data
    }

    private nonisolated static func sanitizedTextureFilename(
        _ name: String,
        semantic: MDLMaterialSemantic
    ) -> String {
        let base = name.isEmpty ? "texture-\(semantic)" : name
        let sanitized = base
            .replacingOccurrences(of: " ", with: "_")
            .replacingOccurrences(of: "/", with: "_")
        return sanitized.hasSuffix(".png") ? sanitized : sanitized + ".png"
    }

    // MARK: - Bounds helpers

    private nonisolated static func vectorMin(_ vectors: [SIMD3<Float>]) -> [Float]? {
        guard let first = vectors.first else { return nil }
        var minValue = first
        for vector in vectors.dropFirst() {
            minValue = simd_min(minValue, vector)
        }
        return [minValue.x, minValue.y, minValue.z]
    }

    private nonisolated static func vectorMax(_ vectors: [SIMD3<Float>]) -> [Float]? {
        guard let first = vectors.first else { return nil }
        var maxValue = first
        for vector in vectors.dropFirst() {
            maxValue = simd_max(maxValue, vector)
        }
        return [maxValue.x, maxValue.y, maxValue.z]
    }
}

// MARK: - Batch export helper

enum ModelExportService {

    /// Identifies one derived output slot of a job.
    struct ExportSlot: Hashable, Sendable {
        let detailLevel: CodableDetailLevel
        let format: ModelExportFormat
    }

    /// One derived-format export, with its provenance record when the file
    /// was watermarked.
    struct ExportedFile {
        let url: URL
        let format: ModelExportFormat
        let detailLevel: CodableDetailLevel
        let record: WatermarkRecord?
        /// True when an already-current watermarked file was left in place
        /// instead of being re-exported under a new key.
        var isUpToDate: Bool = false
    }

    /// Export all completed USDZ outputs for a job into the requested formats.
    /// With `embedWatermark`, each exported file is stamped with its own fresh
    /// per-copy key and returns a record for the caller to persist.
    ///
    /// `recordedHashes` makes re-finalizing idempotent: a slot whose file on
    /// disk still matches the hash of its newest provenance record is left
    /// untouched, so a retry cannot silently re-key files that are already
    /// marked and recorded.
    /// `sharedKey` reuses one saved library key for every file instead of the
    /// default fresh per-copy key.
    nonisolated static func exportCompletedOutputs(
        for job: ReconstructionJob,
        formats: Set<ModelExportFormat>,
        fileManager: FileManager = .default,
        embedWatermark: Bool = false,
        sharedKey: SharedWatermarkKey? = nil,
        recordedHashes: [ExportSlot: String] = [:]
    ) throws -> [ExportedFile] {
        guard !formats.isEmpty else { return [] }

        var exportedFiles: [ExportedFile] = []
        var exportErrors: [Error] = []

        for level in job.requestedDetailLevels {
            let usdzURL = job.outputURL(for: level)
            guard job.hasCompletedOutputFile(for: level, fileManager: fileManager) else { continue }

            for format in formats.sorted(by: { $0.rawValue < $1.rawValue }) {
                let outputURL = job.exportURL(for: level, format: format)

                if let recordedHash = recordedHashes[ExportSlot(detailLevel: level, format: format)],
                   fileManager.fileExists(atPath: outputURL.path),
                   let currentHash = try? WatermarkingService.sha256Hex(of: outputURL),
                   currentHash == recordedHash {
                    exportedFiles.append(ExportedFile(
                        url: outputURL, format: format, detailLevel: level,
                        record: nil, isUpToDate: true
                    ))
                    logger.log("Skipped \(outputURL.lastPathComponent): already exported and recorded.")
                    continue
                }

                let stamp = embedWatermark ? WatermarkStamp.next(sharedKey: sharedKey) : nil
                do {
                    let outcome: WatermarkStampOutcome
                    switch format {
                    case .gaussianSplat:
                        outcome = try SplatSampleGenerator.generate(
                            usdzURL: usdzURL,
                            outputURL: outputURL,
                            watermark: stamp
                        )
                    case .gltf, .glb:
                        outcome = try USDZToGLTFConverter.convert(
                            usdzURL: usdzURL,
                            format: format,
                            outputURL: outputURL,
                            watermark: stamp
                        )
                    }

                    var record: WatermarkRecord?
                    if let stamp {
                        // A watermarked export without a record is untraceable:
                        // remove the file and fail this export rather than
                        // silently breaking the provenance guarantee.
                        do {
                            record = try WatermarkingService.record(
                                for: stamp,
                                outcome: outcome,
                                jobId: job.id,
                                detailLevel: level,
                                format: format.fileExtension,
                                fileURL: outputURL
                            )
                        } catch {
                            try? fileManager.removeItem(at: outputURL)
                            throw error
                        }
                        guard record != nil else {
                            try? fileManager.removeItem(at: outputURL)
                            throw WatermarkingError.nothingEmbedded(outputURL.lastPathComponent)
                        }
                    }
                    exportedFiles.append(ExportedFile(
                        url: outputURL, format: format, detailLevel: level, record: record
                    ))
                } catch {
                    exportErrors.append(error)
                    logger.warning("Export failed for \(outputURL.lastPathComponent): \(error.localizedDescription)")
                }
            }
        }

        if exportedFiles.isEmpty, let firstError = exportErrors.first {
            throw firstError
        }

        return exportedFiles
    }
}
