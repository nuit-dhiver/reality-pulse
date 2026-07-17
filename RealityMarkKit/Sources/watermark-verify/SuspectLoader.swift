import CoreGraphics
import Foundation
import ModelFileIO
import ModelIO
import WatermarkCore
import simd

/// Whatever detection-relevant content could be pulled out of a suspect file.
struct Suspect {
    var positions: [SIMD3<Float>]
    var images: [(name: String, image: CGImage)]
}

enum SuspectLoaderError: Error, CustomStringConvertible {
    case unsupportedFileType(String)
    case nothingExtractable

    var description: String {
        switch self {
        case .unsupportedFileType(let ext):
            return "Unsupported suspect file type '.\(ext)' (expected usdz, glb, gltf, ply, png, or jpg)."
        case .nothingExtractable:
            return "No geometry or images could be extracted from the suspect file."
        }
    }
}

enum SuspectLoader {
    static func load(url: URL) throws -> Suspect {
        switch url.pathExtension.lowercased() {
        case "usdz":
            return try loadUSDZ(url: url)
        case "glb", "gltf":
            let asset = try MiniGLTFReader.read(url: url)
            let images = asset.baseColorImages.enumerated().compactMap { index, data in
                (try? ImageCodec.decode(data)).map { ("image\(index)", $0) }
            }
            return Suspect(positions: asset.positions, images: images)
        case "ply":
            return Suspect(positions: try PLYReader.readPositions(url: url), images: [])
        case "png", "jpg", "jpeg":
            let image = try ImageCodec.decode(Data(contentsOf: url))
            return Suspect(positions: [], images: [(url.lastPathComponent, image)])
        case let ext:
            throw SuspectLoaderError.unsupportedFileType(ext)
        }
    }

    // MARK: - USDZ

    private static func loadUSDZ(url: URL) throws -> Suspect {
        // Texture entries straight from the zip — no USD interpretation needed.
        var images: [(name: String, image: CGImage)] = []
        if let archive = try? UsdzArchive.read(url: url) {
            for entry in archive.entries where ImageCodec.imageFormat(of: entry.data) != nil {
                if let image = try? ImageCodec.decode(entry.data) {
                    images.append((entry.name, image))
                }
            }
        }

        // Geometry via ModelIO (detection is invariant to node transforms that
        // are rigid + uniform scale, so raw vertex positions suffice).
        var positions: [SIMD3<Float>] = []
        let asset = MDLAsset(url: url)
        for index in 0..<asset.count {
            collectPositions(from: asset.object(at: index), into: &positions)
        }

        guard !positions.isEmpty || !images.isEmpty else {
            throw SuspectLoaderError.nothingExtractable
        }
        return Suspect(positions: positions, images: images)
    }

    private static func collectPositions(from object: MDLObject, into positions: inout [SIMD3<Float>]) {
        if let mesh = object as? MDLMesh,
           let attribute = mesh.vertexAttributeData(forAttributeNamed: MDLVertexAttributePosition, as: .float3) {
            let base = attribute.map.bytes
            for vertex in 0..<mesh.vertexCount {
                let pointer = base.advanced(by: vertex * attribute.stride)
                    .assumingMemoryBound(to: Float.self)
                positions.append(SIMD3<Float>(pointer[0], pointer[1], pointer[2]))
            }
        }
        for child in object.children.objects {
            collectPositions(from: child, into: &positions)
        }
    }
}
