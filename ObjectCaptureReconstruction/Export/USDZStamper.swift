/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Texture-channel provenance stamping for finished USDZ files. The USDZ is a
stored (uncompressed) zip: texture image entries are decoded, watermarked, and
re-encoded in place while the `.usdc` geometry bytes are copied verbatim, so
the primary deliverable's structure and materials are never touched.
*/

import Foundation
import ModelFileIO
import ModelIO
import WatermarkCore
import os

private let logger = Logger(subsystem: ObjectCaptureReconstructionApp.subsystem,
                            category: "USDZStamper")

enum USDZStamper {

    enum StamperError: LocalizedError {
        case noStampableTextures(String)

        var errorDescription: String? {
            switch self {
            case .noStampableTextures(let name):
                return "No stampable base-color textures found in \(name)."
            }
        }
    }

    struct StampResult {
        var stampedImages: [WatermarkRecord.TextureChannelInfo.Image]
        var fileSHA256: String
    }

    /// A stamped copy written to a temporary location, not yet swapped in.
    /// Lets the caller persist the provenance record between staging and
    /// committing, so a marked file can never exist without its record.
    struct StagedStamp {
        var temporaryURL: URL
        var stampedImages: [WatermarkRecord.TextureChannelInfo.Image]
        /// SHA-256 of the staged bytes — identical to the destination file's
        /// hash after `commit`.
        var fileSHA256: String
    }

    /// Stamp the base-color textures of a finished USDZ in place, atomically.
    /// Any failure throws before the original file is replaced.
    nonisolated static func stampTextures(usdzURL: URL, stamp: WatermarkStamp) throws -> StampResult {
        let staged = try stage(usdzURL: usdzURL, stamp: stamp)
        do {
            try commit(staged, to: usdzURL)
        } catch {
            discard(staged)
            throw error
        }
        return StampResult(stampedImages: staged.stampedImages, fileSHA256: staged.fileSHA256)
    }

    /// Produce a stamped copy in a temporary location; the original stays
    /// untouched until `commit`.
    nonisolated static func stage(usdzURL: URL, stamp: WatermarkStamp) throws -> StagedStamp {
        var archive = try UsdzArchive.read(url: usdzURL)
        let baseColorNames = baseColorTextureNames(usdzURL: usdzURL)

        var stampedImages: [WatermarkRecord.TextureChannelInfo.Image] = []
        for index in archive.entries.indices {
            let entry = archive.entries[index]
            guard let format = ImageCodec.imageFormat(of: entry.data),
                  isBaseColorEntry(entry.name, knownNames: baseColorNames) else { continue }

            let image = try ImageCodec.decode(entry.data)
            let stamped = try TextureWatermarker.embed(
                image: image,
                key: stamp.key,
                parameters: stamp.textureParameters
            )
            switch format {
            case .png:
                archive.entries[index].data = try ImageCodec.pngData(from: stamped)
            case .jpeg:
                archive.entries[index].data = try ImageCodec.jpegData(from: stamped, quality: 0.95)
            }
            stampedImages.append(WatermarkRecord.TextureChannelInfo.Image(
                name: entry.name,
                semantic: "baseColor",
                width: stamped.width,
                height: stamped.height
            ))
        }

        guard !stampedImages.isEmpty else {
            throw StamperError.noStampableTextures(usdzURL.lastPathComponent)
        }

        let temporaryURL = FileManager.default.temporaryDirectory
            .appending(path: UUID().uuidString + ".usdz")
        try archive.write(to: temporaryURL)

        return StagedStamp(
            temporaryURL: temporaryURL,
            stampedImages: stampedImages,
            fileSHA256: try WatermarkingService.sha256Hex(of: temporaryURL)
        )
    }

    /// Atomically replace the destination with the staged copy.
    nonisolated static func commit(_ staged: StagedStamp, to usdzURL: URL) throws {
        _ = try FileManager.default.replaceItemAt(usdzURL, withItemAt: staged.temporaryURL)
        logger.log("Stamped \(staged.stampedImages.count) texture(s) in \(usdzURL.lastPathComponent, privacy: .public)")
    }

    nonisolated static func discard(_ staged: StagedStamp) {
        try? FileManager.default.removeItem(at: staged.temporaryURL)
    }

    /// Base-color texture filenames according to ModelIO's material graph.
    /// Empty when ModelIO can't say (the name heuristic then decides alone).
    private nonisolated static func baseColorTextureNames(usdzURL: URL) -> Set<String> {
        let asset = MDLAsset(url: usdzURL)
        var names = Set<String>()
        for index in 0..<asset.count {
            guard let object = asset.object(at: index) as? MDLObject else { continue }
            collectBaseColorNames(from: object, into: &names)
        }
        return names
    }

    private nonisolated static func collectBaseColorNames(from object: MDLObject, into names: inout Set<String>) {
        if let mesh = object as? MDLMesh {
            for submesh in mesh.submeshes ?? [] {
                guard let submesh = submesh as? MDLSubmesh,
                      let material = submesh.material,
                      let property = material.property(with: .baseColor),
                      property.type == .texture else { continue }
                if let name = property.textureSamplerValue?.texture?.name, !name.isEmpty {
                    names.insert((name as NSString).lastPathComponent)
                }
                if let path = property.urlValue?.lastPathComponent {
                    names.insert(path)
                }
            }
        }
        for child in object.children.objects {
            collectBaseColorNames(from: child, into: &names)
        }
    }

    /// Object Capture texture names: base color has no channel suffix, other
    /// maps carry norm/rough/metal/ao/… markers.
    private nonisolated static let nonBaseColorMarkers = [
        "norm", "rough", "metal", "ao", "occl", "disp", "opacity", "emissive"
    ]

    private nonisolated static func isBaseColorEntry(_ entryName: String, knownNames: Set<String>) -> Bool {
        let filename = (entryName as NSString).lastPathComponent
        if !knownNames.isEmpty {
            return knownNames.contains(filename)
        }
        let lowered = filename.lowercased()
        return !nonBaseColorMarkers.contains { lowered.contains($0) }
    }
}
