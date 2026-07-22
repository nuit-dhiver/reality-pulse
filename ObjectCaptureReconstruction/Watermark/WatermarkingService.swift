/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Glue between the export pipeline and RealityMarkKit: per-file stamping
instructions, embed outcomes, and export-record assembly. The algorithm is
public (see docs/ARCHITECTURE.md); all security rests on the per-copy key
stored in the export record.
*/

import CryptoKit
import Foundation
import ModelFileIO
import WatermarkCore

/// Per-file stamping instruction handed to an export converter. Every exported
/// file gets its own fresh key, so every distributed file maps to exactly one
/// export record.
struct WatermarkStamp: Sendable {
    let key: WatermarkKey
    /// Label when this stamp reuses a saved library key; nil for a fresh
    /// per-copy key.
    var keyLabel: String?
    let geometryParameters: GeometryWatermarkParameters
    let textureParameters: TextureWatermarkParameters

    init(
        key: WatermarkKey,
        keyLabel: String? = nil,
        geometryParameters: GeometryWatermarkParameters = GeometryWatermarkParameters(),
        textureParameters: TextureWatermarkParameters = TextureWatermarkParameters()
    ) {
        self.key = key
        self.keyLabel = keyLabel
        self.geometryParameters = geometryParameters
        self.textureParameters = textureParameters
    }

    /// A brand-new key, unique to this one file — full per-copy traceability.
    static func fresh() -> WatermarkStamp {
        WatermarkStamp(key: .random())
    }

    /// Reuse a saved library key. Every file stamped with it carries the same
    /// mark, so a leak traces to the label rather than to one copy.
    static func reusing(_ shared: SharedWatermarkKey) -> WatermarkStamp {
        WatermarkStamp(key: shared.key, keyLabel: shared.label)
    }

    /// One file's stamp: the shared key when the job selected one, otherwise
    /// a fresh key.
    static func next(sharedKey: SharedWatermarkKey?) -> WatermarkStamp {
        sharedKey.map(reusing) ?? .fresh()
    }
}

/// A saved library key resolved for use by an export run.
struct SharedWatermarkKey: Sendable {
    let key: WatermarkKey
    let label: String
}

/// What a converter actually embedded (channels can be skipped, e.g. geometry
/// on meshes too small to bin).
struct WatermarkStampOutcome {
    var geometry: WatermarkRecord.GeometryChannelInfo?
    var texture: WatermarkRecord.TextureChannelInfo?

    static let none = WatermarkStampOutcome(geometry: nil, texture: nil)

    var channels: [String] {
        var channels: [String] = []
        if geometry != nil { channels.append(WatermarkRecord.Channel.geometry) }
        if texture != nil { channels.append(WatermarkRecord.Channel.texture) }
        return channels
    }
}

enum WatermarkingError: LocalizedError {
    /// Watermarking was requested but no channel could be embedded — the file
    /// would be untraceable, so the export must not silently succeed.
    case nothingEmbedded(String)

    var errorDescription: String? {
        switch self {
        case .nothingEmbedded(let filename):
            return "Provenance watermark could not be embedded in \(filename) (model too small or textures not encodable)."
        }
    }
}

enum WatermarkingService {
    /// Assemble the persistent record for a stamped export file. Returns nil
    /// when nothing was embedded (no record should exist for unmarked files).
    nonisolated static func record(
        for stamp: WatermarkStamp,
        outcome: WatermarkStampOutcome,
        jobId: UUID,
        detailLevel: CodableDetailLevel,
        format: String,
        fileURL: URL
    ) throws -> WatermarkRecord? {
        let channels = outcome.channels
        guard !channels.isEmpty else { return nil }
        return WatermarkRecord(
            jobId: jobId,
            format: format,
            detailLevel: detailLevel.rawValue,
            filename: fileURL.lastPathComponent,
            filePath: fileURL.path,
            key: stamp.key,
            keyLabel: stamp.keyLabel,
            channels: channels,
            geometry: outcome.geometry,
            texture: outcome.texture,
            fileSHA256: try sha256Hex(of: fileURL)
        )
    }

    nonisolated static func sha256Hex(of url: URL) throws -> String {
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    /// Stamp an encoded texture payload: decode → embed → re-encode as PNG.
    /// Returns nil (leaving the original untouched) if the payload cannot be
    /// decoded — a failed stamp must never break the export.
    nonisolated static func stampedTexture(
        pngData: Data,
        name: String,
        stamp: WatermarkStamp
    ) -> (data: Data, image: WatermarkRecord.TextureChannelInfo.Image)? {
        guard let image = try? ImageCodec.decode(pngData),
              let stamped = try? TextureWatermarker.embed(
                  image: image,
                  key: stamp.key,
                  parameters: stamp.textureParameters
              ),
              let data = try? ImageCodec.pngData(from: stamped) else { return nil }
        let info = WatermarkRecord.TextureChannelInfo.Image(
            name: name,
            semantic: "baseColor",
            width: stamped.width,
            height: stamped.height
        )
        return (data, info)
    }
}
