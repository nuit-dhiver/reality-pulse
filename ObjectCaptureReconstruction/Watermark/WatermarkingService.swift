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
struct WatermarkStamp {
    let key: WatermarkKey
    let geometryParameters: GeometryWatermarkParameters
    let textureParameters: TextureWatermarkParameters

    static func fresh() -> WatermarkStamp {
        WatermarkStamp(
            key: .random(),
            geometryParameters: GeometryWatermarkParameters(),
            textureParameters: TextureWatermarkParameters()
        )
    }
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
