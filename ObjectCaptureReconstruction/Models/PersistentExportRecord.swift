/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
SwiftData model for provenance-watermark export records. One record per
watermarked exported file, holding the per-copy secret key and the embedding
parameters needed for later verification. Records are append-only provenance
history: they intentionally survive job deletion.
*/

import Foundation
import SwiftData
import WatermarkCore

@Model
final class PersistentExportRecord {
    @Attribute(.unique) var recordId: UUID
    var jobId: UUID
    var formatRawValue: String
    var detailLevelRawValue: String
    var filename: String
    var filePath: String
    var keyData: Data
    /// Label when a reused library key produced this file; nil for a fresh
    /// per-copy key.
    var keyLabel: String?
    var createdAt: Date
    var schemaVersion: Int
    var algorithmVersion: Int
    var channelsData: Data
    var geometryInfoData: Data?
    var textureInfoData: Data?
    var fileSHA256: String

    init(record: WatermarkRecord) throws {
        recordId = record.recordId
        jobId = record.jobId
        formatRawValue = record.format
        detailLevelRawValue = record.detailLevel
        filename = record.filename
        filePath = record.filePath ?? ""
        keyData = record.key
        keyLabel = record.keyLabel
        createdAt = record.createdAt
        schemaVersion = record.schemaVersion
        algorithmVersion = record.algorithmVersion
        channelsData = try JSONEncoder().encode(record.channels)
        geometryInfoData = try record.geometry.map { try JSONEncoder().encode($0) }
        textureInfoData = try record.texture.map { try JSONEncoder().encode($0) }
        fileSHA256 = record.fileSHA256
    }

    func toRecord() throws -> WatermarkRecord {
        var record = WatermarkRecord(
            recordId: recordId,
            jobId: jobId,
            format: formatRawValue,
            detailLevel: detailLevelRawValue,
            filename: filename,
            filePath: filePath.isEmpty ? nil : filePath,
            createdAt: createdAt,
            key: try WatermarkKey(data: keyData),
            keyLabel: keyLabel,
            channels: try JSONDecoder().decode([String].self, from: channelsData),
            geometry: try geometryInfoData.map {
                try JSONDecoder().decode(WatermarkRecord.GeometryChannelInfo.self, from: $0)
            },
            texture: try textureInfoData.map {
                try JSONDecoder().decode(WatermarkRecord.TextureChannelInfo.self, from: $0)
            },
            fileSHA256: fileSHA256
        )
        record.schemaVersion = schemaVersion
        record.algorithmVersion = algorithmVersion
        return record
    }
}
