/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
SwiftData model for the labeled watermark key library. A saved key can be
reused across jobs (an ownership stamp shared by many copies) instead of the
default of a fresh per-copy key for every exported file.
*/

import Foundation
import SwiftData
import WatermarkCore

@Model
final class PersistentWatermarkKey {
    @Attribute(.unique) var id: UUID
    @Attribute(.unique) var label: String
    var keyData: Data
    var createdAt: Date
    var lastUsedAt: Date?

    init(id: UUID = UUID(), label: String, key: WatermarkKey, createdAt: Date = Date()) {
        self.id = id
        self.label = label
        self.keyData = key.data
        self.createdAt = createdAt
        self.lastUsedAt = nil
    }

    var watermarkKey: WatermarkKey? {
        try? WatermarkKey(data: keyData)
    }

    var info: WatermarkKeyInfo {
        WatermarkKeyInfo(id: id, label: label, createdAt: createdAt, lastUsedAt: lastUsedAt)
    }
}

/// Key metadata for the UI. Deliberately carries no key material — views never
/// need it and should never display or leak it.
struct WatermarkKeyInfo: Identifiable, Hashable, Sendable {
    let id: UUID
    let label: String
    let createdAt: Date
    let lastUsedAt: Date?
}
