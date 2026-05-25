/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
SwiftData model tracking one-time persistence migrations.
*/

import Foundation
import SwiftData

@Model
final class PersistentMigrationState {
    @Attribute(.unique) var id: String
    var migratedAt: Date

    init(id: String, migratedAt: Date = Date()) {
        self.id = id
        self.migratedAt = migratedAt
    }
}
