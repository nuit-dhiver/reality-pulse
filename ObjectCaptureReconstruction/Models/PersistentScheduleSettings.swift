/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
SwiftData model for persisted scheduler settings.
*/

import Foundation
import SwiftData

@Model
final class PersistentScheduleSettings {
    @Attribute(.unique) var id: String
    var delayedStart: Date?
    var allowedWindowStart: Int?
    var allowedWindowEnd: Int?
    var preventSleepWhileQueueActive: Bool
    var updatedAt: Date

    init(id: String = "main", config: ScheduleConfig) {
        self.id = id
        delayedStart = config.delayedStart
        allowedWindowStart = config.allowedWindowStart
        allowedWindowEnd = config.allowedWindowEnd
        preventSleepWhileQueueActive = config.preventSleepWhileQueueActive
        updatedAt = Date()
    }

    func update(from config: ScheduleConfig) {
        delayedStart = config.delayedStart
        allowedWindowStart = config.allowedWindowStart
        allowedWindowEnd = config.allowedWindowEnd
        preventSleepWhileQueueActive = config.preventSleepWhileQueueActive
        updatedAt = Date()
    }

    var scheduleConfig: ScheduleConfig {
        ScheduleConfig(
            delayedStart: delayedStart,
            allowedWindowStart: allowedWindowStart,
            allowedWindowEnd: allowedWindowEnd,
            preventSleepWhileQueueActive: preventSleepWhileQueueActive
        )
    }
}
