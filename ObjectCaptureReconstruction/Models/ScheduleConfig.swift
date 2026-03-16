/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Configuration for delayed-start and allowed-hours scheduling.
*/

import Foundation

/// Defines when the scheduler is allowed to process jobs.
struct ScheduleConfig: Codable, Equatable {

    /// Optional one-time delayed start. The scheduler will not begin processing
    /// until this date has passed.
    var delayedStart: Date?

    /// Optional recurring allowed-hours window expressed as start/end hours (0-23).
    /// When set, the scheduler only processes during these hours, pausing outside.
    /// The window can wrap midnight (e.g., start=22, end=6 means 10 PM to 6 AM).
    var allowedWindowStart: Int?
    var allowedWindowEnd: Int?

    var hasAllowedWindow: Bool {
        allowedWindowStart != nil && allowedWindowEnd != nil
    }

    // MARK: - Time-window evaluation

    /// Whether the scheduler is allowed to run at the given time.
    func isWithinAllowedWindow(now: Date = Date()) -> Bool {
        // If there's a delayed start that hasn't passed, not allowed.
        if let start = delayedStart, now < start {
            return false
        }

        // If no window is configured, always allowed.
        guard let startHour = allowedWindowStart, let endHour = allowedWindowEnd else {
            return true
        }

        let hour = Calendar.current.component(.hour, from: now)

        if startHour <= endHour {
            // Same-day window (e.g., 9–17)
            return hour >= startHour && hour < endHour
        } else {
            // Wraps midnight (e.g., 22–6)
            return hour >= startHour || hour < endHour
        }
    }

    /// Returns the next `Date` at which the allowed window opens,
    /// or `nil` if no window is configured.
    func nextWindowOpen(after date: Date = Date()) -> Date? {
        // Handle delayed start first.
        if let start = delayedStart, date < start {
            // If there's also a window, the actual start is whichever is later.
            if let windowDate = nextAllowedWindowDate(after: start) {
                return isWithinHourWindow(at: start) ? start : windowDate
            }
            return start
        }

        return nextAllowedWindowDate(after: date)
    }

    // MARK: - Private helpers

    private func nextAllowedWindowDate(after date: Date) -> Date? {
        guard let startHour = allowedWindowStart, allowedWindowEnd != nil else { return nil }

        let calendar = Calendar.current
        var components = calendar.dateComponents([.year, .month, .day], from: date)
        components.hour = startHour
        components.minute = 0
        components.second = 0

        guard var candidate = calendar.date(from: components) else { return nil }

        // If the candidate is in the past or we're already in the window, advance a day.
        if candidate <= date || isWithinHourWindow(at: date) {
            candidate = calendar.date(byAdding: .day, value: 1, to: candidate) ?? candidate
        }

        return candidate
    }

    private func isWithinHourWindow(at date: Date) -> Bool {
        guard let startHour = allowedWindowStart, let endHour = allowedWindowEnd else { return true }
        let hour = Calendar.current.component(.hour, from: date)
        if startHour <= endHour {
            return hour >= startHour && hour < endHour
        } else {
            return hour >= startHour || hour < endHour
        }
    }
}
