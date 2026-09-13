/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Keeps the system awake while the queue is processing.
*/

import Foundation
import os

#if canImport(UIKit)
import UIKit
#endif

private let logger = Logger(subsystem: ObjectCaptureReconstructionApp.subsystem,
                            category: "QueueAwakeAssertion")

/// Prevents the system from going to sleep while a queue run is active.
///
/// On macOS the assertion is a `ProcessInfo` activity that disables idle system
/// sleep, so the queue keeps running with the display off. On iPhone and iPad,
/// `PhotogrammetrySession` only makes progress while the app is in the
/// foreground, so the assertion disables the idle timer to keep the device
/// awake on the job instead.
@MainActor
final class QueueAwakeAssertion {

    #if os(macOS)
    private var activity: NSObjectProtocol?
    #else
    private var isHoldingIdleTimer = false
    #endif

    /// Whether the assertion is currently held.
    var isHeld: Bool {
        #if os(macOS)
        return activity != nil
        #else
        return isHoldingIdleTimer
        #endif
    }

    func begin() {
        guard !isHeld else { return }

        #if os(macOS)
        activity = ProcessInfo.processInfo.beginActivity(
            options: [.idleSystemSleepDisabled],
            reason: "Reality Pulse queue is active"
        )
        logger.log("Sleep prevention activity started.")
        #else
        UIApplication.shared.isIdleTimerDisabled = true
        isHoldingIdleTimer = true
        logger.log("Idle timer disabled while the queue is active.")
        #endif
    }

    func end() {
        guard isHeld else { return }

        #if os(macOS)
        if let activity {
            ProcessInfo.processInfo.endActivity(activity)
        }
        activity = nil
        logger.log("Sleep prevention activity ended.")
        #else
        UIApplication.shared.isIdleTimerDisabled = false
        isHoldingIdleTimer = false
        logger.log("Idle timer re-enabled.")
        #endif
    }
}
