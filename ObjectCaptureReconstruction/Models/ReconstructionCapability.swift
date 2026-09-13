/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Describes the on-device Object Capture reconstruction features the current
platform and device support.
*/

import Foundation
import RealityKit

/// Platform and device capabilities for on-device reconstruction.
///
/// macOS exposes every `PhotogrammetrySession.Request.Detail` level. Object
/// Capture on iOS and iPadOS only exposes `.reduced`, so the iPhone and iPad
/// builds offer that single level and refuse jobs that ask for a level this
/// platform cannot produce.
enum ReconstructionCapability {

    /// Detail levels Object Capture can produce on this platform, ordered from
    /// fastest to most detailed.
    static var supportedDetailLevels: [CodableDetailLevel] {
        #if os(macOS)
        return [.preview, .reduced, .medium, .full, .raw, .custom]
        #else
        return [.reduced]
        #endif
    }

    /// The detail level a new job starts with.
    static var defaultDetailLevel: CodableDetailLevel {
        #if os(macOS)
        return .medium
        #else
        return .reduced
        #endif
    }

    /// Whether additional detail levels can be requested alongside the primary one.
    static var supportsMultipleDetailLevels: Bool {
        supportedDetailLevels.count > 1
    }

    /// Whether this platform exposes the custom detail specification, which
    /// controls polygon count, texture maps, texture format, and texture size.
    static var supportsCustomDetailSpecification: Bool {
        #if os(macOS)
        return true
        #else
        return false
        #endif
    }

    /// Whether this platform lets a job choose the output mesh primitive.
    static var supportsMeshPrimitiveSelection: Bool {
        #if os(macOS)
        return true
        #else
        return false
        #endif
    }

    /// Whether the hardware this build is running on can create a
    /// `PhotogrammetrySession`.
    static var isSupportedOnThisDevice: Bool {
        PhotogrammetrySession.isSupported
    }

    /// Message shown when the device itself cannot reconstruct models.
    static var deviceUnsupportedMessage: String {
        #if os(macOS)
        return "This Mac does not support Apple Object Capture reconstruction."
        #else
        return "This device does not support on-device Apple Object Capture reconstruction. Reconstruction needs an iPhone or iPad with Object Capture support."
        #endif
    }

    /// Message shown when a queued job asks for detail levels this platform
    /// cannot produce, for example a job created on a Mac and opened on iPhone.
    static func unsupportedDetailLevelMessage(for levels: [CodableDetailLevel]) -> String {
        let names = levels
            .map(\.displayName)
            .sorted()
            .joined(separator: ", ")
        let supported = supportedDetailLevels
            .map(\.displayName)
            .joined(separator: ", ")
        return "\(names) detail is not available on this device. Supported detail: \(supported)."
    }
}
