/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Codable mirror of PhotogrammetrySession.Configuration for JSON persistence.
*/

import Foundation
import RealityKit

/// A fully `Codable` representation of `PhotogrammetrySession.Configuration`
/// and its nested types. Convert to/from the real framework type with
/// `toSessionConfiguration()` and `init(from:)`.
///
/// Every stored field exists on all platforms so a saved job decodes the same
/// way everywhere. Only the framework conversions are platform specific: Object
/// Capture on iPhone and iPad has no mesh primitive selection and no custom
/// detail specification, so those settings are stored but not applied there.
struct CodableSessionConfiguration: Codable, Equatable {

    // MARK: - Top-level fields

    var meshPrimitive: CodableMeshPrimitive = .triangle
    var isObjectMaskingEnabled: Bool = true
    var ignoreBoundingBox: Bool = false
    var customDetailSpecification: CodableCustomDetailSpecification = CodableCustomDetailSpecification()

    // MARK: - Nested Codable types

    enum CodableMeshPrimitive: String, Codable, CaseIterable {
        case triangle
        case quad

        #if os(macOS)
        init(from primitive: PhotogrammetrySession.Configuration.MeshPrimitive) {
            switch primitive {
            case .triangle: self = .triangle
            case .quad: self = .quad
            @unknown default: self = .triangle
            }
        }

        var toFrameworkType: PhotogrammetrySession.Configuration.MeshPrimitive {
            switch self {
            case .triangle: return .triangle
            case .quad: return .quad
            }
        }
        #endif
    }

    struct CodableCustomDetailSpecification: Codable, Equatable {
        var maximumPolygonCount: UInt = 0
        var outputTextureMapsRawValue: UInt = 1 // diffuseColor by default
        var textureFormat: CodableTextureFormat = .png
        var maximumTextureDimension: CodableTextureDimension = .fourK

        init() {
            #if os(macOS)
            let defaults = PhotogrammetrySession.Configuration.CustomDetailSpecification()
            outputTextureMapsRawValue = defaults.outputTextureMaps.rawValue
            maximumPolygonCount = defaults.maximumPolygonCount
            #endif
        }

        #if os(macOS)
        init(from spec: PhotogrammetrySession.Configuration.CustomDetailSpecification) {
            maximumPolygonCount = spec.maximumPolygonCount
            outputTextureMapsRawValue = spec.outputTextureMaps.rawValue
            textureFormat = CodableTextureFormat(from: spec.textureFormat)
            maximumTextureDimension = CodableTextureDimension(from: spec.maximumTextureDimension)
        }

        func toFrameworkType() -> PhotogrammetrySession.Configuration.CustomDetailSpecification {
            var spec = PhotogrammetrySession.Configuration.CustomDetailSpecification()
            spec.maximumPolygonCount = maximumPolygonCount
            spec.outputTextureMaps = .init(rawValue: outputTextureMapsRawValue)
            spec.textureFormat = textureFormat.toFrameworkType
            spec.maximumTextureDimension = maximumTextureDimension.toFrameworkType
            return spec
        }
        #endif
    }

    enum CodableTextureFormat: Codable, Equatable {
        case png
        case jpeg(compressionQuality: Float)

        #if os(macOS)
        init(from format: PhotogrammetrySession.Configuration.CustomDetailSpecification.TextureFormat) {
            switch format {
            case .png:
                self = .png
            case .jpeg(let quality):
                self = .jpeg(compressionQuality: quality)
            @unknown default:
                self = .png
            }
        }

        var toFrameworkType: PhotogrammetrySession.Configuration.CustomDetailSpecification.TextureFormat {
            switch self {
            case .png: return .png
            case .jpeg(let quality): return .jpeg(compressionQuality: quality)
            }
        }
        #endif
    }

    enum CodableTextureDimension: String, Codable, CaseIterable {
        case oneK, twoK, fourK, eightK, sixteenK

        #if os(macOS)
        init(from dimension: PhotogrammetrySession.Configuration.CustomDetailSpecification.TextureDimension) {
            switch dimension {
            case .oneK: self = .oneK
            case .twoK: self = .twoK
            case .fourK: self = .fourK
            case .eightK: self = .eightK
            case .sixteenK: self = .sixteenK
            @unknown default: self = .fourK
            }
        }

        var toFrameworkType: PhotogrammetrySession.Configuration.CustomDetailSpecification.TextureDimension {
            switch self {
            case .oneK: return .oneK
            case .twoK: return .twoK
            case .fourK: return .fourK
            case .eightK: return .eightK
            case .sixteenK: return .sixteenK
            }
        }
        #endif
    }

    // MARK: - Conversion

    init() {}

    init(from config: PhotogrammetrySession.Configuration) {
        isObjectMaskingEnabled = config.isObjectMaskingEnabled
        ignoreBoundingBox = config.ignoreBoundingBox
        #if os(macOS)
        meshPrimitive = CodableMeshPrimitive(from: config.meshPrimitive)
        customDetailSpecification = CodableCustomDetailSpecification(from: config.customDetailSpecification)
        #endif
    }

    func toSessionConfiguration() -> PhotogrammetrySession.Configuration {
        var config = PhotogrammetrySession.Configuration()
        config.isObjectMaskingEnabled = isObjectMaskingEnabled
        config.ignoreBoundingBox = ignoreBoundingBox
        #if os(macOS)
        config.meshPrimitive = meshPrimitive.toFrameworkType
        config.customDetailSpecification = customDetailSpecification.toFrameworkType()
        #endif
        return config
    }
}
