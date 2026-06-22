/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Samples the base (diffuse) color of an `MDLMaterial` at a UV coordinate. The
base-color texture is decoded once into an RGBA8 buffer and cached; materials
without a texture fall back to the flat base-color factor (or white).
*/

import Foundation
import ModelIO
import CoreGraphics
import simd

final class TextureColorSampler {

    private let width: Int
    private let height: Int
    private let pixels: [UInt8]?       // RGBA8, row 0 = top
    private let flatColor: SIMD3<Float>

    init(material: MDLMaterial?) {
        var decodedWidth = 0
        var decodedHeight = 0
        var decodedPixels: [UInt8]?
        var fallback = SIMD3<Float>(1, 1, 1)

        if let material, let property = material.property(with: .baseColor) {
            switch property.type {
            case .float4:
                let color = property.float4Value
                fallback = SIMD3<Float>(color.x, color.y, color.z)
            case .float3:
                let color = property.float3Value
                fallback = SIMD3<Float>(color.x, color.y, color.z)
            case .texture:
                if let texture = property.textureSamplerValue?.texture,
                   let decoded = Self.decode(texture) {
                    decodedWidth = decoded.width
                    decodedHeight = decoded.height
                    decodedPixels = decoded.pixels
                }
            default:
                break
            }
        }

        self.width = decodedWidth
        self.height = decodedHeight
        self.pixels = decodedPixels
        self.flatColor = fallback
    }

    /// Color in `[0, 1]` at the given raw (un-flipped) UV. Wraps out-of-range UVs.
    func color(at uv: SIMD2<Float>) -> SIMD3<Float> {
        guard let pixels, width > 0, height > 0 else { return flatColor }

        var u = uv.x - floor(uv.x)
        var v = uv.y - floor(uv.y)
        if !u.isFinite { u = 0 }
        if !v.isFinite { v = 0 }

        // CGImage origin is top-left while MDL UV origin is bottom-left → flip V.
        let x = min(width - 1, max(0, Int(u * Float(width))))
        let y = min(height - 1, max(0, Int((1 - v) * Float(height))))
        let offset = (y * width + x) * 4
        return SIMD3<Float>(
            Float(pixels[offset]) / 255,
            Float(pixels[offset + 1]) / 255,
            Float(pixels[offset + 2]) / 255
        )
    }

    // MARK: - Texture decoding

    private static func decode(_ texture: MDLTexture) -> (width: Int, height: Int, pixels: [UInt8])? {
        guard let cgImage = texture.imageFromTexture()?.takeUnretainedValue() else {
            return nil
        }
        let width = cgImage.width
        let height = cgImage.height
        guard width > 0, height > 0 else { return nil }

        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        let success = pixels.withUnsafeMutableBytes { buffer -> Bool in
            guard let context = CGContext(
                data: buffer.baseAddress,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: width * 4,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            ) else {
                return false
            }
            context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        return success ? (width, height, pixels) : nil
    }
}
