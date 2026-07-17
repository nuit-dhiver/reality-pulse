import CoreGraphics
import XCTest
@testable import ModelFileIO
@testable import WatermarkCore

final class ImageCodecTests: XCTestCase {
    /// Photo-like synthetic texture (mirrors the WatermarkCoreTests fixture).
    private func makeTestImage(width: Int, height: Int) -> CGImage {
        var state: UInt64 = 99
        func nextDouble() -> Double {
            state &+= 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return Double((z ^ (z >> 31)) >> 11) * (1.0 / 9_007_199_254_740_992.0)
        }

        var pixels = [UInt8](repeating: 255, count: width * height * 4)
        for y in 0..<height {
            for x in 0..<width {
                let gradient = 90.0 + 80.0 * Double(x) / Double(width)
                let structure = 40.0 * sin(Double(x) * 0.11) * cos(Double(y) * 0.07)
                let grain = (nextDouble() - 0.5) * 14
                let offset = (y * width + x) * 4
                pixels[offset] = UInt8(min(255, max(0, gradient + structure + grain)))
                pixels[offset + 1] = UInt8(min(255, max(0, gradient * 0.8 + structure + grain)))
                pixels[offset + 2] = UInt8(min(255, max(0, gradient * 0.6 - structure + grain)))
            }
        }
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
        return pixels.withUnsafeMutableBytes { buffer in
            CGContext(
                data: buffer.baseAddress,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: width * 4,
                space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            )!.makeImage()!
        }
    }

    func testPNGRoundTripIsLossless() throws {
        let image = makeTestImage(width: 64, height: 64)
        let decoded = try ImageCodec.decode(ImageCodec.pngData(from: image))
        XCTAssertEqual(RGBARaster(image: decoded)!.pixels, RGBARaster(image: image)!.pixels)
    }

    func testImageFormatDetection() throws {
        let image = makeTestImage(width: 16, height: 16)
        XCTAssertEqual(ImageCodec.imageFormat(of: try ImageCodec.pngData(from: image)), .png)
        XCTAssertEqual(ImageCodec.imageFormat(of: try ImageCodec.jpegData(from: image, quality: 0.9)), .jpeg)
        XCTAssertNil(ImageCodec.imageFormat(of: Data([0x00, 0x01, 0x02, 0x03])))
    }

    func testWatermarkSurvivesJPEGReencode() throws {
        let image = makeTestImage(width: 256, height: 256)
        let key = try WatermarkKey(data: Data(repeating: 21, count: WatermarkKey.byteCount))
        let parameters = TextureWatermarkParameters()
        let stamped = try TextureWatermarker.embed(image: image, key: key, parameters: parameters)

        // PNG → JPEG(q0.8) → decode, then detect.
        let pngDecoded = try ImageCodec.decode(ImageCodec.pngData(from: stamped))
        let jpegDecoded = try ImageCodec.decode(ImageCodec.jpegData(from: pngDecoded, quality: 0.8))

        let detection = TextureWatermarker.detect(image: jpegDecoded, key: key, parameters: parameters)
        XCTAssertLessThan(detection.pValue, 1e-6)

        let wrongKey = try WatermarkKey(data: Data(repeating: 22, count: WatermarkKey.byteCount))
        let wrongDetection = TextureWatermarker.detect(image: jpegDecoded, key: wrongKey, parameters: parameters)
        XCTAssertGreaterThan(wrongDetection.pValue, 1e-3)
    }
}
