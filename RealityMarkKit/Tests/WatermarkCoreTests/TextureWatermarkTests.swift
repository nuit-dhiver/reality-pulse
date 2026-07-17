import CoreGraphics
import XCTest
@testable import WatermarkCore

final class TextureWatermarkTests: XCTestCase {
    private let parameters = TextureWatermarkParameters()

    /// Photo-like synthetic texture: smooth gradients, structure, and grain.
    static func makeTestImage(width: Int, height: Int, seed: UInt64) -> CGImage {
        var rng = SplitMix64(seed: seed)
        var pixels = [UInt8](repeating: 255, count: width * height * 4)
        for y in 0..<height {
            for x in 0..<width {
                let gradient = 90.0 + 80.0 * Double(x) / Double(width)
                let structure = 40.0 * sin(Double(x) * 0.11) * cos(Double(y) * 0.07)
                let grain = rng.nextGaussian() * 6
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

    static func psnr(_ first: CGImage, _ second: CGImage) -> Double {
        let firstRaster = RGBARaster(image: first)!
        let secondRaster = RGBARaster(image: second)!
        var squaredError = 0.0
        var sampleCount = 0
        for offset in 0..<(firstRaster.width * firstRaster.height * 4) where offset % 4 != 3 {
            let difference = Double(firstRaster.pixels[offset]) - Double(secondRaster.pixels[offset])
            squaredError += difference * difference
            sampleCount += 1
        }
        let meanSquaredError = squaredError / Double(sampleCount)
        return meanSquaredError > 0 ? 10 * log10(255.0 * 255.0 / meanSquaredError) : .infinity
    }

    func testDCTRoundTripIsLossless() {
        var rng = SplitMix64(seed: 8)
        let block = (0..<64).map { _ in Float(rng.nextDouble() * 255) }
        var coefficients = [Float](repeating: 0, count: 64)
        var restored = [Float](repeating: 0, count: 64)
        DCT8x8.forward(block, into: &coefficients)
        DCT8x8.inverse(coefficients, into: &restored)
        for (original, roundTripped) in zip(block, restored) {
            XCTAssertEqual(original, roundTripped, accuracy: 1e-3)
        }
    }

    func testZigzagOrderIsAPermutationStartingAtDC() {
        XCTAssertEqual(DCT8x8.zigzagOrder.sorted(), Array(0..<64))
        XCTAssertEqual(DCT8x8.zigzagOrder[0], 0)
        XCTAssertEqual(DCT8x8.zigzagOrder[1], 1)
        XCTAssertEqual(DCT8x8.zigzagOrder[2], 8)
    }

    func testEmbedDetectRoundTrip() throws {
        let image = Self.makeTestImage(width: 256, height: 256, seed: 1)
        let key = Fixtures.key(seed: 21)
        let stamped = try TextureWatermarker.embed(image: image, key: key, parameters: parameters)

        let detection = TextureWatermarker.detect(image: stamped, key: key, parameters: parameters)
        XCTAssertEqual(detection.chipCount, 32 * 32 * 22)
        XCTAssertGreaterThan(detection.zScore, 6)
        XCTAssertLessThan(detection.pValue, 1e-9)
    }

    func testEmbedIsImperceptible() throws {
        let image = Self.makeTestImage(width: 256, height: 256, seed: 1)
        let stamped = try TextureWatermarker.embed(
            image: image, key: Fixtures.key(seed: 21), parameters: parameters
        )
        XCTAssertGreaterThanOrEqual(Self.psnr(image, stamped), 45)
    }

    func testWrongKeyDetectsAtChanceLevel() throws {
        let image = Self.makeTestImage(width: 256, height: 256, seed: 1)
        let stamped = try TextureWatermarker.embed(
            image: image, key: Fixtures.key(seed: 21), parameters: parameters
        )

        for seed in 60..<70 {
            let detection = TextureWatermarker.detect(
                image: stamped, key: Fixtures.key(seed: UInt8(seed)), parameters: parameters
            )
            XCTAssertGreaterThan(detection.pValue, 1e-3, "wrong key seed \(seed) z=\(detection.zScore)")
            XCTAssertLessThan(abs(detection.zScore), 4)
        }
    }

    func testUnmarkedImageDetectsAtChanceLevel() {
        let image = Self.makeTestImage(width: 256, height: 256, seed: 1)
        let detection = TextureWatermarker.detect(
            image: image, key: Fixtures.key(seed: 21), parameters: parameters
        )
        XCTAssertGreaterThan(detection.pValue, 1e-3)
    }

    func testDetectionSurvivesDownscaleWhenRestoredToRecordedSize() throws {
        let image = Self.makeTestImage(width: 256, height: 256, seed: 1)
        let key = Fixtures.key(seed: 21)
        let stamped = try TextureWatermarker.embed(image: image, key: key, parameters: parameters)

        let downscaled = RGBARaster.resample(stamped, width: 192, height: 192)!
        let detection = TextureWatermarker.detect(
            image: downscaled, key: key, parameters: parameters, originalSize: (width: 256, height: 256)
        )
        XCTAssertLessThan(detection.pValue, 1e-6)
    }
}
