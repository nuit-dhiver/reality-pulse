import CoreGraphics
import Foundation

public struct TextureDetectionResult: Codable, Sendable, Equatable {
    /// Normalized correlation between the keyed chip sequence and the observed
    /// mid-band coefficients (zScore / √chipCount).
    public var correlation: Double
    /// Σ chipᵢ·coefficientᵢ / √(Σ coefficientᵢ²) — standard normal under the
    /// null hypothesis that the image was not marked with this key.
    public var zScore: Double
    /// One-sided normal tail P[Z ≥ zScore].
    public var pValue: Double
    public var chipCount: Int
}

/// Blind keyed additive spread-spectrum watermark in the mid-band DCT
/// coefficients of the luma plane.
///
/// Each complete 8×8 block carries one keyed ±1 chip per mid-band coefficient.
/// The amplitude is far below visibility (PSNR ≥ ~45 dB) but with tens of
/// thousands of chips the correlation statistic survives PNG/JPEG re-encoding
/// and mild rescaling (the detector resamples the suspect image back to the
/// recorded original size before correlating).
public enum TextureWatermarker {
    static let chipsContext = "rp-wm/1/tex/chips"

    // MARK: - Embedding

    public static func embed(
        image: CGImage,
        key: WatermarkKey,
        parameters: TextureWatermarkParameters
    ) throws -> CGImage {
        guard var raster = RGBARaster(image: image) else { throw WatermarkError.unsupportedImage }

        var luma = raster.lumaPlane()
        let original = luma
        modulateBlocks(luma: &luma, width: raster.width, height: raster.height, key: key, parameters: parameters) {
            dct, chip in
            dct + parameters.amplitude * chip
        }

        raster.addLumaDelta(current: luma, original: original)
        guard let stamped = raster.makeImage() else { throw WatermarkError.unsupportedImage }
        return stamped
    }

    // MARK: - Detection

    /// Pass the record's original pixel size so a rescaled suspect image is
    /// resampled back onto the embedding block grid first.
    public static func detect(
        image: CGImage,
        key: WatermarkKey,
        parameters: TextureWatermarkParameters,
        originalSize: (width: Int, height: Int)? = nil
    ) -> TextureDetectionResult {
        var sourceImage = image
        if let originalSize,
           originalSize.width > 0, originalSize.height > 0,
           originalSize.width != image.width || originalSize.height != image.height,
           let resampled = RGBARaster.resample(image, width: originalSize.width, height: originalSize.height) {
            sourceImage = resampled
        }

        guard let raster = RGBARaster(image: sourceImage) else {
            return TextureDetectionResult(correlation: 0, zScore: 0, pValue: 1, chipCount: 0)
        }

        var luma = raster.lumaPlane()
        var chipProduct = 0.0
        var coefficientEnergy = 0.0
        var chipCount = 0
        modulateBlocks(luma: &luma, width: raster.width, height: raster.height, key: key, parameters: parameters) {
            dct, chip in
            chipProduct += Double(chip) * Double(dct)
            coefficientEnergy += Double(dct) * Double(dct)
            chipCount += 1
            return dct
        }

        guard chipCount > 0, coefficientEnergy > .ulpOfOne else {
            return TextureDetectionResult(correlation: 0, zScore: 0, pValue: 0.5, chipCount: chipCount)
        }
        let zScore = chipProduct / coefficientEnergy.squareRoot()
        return TextureDetectionResult(
            correlation: zScore / Double(chipCount).squareRoot(),
            zScore: zScore,
            pValue: WatermarkStatistics.normalUpperTailPValue(z: zScore),
            chipCount: chipCount
        )
    }

    // MARK: - Shared block scan

    /// Walks every complete block in row-major order, visiting the mid-band
    /// coefficients in zigzag order with their keyed chips — the exact same
    /// traversal for embed and detect, so the PRF streams stay aligned.
    private static func modulateBlocks(
        luma: inout [Float],
        width: Int,
        height: Int,
        key: WatermarkKey,
        parameters: TextureWatermarkParameters,
        transform: (Float, Float) -> Float
    ) {
        let blockSize = parameters.blockSize
        guard blockSize == 8 else { return }
        let midBand = DCT8x8.zigzagPositions(in: parameters.midBandRange)
        guard !midBand.isEmpty else { return }

        let blocksX = width / blockSize
        let blocksY = height / blockSize
        var prf = KeyedPRF(key: key, context: chipsContext)
        var block = [Float](repeating: 0, count: blockSize * blockSize)
        var coefficients = [Float](repeating: 0, count: blockSize * blockSize)

        for blockY in 0..<blocksY {
            for blockX in 0..<blocksX {
                let originX = blockX * blockSize
                let originY = blockY * blockSize
                for row in 0..<blockSize {
                    let sourceOffset = (originY + row) * width + originX
                    for column in 0..<blockSize {
                        block[row * blockSize + column] = luma[sourceOffset + column]
                    }
                }

                DCT8x8.forward(block, into: &coefficients)
                var changed = false
                for position in midBand {
                    let chip = prf.nextChip()
                    let updated = transform(coefficients[position], chip)
                    if updated != coefficients[position] {
                        coefficients[position] = updated
                        changed = true
                    }
                }
                guard changed else { continue }

                DCT8x8.inverse(coefficients, into: &block)
                for row in 0..<blockSize {
                    let destinationOffset = (originY + row) * width + originX
                    for column in 0..<blockSize {
                        luma[destinationOffset + column] = block[row * blockSize + column]
                    }
                }
            }
        }
    }
}

/// Orthonormal 8×8 DCT-II with precomputed basis.
enum DCT8x8 {
    static let size = 8

    static let basis: [Float] = {
        var basis = [Float](repeating: 0, count: size * size)
        for k in 0..<size {
            let scale = k == 0 ? (1.0 / Double(size)).squareRoot() : (2.0 / Double(size)).squareRoot()
            for n in 0..<size {
                basis[k * size + n] = Float(scale * cos(Double.pi * (2 * Double(n) + 1) * Double(k) / (2 * Double(size))))
            }
        }
        return basis
    }()

    /// Row-major zigzag scan positions of an 8×8 block.
    static let zigzagOrder: [Int] = {
        var order = [Int]()
        for diagonal in 0..<(2 * size - 1) {
            var indices = [Int]()
            for row in 0..<size {
                let column = diagonal - row
                if (0..<size).contains(column) {
                    indices.append(row * size + column)
                }
            }
            order.append(contentsOf: diagonal % 2 == 0 ? indices.reversed() : indices)
        }
        return order
    }()

    static func zigzagPositions(in range: ClosedRange<Int>) -> [Int] {
        let clamped = max(0, range.lowerBound)...min(zigzagOrder.count - 1, range.upperBound)
        guard clamped.lowerBound <= clamped.upperBound else { return [] }
        return clamped.map { zigzagOrder[$0] }
    }

    /// D = C · X · Cᵀ
    static func forward(_ block: [Float], into output: inout [Float]) {
        let temporary = multiplyAB(basis, block)
        output = multiplyABt(temporary, basis)
    }

    /// X = Cᵀ · D · C
    static func inverse(_ coefficients: [Float], into output: inout [Float]) {
        let temporary = multiplyAtB(basis, coefficients)
        output = multiplyAB(temporary, basis)
    }

    private static func multiplyAB(_ a: [Float], _ b: [Float]) -> [Float] {
        var output = [Float](repeating: 0, count: size * size)
        for row in 0..<size {
            for column in 0..<size {
                var sum: Float = 0
                for k in 0..<size {
                    sum += a[row * size + k] * b[k * size + column]
                }
                output[row * size + column] = sum
            }
        }
        return output
    }

    private static func multiplyABt(_ a: [Float], _ b: [Float]) -> [Float] {
        var output = [Float](repeating: 0, count: size * size)
        for row in 0..<size {
            for column in 0..<size {
                var sum: Float = 0
                for k in 0..<size {
                    sum += a[row * size + k] * b[column * size + k]
                }
                output[row * size + column] = sum
            }
        }
        return output
    }

    private static func multiplyAtB(_ a: [Float], _ b: [Float]) -> [Float] {
        var output = [Float](repeating: 0, count: size * size)
        for row in 0..<size {
            for column in 0..<size {
                var sum: Float = 0
                for k in 0..<size {
                    sum += a[k * size + row] * b[k * size + column]
                }
                output[row * size + column] = sum
            }
        }
        return output
    }
}

/// RGBA8 pixel buffer with luma extraction and delta write-back.
struct RGBARaster {
    let width: Int
    let height: Int
    var pixels: [UInt8]

    init?(image: CGImage) {
        let width = image.width
        let height = image.height
        guard width > 0, height > 0 else { return nil }

        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        let drawn = pixels.withUnsafeMutableBytes { buffer -> Bool in
            guard let context = CGContext(
                data: buffer.baseAddress,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: width * 4,
                space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            ) else { return false }
            context.interpolationQuality = .none
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard drawn else { return nil }

        self.width = width
        self.height = height
        self.pixels = pixels
    }

    static func resample(_ image: CGImage, width: Int, height: Int) -> CGImage? {
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return nil }
        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        return context.makeImage()
    }

    /// BT.601 luma on the 0–255 scale.
    func lumaPlane() -> [Float] {
        var luma = [Float](repeating: 0, count: width * height)
        for pixel in 0..<(width * height) {
            let offset = pixel * 4
            luma[pixel] = 0.299 * Float(pixels[offset])
                + 0.587 * Float(pixels[offset + 1])
                + 0.114 * Float(pixels[offset + 2])
        }
        return luma
    }

    /// Distributes the per-pixel luma change equally onto R, G, B with clamping.
    mutating func addLumaDelta(current: [Float], original: [Float]) {
        for pixel in 0..<(width * height) {
            let delta = current[pixel] - original[pixel]
            guard abs(delta) > 1e-4 else { continue }
            let offset = pixel * 4
            for channel in 0..<3 {
                let value = Float(pixels[offset + channel]) + delta
                pixels[offset + channel] = UInt8(min(255, max(0, value.rounded())))
            }
        }
    }

    func makeImage() -> CGImage? {
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        var pixels = pixels
        return pixels.withUnsafeMutableBytes { buffer -> CGImage? in
            guard let context = CGContext(
                data: buffer.baseAddress,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: width * 4,
                space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            ) else { return nil }
            return context.makeImage()
        }
    }
}
