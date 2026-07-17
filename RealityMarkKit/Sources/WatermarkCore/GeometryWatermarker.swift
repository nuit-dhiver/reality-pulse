import Foundation
import simd

public struct GeometryEmbedResult: Codable, Sendable, Equatable {
    /// Bin count actually used after auto-fitting to the vertex count.
    /// 0 means the channel was skipped (too few vertices or degenerate shape).
    public var effectiveBinCount: Int
    /// Bins that had enough members to carry a bit.
    public var embeddedBits: Int
    /// Largest vertex displacement introduced, in model units. Displacement is
    /// purely radial and bounded by one bin width, so this stays imperceptible.
    public var maxDisplacement: Float
    public var bboxDiagonal: Float

    public var isEmbedded: Bool { effectiveBinCount > 0 && embeddedBits > 0 }
}

public struct GeometryDetectionResult: Codable, Sendable, Equatable {
    /// Bins that qualified at detection time.
    public var totalBits: Int
    public var matchedBits: Int
    /// P[an unrelated key matches ≥ matchedBits of totalBits] (binomial tail).
    public var pValue: Double
}

/// Blind keyed watermark on the distribution of vertex radial norms
/// (Cho–Prost–Jung, IEEE Trans. Signal Processing 2007), keyed per copy.
///
/// Norms are translation-normalized against the centroid and scale-normalized
/// against the norm range, so detection is invariant to vertex reordering,
/// translation, rotation, and uniform scaling, and needs only the position
/// multiset — no topology. The key drives both the expected bit per bin and a
/// bin permutation; without it the embedded pattern is statistically invisible.
public enum GeometryWatermarker {
    static let bitsContext = "rp-wm/1/geo/bits"
    static let permutationContext = "rp-wm/1/geo/perm"
    static let minBinCount = 16

    // MARK: - Embedding

    public static func embed(
        positions: inout [SIMD3<Float>],
        key: WatermarkKey,
        parameters: GeometryWatermarkParameters
    ) -> GeometryEmbedResult {
        guard let field = RadialField(positions: positions, trimFraction: parameters.trimFraction) else {
            return GeometryEmbedResult(
                effectiveBinCount: 0, embeddedBits: 0,
                maxDisplacement: 0, bboxDiagonal: RadialField.bboxDiagonal(of: positions)
            )
        }

        let binCount = fittedBinCount(
            requested: parameters.binCount,
            vertexCount: positions.count,
            minVerticesPerBin: parameters.minVerticesPerBin
        )
        guard let binCount else {
            return GeometryEmbedResult(
                effectiveBinCount: 0, embeddedBits: 0,
                maxDisplacement: 0, bboxDiagonal: field.bboxDiagonal
            )
        }

        let expectedBits = expectedBits(key: key, binCount: binCount)
        let bins = field.binMembers(binCount: binCount)

        var embeddedBits = 0
        var maxDisplacement = 0.0
        for bin in 0..<binCount {
            let members = bins[bin]
            guard members.count >= parameters.minVerticesPerBin else { continue }

            let locals = members.map { field.localCoordinate(of: $0, bin: bin, binCount: binCount) }
            let target = expectedBits[bin] ? 0.5 + parameters.strength : 0.5 - parameters.strength
            let exponent = solveExponent(
                values: locals,
                target: target,
                maxIterations: parameters.maxIterations,
                tolerance: parameters.meanTolerance
            )

            for (memberIndex, local) in zip(members, locals) {
                let newNorm = field.norm(fromLocal: pow(local, exponent), bin: bin, binCount: binCount)
                let displacement = abs(newNorm - field.norms[memberIndex])
                maxDisplacement = max(maxDisplacement, displacement)
                positions[memberIndex] = field.position(positions[memberIndex], scaledToNorm: newNorm, memberIndex: memberIndex)
            }
            embeddedBits += 1
        }

        return GeometryEmbedResult(
            effectiveBinCount: binCount,
            embeddedBits: embeddedBits,
            maxDisplacement: Float(maxDisplacement),
            bboxDiagonal: field.bboxDiagonal
        )
    }

    // MARK: - Detection

    /// `parameters.binCount` must be the `effectiveBinCount` recorded at embed
    /// time (see `WatermarkRecord.GeometryChannelInfo.detectionParameters`).
    public static func detect(
        positions: [SIMD3<Float>],
        key: WatermarkKey,
        parameters: GeometryWatermarkParameters
    ) -> GeometryDetectionResult {
        guard parameters.binCount > 0,
              let field = RadialField(positions: positions, trimFraction: parameters.trimFraction) else {
            return GeometryDetectionResult(totalBits: 0, matchedBits: 0, pValue: 1)
        }

        let binCount = parameters.binCount
        let expectedBits = expectedBits(key: key, binCount: binCount)
        let bins = field.binMembers(binCount: binCount)

        var totalBits = 0
        var matchedBits = 0
        for bin in 0..<binCount {
            let members = bins[bin]
            guard members.count >= parameters.minVerticesPerBin else { continue }

            var sum = 0.0
            for memberIndex in members {
                sum += field.localCoordinate(of: memberIndex, bin: bin, binCount: binCount)
            }
            let extractedBit = (sum / Double(members.count)) > 0.5
            totalBits += 1
            if extractedBit == expectedBits[bin] { matchedBits += 1 }
        }

        return GeometryDetectionResult(
            totalBits: totalBits,
            matchedBits: matchedBits,
            pValue: WatermarkStatistics.binomialTailPValue(matched: matchedBits, total: totalBits)
        )
    }

    // MARK: - Shared keyed decisions

    /// Expected bit per bin: PRF bit sequence read through a keyed permutation.
    static func expectedBits(key: WatermarkKey, binCount: Int) -> [Bool] {
        var permutationPRF = KeyedPRF(key: key, context: permutationContext)
        let permutation = permutationPRF.permutation(count: binCount)
        var bitsPRF = KeyedPRF(key: key, context: bitsContext)
        let rawBits = (0..<binCount).map { _ in bitsPRF.nextBit() }
        return (0..<binCount).map { rawBits[permutation[$0]] }
    }

    static func fittedBinCount(requested: Int, vertexCount: Int, minVerticesPerBin: Int) -> Int? {
        var binCount = requested
        while vertexCount < binCount * minVerticesPerBin && binCount > minBinCount {
            binCount /= 2
        }
        return vertexCount >= binCount * minVerticesPerBin ? binCount : nil
    }

    /// Bisect k so that mean(xᵏ) hits the target; mean(xᵏ) is monotonically
    /// decreasing in k for x ∈ [0, 1].
    static func solveExponent(values: [Double], target: Double, maxIterations: Int, tolerance: Double) -> Double {
        func mean(exponent: Double) -> Double {
            values.reduce(0) { $0 + pow($1, exponent) } / Double(values.count)
        }
        var low = -6.0
        var high = 6.0
        for _ in 0..<maxIterations {
            let mid = (low + high) / 2
            let value = mean(exponent: exp2(mid))
            if abs(value - target) <= tolerance { return exp2(mid) }
            if value > target {
                low = mid
            } else {
                high = mid
            }
        }
        return exp2((low + high) / 2)
    }
}

/// Centroid-relative radial norms normalized over a trimmed-quantile range —
/// the invariant coordinate system shared by embedding and detection.
struct RadialField {
    let centroid: SIMD3<Double>
    let norms: [Double]
    /// Trimmed lower/upper quantiles of the norm distribution.
    let lowerNorm: Double
    let upperNorm: Double
    let bboxDiagonal: Float

    var normRange: Double { upperNorm - lowerNorm }

    init?(positions: [SIMD3<Float>], trimFraction: Double) {
        guard positions.count > 1 else { return nil }

        var sum = SIMD3<Double>()
        for position in positions {
            sum += SIMD3<Double>(position)
        }
        let centroid = sum / Double(positions.count)

        var norms = [Double]()
        norms.reserveCapacity(positions.count)
        for position in positions {
            norms.append(simd_length(SIMD3<Double>(position) - centroid))
        }

        let sorted = norms.sorted()
        let trim = max(0, min(trimFraction, 0.25))
        let lastIndex = Double(sorted.count - 1)
        let lowerNorm = sorted[Int(trim * lastIndex)]
        let upperNorm = sorted[Int((1 - trim) * lastIndex)]
        guard upperNorm - lowerNorm > .ulpOfOne else { return nil }

        self.centroid = centroid
        self.norms = norms
        self.lowerNorm = lowerNorm
        self.upperNorm = upperNorm
        self.bboxDiagonal = Self.bboxDiagonal(of: positions)
    }

    static func bboxDiagonal(of positions: [SIMD3<Float>]) -> Float {
        guard var minCorner = positions.first else { return 0 }
        var maxCorner = minCorner
        for position in positions {
            minCorner = simd_min(minCorner, position)
            maxCorner = simd_max(maxCorner, position)
        }
        return simd_length(maxCorner - minCorner)
    }

    /// Vertices whose norms fall outside the trimmed range are excluded.
    func binMembers(binCount: Int) -> [[Int]] {
        var bins = [[Int]](repeating: [], count: binCount)
        for (index, norm) in norms.enumerated() {
            let normalized = (norm - lowerNorm) / normRange
            guard normalized >= 0, normalized < 1 else { continue }
            let bin = min(Int(normalized * Double(binCount)), binCount - 1)
            bins[bin].append(index)
        }
        return bins
    }

    /// Position of the norm within its bin, in [0, 1].
    func localCoordinate(of index: Int, bin: Int, binCount: Int) -> Double {
        let normalized = (norms[index] - lowerNorm) / normRange
        return normalized * Double(binCount) - Double(bin)
    }

    func norm(fromLocal local: Double, bin: Int, binCount: Int) -> Double {
        let normalized = (Double(bin) + local) / Double(binCount)
        return lowerNorm + normalized * normRange
    }

    /// Radially rescale a position to the given centroid-relative norm.
    func position(_ position: SIMD3<Float>, scaledToNorm newNorm: Double, memberIndex: Int) -> SIMD3<Float> {
        let oldNorm = norms[memberIndex]
        guard oldNorm > .ulpOfOne else { return position }
        let scaled = centroid + (SIMD3<Double>(position) - centroid) * (newNorm / oldNorm)
        return SIMD3<Float>(scaled)
    }
}
