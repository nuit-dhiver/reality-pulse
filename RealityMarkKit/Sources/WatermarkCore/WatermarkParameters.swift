import Foundation

/// Parameters for the geometry channel (keyed Cho–Prost–Jung vertex-norm
/// distribution watermark). All values are public; they are stored in the
/// export record so detection replays the exact embedding configuration.
public struct GeometryWatermarkParameters: Codable, Sendable, Equatable {
    /// Requested number of norm bins (one embedded bit per bin). The embedder
    /// halves this (floor 16) until every bin can expect `minVerticesPerBin`
    /// members; the count actually used is recorded as `effectiveBinCount`.
    public var binCount: Int
    /// Target |bin mean − 0.5| after embedding (α).
    public var strength: Double
    /// Bins with fewer members are skipped at embed and detect time.
    public var minVerticesPerBin: Int
    /// Bisection iterations for the power-law exponent search.
    public var maxIterations: Int
    /// Acceptable deviation from the target bin mean.
    public var meanTolerance: Double
    /// Fraction trimmed from each end of the norm distribution before the bin
    /// range is derived. Trimmed quantiles barely move under noise, re-save
    /// quantization, or subsampling, whereas the raw min/max shift by several
    /// σ and would drag every bin edge with them. Vertices outside the trimmed
    /// range are excluded from embedding and detection.
    public var trimFraction: Double

    public init(
        binCount: Int = 64,
        strength: Double = 0.04,
        minVerticesPerBin: Int = 32,
        maxIterations: Int = 30,
        meanTolerance: Double = 0.005,
        trimFraction: Double = 0.005
    ) {
        self.binCount = binCount
        self.strength = strength
        self.minVerticesPerBin = minVerticesPerBin
        self.maxIterations = maxIterations
        self.meanTolerance = meanTolerance
        self.trimFraction = trimFraction
    }
}

/// Parameters for the texture channel (keyed additive spread spectrum in the
/// mid-band DCT coefficients of the luma plane).
public struct TextureWatermarkParameters: Codable, Sendable, Equatable {
    /// DCT block edge length in pixels.
    public var blockSize: Int
    /// Zigzag indices (0 = DC) of the coefficients that carry chips.
    public var midBandLowerIndex: Int
    public var midBandUpperIndex: Int
    /// Chip amplitude on the 0–255 luma DCT scale.
    public var amplitude: Float

    public var midBandRange: ClosedRange<Int> { midBandLowerIndex...midBandUpperIndex }

    public init(
        blockSize: Int = 8,
        midBandLowerIndex: Int = 6,
        midBandUpperIndex: Int = 27,
        amplitude: Float = 2.0
    ) {
        self.blockSize = blockSize
        self.midBandLowerIndex = midBandLowerIndex
        self.midBandUpperIndex = midBandUpperIndex
        self.amplitude = amplitude
    }
}
