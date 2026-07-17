import Foundation
import simd
@testable import WatermarkCore

/// Deterministic PRNG for reproducible test fixtures (same generator family as
/// the app's SurfaceSampler; not security-relevant).
struct SplitMix64 {
    private var state: UInt64

    init(seed: UInt64) {
        state = seed
    }

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    mutating func nextDouble() -> Double {
        Double(next() >> 11) * (1.0 / 9_007_199_254_740_992.0)
    }

    mutating func nextGaussian() -> Double {
        var u = nextDouble()
        if u <= .ulpOfOne { u = .ulpOfOne }
        let v = nextDouble()
        return (-2 * log(u)).squareRoot() * cos(2 * .pi * v)
    }
}

enum Fixtures {
    /// Point cloud with a broad radial-norm distribution (radius uniform in
    /// [0.2, 1.0] times a random direction) — shaped like the photogrammetry
    /// meshes the watermark targets, unlike a thin spherical shell.
    static func blobCloud(count: Int, seed: UInt64) -> [SIMD3<Float>] {
        var rng = SplitMix64(seed: seed)
        var positions = [SIMD3<Float>]()
        positions.reserveCapacity(count)
        for _ in 0..<count {
            var direction = SIMD3<Double>(rng.nextGaussian(), rng.nextGaussian(), rng.nextGaussian())
            let length = simd_length(direction)
            direction = length > .ulpOfOne ? direction / length : SIMD3<Double>(1, 0, 0)
            let radius = 0.2 + 0.8 * rng.nextDouble()
            positions.append(SIMD3<Float>(direction * radius))
        }
        return positions
    }

    static func key(seed: UInt8) -> WatermarkKey {
        try! WatermarkKey(data: Data(repeating: seed, count: WatermarkKey.byteCount))
    }

    static func addGaussianNoise(_ positions: [SIMD3<Float>], sigma: Double, seed: UInt64) -> [SIMD3<Float>] {
        var rng = SplitMix64(seed: seed)
        return positions.map { position in
            position + SIMD3<Float>(
                Float(rng.nextGaussian() * sigma),
                Float(rng.nextGaussian() * sigma),
                Float(rng.nextGaussian() * sigma)
            )
        }
    }

    static func similarityTransform(
        _ positions: [SIMD3<Float>],
        angle: Double,
        axis: SIMD3<Double>,
        scale: Double,
        translation: SIMD3<Double>
    ) -> [SIMD3<Float>] {
        let rotation = simd_quatd(angle: angle, axis: simd_normalize(axis))
        return positions.map { position in
            let transformed = rotation.act(SIMD3<Double>(position)) * scale + translation
            return SIMD3<Float>(transformed)
        }
    }

    static func shuffled(_ positions: [SIMD3<Float>], seed: UInt64) -> [SIMD3<Float>] {
        var rng = SplitMix64(seed: seed)
        var shuffled = positions
        for i in stride(from: shuffled.count - 1, through: 1, by: -1) {
            let j = Int(rng.next() % UInt64(i + 1))
            shuffled.swapAt(i, j)
        }
        return shuffled
    }

    static func subsampled(_ positions: [SIMD3<Float>], keepRatio: Double, seed: UInt64) -> [SIMD3<Float>] {
        var rng = SplitMix64(seed: seed)
        return positions.filter { _ in rng.nextDouble() < keepRatio }
    }
}
