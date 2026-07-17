import Foundation

public enum WatermarkStatistics {
    /// P[X ≥ matched] for X ~ Binomial(total, ½) — the chance an unrelated key
    /// (or unmarked model) matches at least this many bits.
    public static func binomialTailPValue(matched: Int, total: Int) -> Double {
        guard total > 0 else { return 1 }
        let clamped = max(0, min(matched, total))
        let logHalfPowN = -Double(total) * log(2)
        var tail = 0.0
        for k in clamped...total {
            let logChoose = lgamma(Double(total) + 1)
                - lgamma(Double(k) + 1)
                - lgamma(Double(total - k) + 1)
            tail += exp(logChoose + logHalfPowN)
        }
        return min(1, tail)
    }

    /// One-sided upper tail P[Z ≥ z] for Z ~ N(0, 1).
    public static func normalUpperTailPValue(z: Double) -> Double {
        0.5 * erfc(z / 2.0.squareRoot())
    }
}
