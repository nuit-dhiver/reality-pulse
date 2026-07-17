import XCTest
@testable import WatermarkCore

final class WatermarkStatisticsTests: XCTestCase {
    func testBinomialTailKnownValues() {
        XCTAssertEqual(WatermarkStatistics.binomialTailPValue(matched: 0, total: 10), 1.0, accuracy: 1e-12)
        XCTAssertEqual(
            WatermarkStatistics.binomialTailPValue(matched: 10, total: 10),
            pow(0.5, 10),
            accuracy: 1e-12
        )
        // P[X >= 8], X ~ Bin(10, ½) = (45 + 10 + 1) / 1024
        XCTAssertEqual(
            WatermarkStatistics.binomialTailPValue(matched: 8, total: 10),
            56.0 / 1024.0,
            accuracy: 1e-12
        )
    }

    func testBinomialTailEdgeCases() {
        XCTAssertEqual(WatermarkStatistics.binomialTailPValue(matched: 0, total: 0), 1.0)
        XCTAssertEqual(WatermarkStatistics.binomialTailPValue(matched: -5, total: 10), 1.0, accuracy: 1e-12)
        XCTAssertEqual(
            WatermarkStatistics.binomialTailPValue(matched: 15, total: 10),
            pow(0.5, 10),
            accuracy: 1e-12
        )
    }

    func testNormalUpperTailKnownValues() {
        XCTAssertEqual(WatermarkStatistics.normalUpperTailPValue(z: 0), 0.5, accuracy: 1e-12)
        XCTAssertEqual(WatermarkStatistics.normalUpperTailPValue(z: 1.6448536269514722), 0.05, accuracy: 1e-9)
        XCTAssertEqual(WatermarkStatistics.normalUpperTailPValue(z: 3.09023230616781), 1e-3, accuracy: 1e-9)
    }
}
