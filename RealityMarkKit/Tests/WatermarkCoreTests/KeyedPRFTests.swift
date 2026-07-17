import XCTest
@testable import WatermarkCore

final class KeyedPRFTests: XCTestCase {
    func testStreamIsDeterministic() {
        var first = KeyedPRF(key: Fixtures.key(seed: 7), context: "rp-wm/test")
        var second = KeyedPRF(key: Fixtures.key(seed: 7), context: "rp-wm/test")
        let firstBytes = (0..<128).map { _ in first.nextByte() }
        let secondBytes = (0..<128).map { _ in second.nextByte() }
        XCTAssertEqual(firstBytes, secondBytes)
    }

    func testDifferentKeysAndContextsDiverge() {
        var base = KeyedPRF(key: Fixtures.key(seed: 7), context: "rp-wm/test")
        var otherKey = KeyedPRF(key: Fixtures.key(seed: 8), context: "rp-wm/test")
        var otherContext = KeyedPRF(key: Fixtures.key(seed: 7), context: "rp-wm/other")

        let baseBytes = (0..<64).map { _ in base.nextByte() }
        XCTAssertNotEqual(baseBytes, (0..<64).map { _ in otherKey.nextByte() })
        XCTAssertNotEqual(baseBytes, (0..<64).map { _ in otherContext.nextByte() })
    }

    func testBitsAreRoughlyBalanced() {
        var prf = KeyedPRF(key: Fixtures.key(seed: 3), context: "rp-wm/test")
        let ones = (0..<10_000).filter { _ in prf.nextBit() }.count
        XCTAssertGreaterThan(ones, 4_700)
        XCTAssertLessThan(ones, 5_300)
    }

    func testBoundedDrawIsInRangeAndCoversValues() {
        var prf = KeyedPRF(key: Fixtures.key(seed: 5), context: "rp-wm/test")
        var seen = Set<Int>()
        for _ in 0..<1_000 {
            let value = prf.next(upperBound: 10)
            XCTAssertTrue((0..<10).contains(value))
            seen.insert(value)
        }
        XCTAssertEqual(seen.count, 10)
    }

    func testPermutationIsValidAndKeyDependent() {
        var first = KeyedPRF(key: Fixtures.key(seed: 1), context: "rp-wm/test")
        var second = KeyedPRF(key: Fixtures.key(seed: 2), context: "rp-wm/test")
        let permutation = first.permutation(count: 64)
        XCTAssertEqual(permutation.sorted(), Array(0..<64))
        XCTAssertNotEqual(permutation, second.permutation(count: 64))
    }
}
