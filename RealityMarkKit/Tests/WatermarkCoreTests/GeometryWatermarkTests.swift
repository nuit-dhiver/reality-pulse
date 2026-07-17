import XCTest
import simd
@testable import WatermarkCore

final class GeometryWatermarkTests: XCTestCase {
    private let parameters = GeometryWatermarkParameters()

    private func embeddedFixture(
        count: Int = 50_000,
        cloudSeed: UInt64 = 42,
        keySeed: UInt8 = 11
    ) -> (positions: [SIMD3<Float>], key: WatermarkKey, result: GeometryEmbedResult) {
        var positions = Fixtures.blobCloud(count: count, seed: cloudSeed)
        let key = Fixtures.key(seed: keySeed)
        let result = GeometryWatermarker.embed(positions: &positions, key: key, parameters: parameters)
        return (positions, key, result)
    }

    private func detectionParameters(for result: GeometryEmbedResult) -> GeometryWatermarkParameters {
        var parameters = parameters
        parameters.binCount = result.effectiveBinCount
        return parameters
    }

    // MARK: - Roundtrip

    func testEmbedDetectRoundTrip() {
        let (positions, key, result) = embeddedFixture()
        XCTAssertEqual(result.effectiveBinCount, 64)
        XCTAssertGreaterThan(result.embeddedBits, 48)

        let detection = GeometryWatermarker.detect(
            positions: positions, key: key, parameters: detectionParameters(for: result)
        )
        XCTAssertGreaterThan(detection.totalBits, 48)
        XCTAssertEqual(detection.matchedBits, detection.totalBits)
        XCTAssertLessThan(detection.pValue, 1e-9)
    }

    func testSmallCloudAutoReducesBins() {
        var positions = Fixtures.blobCloud(count: 900, seed: 5)
        let key = Fixtures.key(seed: 4)
        let result = GeometryWatermarker.embed(positions: &positions, key: key, parameters: parameters)
        XCTAssertEqual(result.effectiveBinCount, 16)

        let detection = GeometryWatermarker.detect(
            positions: positions, key: key, parameters: detectionParameters(for: result)
        )
        XCTAssertLessThan(detection.pValue, 1e-3)
    }

    func testTinyOrDegenerateCloudSkipsChannel() {
        var tiny = Fixtures.blobCloud(count: 100, seed: 9)
        let tinyResult = GeometryWatermarker.embed(positions: &tiny, key: Fixtures.key(seed: 1), parameters: parameters)
        XCTAssertEqual(tinyResult.effectiveBinCount, 0)
        XCTAssertFalse(tinyResult.isEmbedded)

        var degenerate = [SIMD3<Float>](repeating: SIMD3<Float>(1, 2, 3), count: 10_000)
        let degenerateResult = GeometryWatermarker.embed(
            positions: &degenerate, key: Fixtures.key(seed: 1), parameters: parameters
        )
        XCTAssertFalse(degenerateResult.isEmbedded)
        XCTAssertEqual(degenerate, [SIMD3<Float>](repeating: SIMD3<Float>(1, 2, 3), count: 10_000))
    }

    // MARK: - Robustness

    func testDetectionSurvivesGaussianNoise() {
        let (positions, key, result) = embeddedFixture()
        let sigma = Double(result.bboxDiagonal) * 0.001
        let noisy = Fixtures.addGaussianNoise(positions, sigma: sigma, seed: 77)

        let detection = GeometryWatermarker.detect(
            positions: noisy, key: key, parameters: detectionParameters(for: result)
        )
        XCTAssertLessThan(detection.pValue, 1e-6)
    }

    func testDetectionSurvivesSimilarityTransform() {
        let (positions, key, result) = embeddedFixture()
        let transformed = Fixtures.similarityTransform(
            positions,
            angle: 1.1,
            axis: SIMD3<Double>(0.3, 1.0, -0.5),
            scale: 2.37,
            translation: SIMD3<Double>(10, -4, 7.5)
        )

        let detection = GeometryWatermarker.detect(
            positions: transformed, key: key, parameters: detectionParameters(for: result)
        )
        XCTAssertLessThan(detection.pValue, 1e-6)
    }

    func testDetectionSurvivesVertexReordering() {
        let (positions, key, result) = embeddedFixture()
        let shuffled = Fixtures.shuffled(positions, seed: 123)

        let detection = GeometryWatermarker.detect(
            positions: shuffled, key: key, parameters: detectionParameters(for: result)
        )
        XCTAssertEqual(detection.matchedBits, detection.totalBits)
        XCTAssertLessThan(detection.pValue, 1e-9)
    }

    func testDetectionSurvivesFiftyPercentSubsample() {
        let (positions, key, result) = embeddedFixture()
        let subsampled = Fixtures.subsampled(positions, keepRatio: 0.5, seed: 321)
        XCTAssertLessThan(subsampled.count, positions.count * 6 / 10)

        let detection = GeometryWatermarker.detect(
            positions: subsampled, key: key, parameters: detectionParameters(for: result)
        )
        XCTAssertLessThan(detection.pValue, 1e-6)
    }

    // MARK: - False positives

    func testWrongKeysDetectAtChanceLevel() {
        let (positions, _, result) = embeddedFixture()
        let detectParameters = detectionParameters(for: result)

        var matchRatios = [Double]()
        for seed in 100..<200 {
            var keyBytes = Data(repeating: UInt8(seed % 256), count: WatermarkKey.byteCount)
            keyBytes[0] = UInt8(seed / 256)
            let wrongKey = try! WatermarkKey(data: keyBytes)
            let detection = GeometryWatermarker.detect(
                positions: positions, key: wrongKey, parameters: detectParameters
            )
            XCTAssertGreaterThan(
                detection.pValue, 1e-3,
                "wrong key seed \(seed) matched \(detection.matchedBits)/\(detection.totalBits)"
            )
            matchRatios.append(Double(detection.matchedBits) / Double(detection.totalBits))
        }

        let meanRatio = matchRatios.reduce(0, +) / Double(matchRatios.count)
        XCTAssertEqual(meanRatio, 0.5, accuracy: 0.05)
    }

    func testUnmarkedCloudDetectsAtChanceLevel() {
        let positions = Fixtures.blobCloud(count: 50_000, seed: 42)
        var detectParameters = parameters
        detectParameters.binCount = 64
        let detection = GeometryWatermarker.detect(
            positions: positions, key: Fixtures.key(seed: 11), parameters: detectParameters
        )
        XCTAssertGreaterThan(detection.pValue, 1e-3)
    }

    // MARK: - Imperceptibility

    func testDisplacementIsBoundedAndReportedAccurately() {
        let original = Fixtures.blobCloud(count: 50_000, seed: 42)
        var positions = original
        let result = GeometryWatermarker.embed(
            positions: &positions, key: Fixtures.key(seed: 11), parameters: parameters
        )

        var actualMax: Float = 0
        for (before, after) in zip(original, positions) {
            actualMax = max(actualMax, simd_length(after - before))
        }
        XCTAssertLessThanOrEqual(actualMax, result.maxDisplacement + 1e-5)
        XCTAssertLessThanOrEqual(result.maxDisplacement, result.bboxDiagonal * 0.005)
    }
}
