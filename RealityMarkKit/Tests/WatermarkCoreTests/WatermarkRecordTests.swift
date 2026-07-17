import XCTest
@testable import WatermarkCore

final class WatermarkRecordTests: XCTestCase {
    private func makeRecord() -> WatermarkRecord {
        WatermarkRecord(
            jobId: UUID(),
            format: "glb",
            detailLevel: "full",
            filename: "chair-full.glb",
            filePath: "/tmp/chair-full.glb",
            createdAt: Date(timeIntervalSince1970: 1_752_000_000),
            key: Fixtures.key(seed: 42),
            channels: [WatermarkRecord.Channel.geometry, WatermarkRecord.Channel.texture],
            geometry: .init(
                parameters: GeometryWatermarkParameters(),
                effectiveBinCount: 32,
                embeddedBits: 30
            ),
            texture: .init(
                parameters: TextureWatermarkParameters(),
                images: [.init(name: "baseColor.png", semantic: "baseColor", width: 2048, height: 2048)]
            ),
            fileSHA256: String(repeating: "ab", count: 32)
        )
    }

    func testJSONRoundTrip() throws {
        let record = makeRecord()
        let data = try record.jsonData()
        let decoded = try WatermarkRecord(jsonData: data)
        XCTAssertEqual(decoded, record)
    }

    func testDetectionParametersUseEffectiveBinCount() {
        let record = makeRecord()
        XCTAssertEqual(record.geometry?.parameters.binCount, 64)
        XCTAssertEqual(record.geometry?.detectionParameters.binCount, 32)
    }

    func testKeyRoundTripsThroughRecord() throws {
        let record = makeRecord()
        XCTAssertEqual(try record.watermarkKey, Fixtures.key(seed: 42))
    }

    func testVersionsAreStamped() {
        let record = makeRecord()
        XCTAssertEqual(record.schemaVersion, WatermarkRecord.currentSchemaVersion)
        XCTAssertEqual(record.algorithmVersion, WatermarkRecord.currentAlgorithmVersion)
    }
}
