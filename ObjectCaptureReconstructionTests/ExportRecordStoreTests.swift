import XCTest
@testable import Object_Capture_Reconstruction
import WatermarkCore

@MainActor
final class ExportRecordStoreTests: XCTestCase {

    private func makeRecord(
        jobId: UUID,
        format: String = "glb",
        detailLevel: String = "medium",
        createdAt: Date = Date(timeIntervalSince1970: 1_752_000_000),
        sha: String = String(repeating: "ab", count: 32)
    ) -> WatermarkRecord {
        WatermarkRecord(
            jobId: jobId,
            format: format,
            detailLevel: detailLevel,
            filename: "model-\(detailLevel).\(format)",
            filePath: "/tmp/model-\(detailLevel).\(format)",
            createdAt: createdAt,
            key: .random(),
            channels: [WatermarkRecord.Channel.geometry],
            geometry: .init(parameters: GeometryWatermarkParameters(), effectiveBinCount: 64, embeddedBits: 60),
            texture: nil,
            fileSHA256: sha
        )
    }

    func testRecordsRoundTripThroughStore() throws {
        let store = try JobStore(inMemory: true)
        let jobId = UUID()
        let record = makeRecord(jobId: jobId)

        store.saveExportRecords([record])
        let loaded = store.exportRecords(jobId: jobId)

        XCTAssertEqual(loaded.count, 1)
        XCTAssertEqual(loaded.first?.recordId, record.recordId)
        XCTAssertEqual(loaded.first?.key, record.key)
        XCTAssertEqual(loaded.first?.channels, record.channels)
        XCTAssertEqual(loaded.first?.geometry, record.geometry)
        XCTAssertEqual(loaded.first?.fileSHA256, record.fileSHA256)
        XCTAssertEqual(loaded.first?.format, "glb")
    }

    func testLatestRecordPicksNewestForFileSlot() throws {
        let store = try JobStore(inMemory: true)
        let jobId = UUID()
        let older = makeRecord(jobId: jobId, createdAt: Date(timeIntervalSince1970: 1_000), sha: "old")
        let newer = makeRecord(jobId: jobId, createdAt: Date(timeIntervalSince1970: 2_000), sha: "new")
        let otherLevel = makeRecord(jobId: jobId, detailLevel: "full", createdAt: Date(timeIntervalSince1970: 3_000))

        store.saveExportRecords([older, newer, otherLevel])

        let latest = store.latestExportRecord(jobId: jobId, detailLevel: "medium", format: "glb")
        XCTAssertEqual(latest?.recordId, newer.recordId)
        XCTAssertNil(store.latestExportRecord(jobId: jobId, detailLevel: "medium", format: "ply"))
    }

    func testRecordsSurviveJobDeletion() throws {
        let store = try JobStore(inMemory: true)
        let job = ReconstructionJob(
            imageFolder: URL(fileURLWithPath: "/tmp/images"),
            modelFolder: URL(fileURLWithPath: "/tmp/models"),
            modelName: "Chair"
        )
        store.saveJob(job)
        store.saveExportRecords([makeRecord(jobId: job.id)])

        store.deleteJob(id: job.id)

        XCTAssertTrue(store.loadJobs().isEmpty)
        XCTAssertEqual(store.exportRecords(jobId: job.id).count, 1,
                       "provenance records must outlive the job")
    }

    func testWatermarkFlagPersistsThroughStore() throws {
        let store = try JobStore(inMemory: true)
        var job = ReconstructionJob(
            imageFolder: URL(fileURLWithPath: "/tmp/images"),
            modelFolder: URL(fileURLWithPath: "/tmp/models"),
            modelName: "Lamp"
        )
        job.watermarkEnabled = true
        store.saveJob(job)

        let reloaded = store.loadJobs().first
        XCTAssertEqual(reloaded?.isWatermarkEnabled, true)
    }
}
