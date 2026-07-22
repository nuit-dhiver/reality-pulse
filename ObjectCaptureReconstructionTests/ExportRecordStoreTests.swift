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

    // MARK: - Key library

    func testCreatedKeysAreListedAndResolvable() throws {
        let store = try JobStore(inMemory: true)

        let created = try XCTUnwrap(store.createWatermarkKey(label: "Client A"))
        XCTAssertEqual(created.label, "Client A")

        XCTAssertEqual(store.watermarkKeys().map(\.label), ["Client A"])

        let resolved = try XCTUnwrap(store.watermarkKey(id: created.id))
        XCTAssertEqual(resolved.label, "Client A")
        XCTAssertEqual(resolved.key.data.count, WatermarkKey.byteCount)

        // Resolving marks the key as used.
        XCTAssertNotNil(store.watermarkKeys().first?.lastUsedAt)
    }

    func testKeyLabelsAreUniqueAndTrimmedAndNonEmpty() throws {
        let store = try JobStore(inMemory: true)

        XCTAssertNotNil(store.createWatermarkKey(label: "Portfolio"))
        XCTAssertNil(store.createWatermarkKey(label: "Portfolio"), "duplicate labels must be rejected")
        XCTAssertNil(store.createWatermarkKey(label: "  Portfolio  "), "labels are compared trimmed")
        XCTAssertNil(store.createWatermarkKey(label: "   "), "blank labels must be rejected")
        XCTAssertEqual(store.watermarkKeys().count, 1)
    }

    func testDistinctKeysHaveDistinctMaterial() throws {
        let store = try JobStore(inMemory: true)
        let first = try XCTUnwrap(store.createWatermarkKey(label: "One"))
        let second = try XCTUnwrap(store.createWatermarkKey(label: "Two"))

        let firstKey = try XCTUnwrap(store.watermarkKey(id: first.id)).key
        let secondKey = try XCTUnwrap(store.watermarkKey(id: second.id)).key
        XCTAssertNotEqual(firstKey, secondKey)
    }

    func testMissingKeyResolvesToNil() throws {
        let store = try JobStore(inMemory: true)
        XCTAssertNil(store.watermarkKey(id: UUID()))
    }

    func testSelectedKeyPersistsWithJob() throws {
        let store = try JobStore(inMemory: true)
        let created = try XCTUnwrap(store.createWatermarkKey(label: "Client B"))

        var job = ReconstructionJob(
            imageFolder: URL(fileURLWithPath: "/tmp/images"),
            modelFolder: URL(fileURLWithPath: "/tmp/models"),
            modelName: "Statue"
        )
        job.watermarkEnabled = true
        job.watermarkKeyId = created.id
        store.saveJob(job)

        XCTAssertEqual(store.loadJobs().first?.watermarkKeyId, created.id)
    }

    func testKeyLabelRoundTripsThroughRecord() throws {
        let store = try JobStore(inMemory: true)
        let jobId = UUID()
        var record = makeRecord(jobId: jobId)
        record.keyLabel = "Client A"

        store.saveExportRecords([record])
        XCTAssertEqual(store.exportRecords(jobId: jobId).first?.keyLabel, "Client A")

        // Per-copy keys stay unlabeled.
        let otherJobId = UUID()
        store.saveExportRecords([makeRecord(jobId: otherJobId)])
        XCTAssertNil(store.exportRecords(jobId: otherJobId).first?.keyLabel)
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
