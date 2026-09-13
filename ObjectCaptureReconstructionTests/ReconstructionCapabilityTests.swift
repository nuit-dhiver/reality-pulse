import XCTest
@testable import Object_Capture_Reconstruction

/// Covers the platform differences between the macOS build and the iPhone and
/// iPad builds: which detail levels Object Capture offers, which requests a job
/// produces, and that stored job settings keep the same shape everywhere.
@MainActor
final class ReconstructionCapabilityTests: XCTestCase {

    func testSupportedDetailLevelsMatchThePlatform() {
        #if os(macOS)
        XCTAssertEqual(
            ReconstructionCapability.supportedDetailLevels,
            [.preview, .reduced, .medium, .full, .raw, .custom]
        )
        XCTAssertTrue(ReconstructionCapability.supportsMultipleDetailLevels)
        XCTAssertTrue(ReconstructionCapability.supportsCustomDetailSpecification)
        XCTAssertTrue(ReconstructionCapability.supportsMeshPrimitiveSelection)
        #else
        // Object Capture on iPhone and iPad only exposes reduced detail.
        XCTAssertEqual(ReconstructionCapability.supportedDetailLevels, [.reduced])
        XCTAssertFalse(ReconstructionCapability.supportsMultipleDetailLevels)
        XCTAssertFalse(ReconstructionCapability.supportsCustomDetailSpecification)
        XCTAssertFalse(ReconstructionCapability.supportsMeshPrimitiveSelection)
        #endif
    }

    func testDefaultDetailLevelIsSupported() {
        XCTAssertTrue(ReconstructionCapability.defaultDetailLevel.isSupportedOnThisPlatform)
        XCTAssertNotNil(ReconstructionCapability.defaultDetailLevel.frameworkDetail)
    }

    func testUnsupportedDetailLevelsAreReportedForTheJob() {
        var additionalDetailLevels = CodableDetailLevelOptions()
        additionalDetailLevels.isSelected = true
        additionalDetailLevels.medium = true

        let job = makeJob(
            primaryDetailLevel: .reduced,
            additionalDetailLevels: additionalDetailLevels
        )

        #if os(macOS)
        XCTAssertEqual(job.unsupportedDetailLevels, [])
        #else
        XCTAssertEqual(job.unsupportedDetailLevels, [.medium])
        #endif
    }

    func testRequestsAreOnlyCreatedForSupportedDetailLevels() {
        var additionalDetailLevels = CodableDetailLevelOptions()
        additionalDetailLevels.isSelected = true
        additionalDetailLevels.medium = true

        let job = makeJob(
            primaryDetailLevel: .reduced,
            additionalDetailLevels: additionalDetailLevels
        )

        let requests = job.createReconstructionRequests()

        #if os(macOS)
        XCTAssertEqual(requests.count, 2)
        #else
        XCTAssertEqual(requests.count, 1)
        #endif
    }

    func testUnsupportedDetailLevelMessageNamesTheLevels() {
        let message = ReconstructionCapability.unsupportedDetailLevelMessage(for: [.full, .medium])

        XCTAssertTrue(message.contains("Full"))
        XCTAssertTrue(message.contains("Medium"))
    }

    func testNewJobDraftStartsOnASupportedDetailLevel() {
        let draft = JobDraft()

        XCTAssertTrue(draft.detailLevelOptionUnderQualityMenu.isSupportedOnThisPlatform)
    }

    /// The stored settings keep every field on all platforms, even the ones
    /// only macOS applies, so a saved job decodes the same way everywhere.
    func testSessionConfigurationRoundTripsEveryFieldThroughJSON() throws {
        var configuration = CodableSessionConfiguration()
        configuration.meshPrimitive = .quad
        configuration.isObjectMaskingEnabled = false
        configuration.ignoreBoundingBox = true
        configuration.customDetailSpecification.maximumPolygonCount = 12_345
        configuration.customDetailSpecification.maximumTextureDimension = .eightK
        configuration.customDetailSpecification.textureFormat = .jpeg(compressionQuality: 0.5)

        let data = try JSONEncoder().encode(configuration)
        let decoded = try JSONDecoder().decode(CodableSessionConfiguration.self, from: data)

        XCTAssertEqual(decoded, configuration)
    }

    private func makeJob(
        primaryDetailLevel: CodableDetailLevel,
        additionalDetailLevels: CodableDetailLevelOptions
    ) -> ReconstructionJob {
        ReconstructionJob(
            imageFolder: URL(fileURLWithPath: "/tmp/capability-images"),
            modelFolder: URL(fileURLWithPath: "/tmp/capability-models"),
            modelName: "Capability",
            primaryDetailLevel: primaryDetailLevel,
            additionalDetailLevels: additionalDetailLevels
        )
    }
}
