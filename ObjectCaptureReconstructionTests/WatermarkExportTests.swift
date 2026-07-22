import XCTest
@testable import Object_Capture_Reconstruction
import ModelFileIO
import ModelIO
import WatermarkCore
import simd

/// End-to-end checks that the export pipeline embeds a detectable, keyed
/// provenance watermark in GLB and splat-PLY outputs, and that wrong keys
/// detect at chance level.
@MainActor
final class WatermarkExportTests: XCTestCase {

    /// Fixed keys keep the test fully deterministic: which bits land in which
    /// bins (and therefore the exact detection statistic) depends on the key.
    private func fixedStamp(byte: UInt8 = 0x2A) throws -> WatermarkStamp {
        WatermarkStamp(
            key: try WatermarkKey(data: Data(repeating: byte, count: WatermarkKey.byteCount)),
            geometryParameters: GeometryWatermarkParameters(),
            textureParameters: TextureWatermarkParameters()
        )
    }

    private func wrongKey() throws -> WatermarkKey {
        try WatermarkKey(data: Data(repeating: 0x2B, count: WatermarkKey.byteCount))
    }

    func testGLBExportEmbedsDetectableGeometryWatermark() throws {
        let directory = try makeTemporaryDirectory()
        let sourceURL = directory.appending(path: "box.usdc")
        let glbURL = directory.appending(path: "box.glb")
        try writeSampleUSD(to: sourceURL)

        let stamp = try fixedStamp()
        let outcome = try USDZToGLTFConverter.convert(
            usdzURL: sourceURL,
            format: .glb,
            outputURL: glbURL,
            watermark: stamp
        )

        let geometry = try XCTUnwrap(outcome.geometry, "fixture should have enough vertices to embed")
        XCTAssertGreaterThan(geometry.embeddedBits, 0)

        let asset = try MiniGLTFReader.read(url: glbURL)
        XCTAssertGreaterThanOrEqual(asset.positions.count, 2048)

        let detection = GeometryWatermarker.detect(
            positions: asset.positions,
            key: stamp.key,
            parameters: geometry.detectionParameters
        )
        XCTAssertEqual(detection.matchedBits, detection.totalBits)
        XCTAssertLessThan(detection.pValue, 1e-6)

        let wrongDetection = GeometryWatermarker.detect(
            positions: asset.positions,
            key: try wrongKey(),
            parameters: geometry.detectionParameters
        )
        XCTAssertGreaterThan(wrongDetection.pValue, 1e-3)
    }

    func testSplatExportEmbedsDetectableGeometryWatermark() throws {
        let directory = try makeTemporaryDirectory()
        let sourceURL = directory.appending(path: "box.usdc")
        let plyURL = directory.appending(path: "box.ply")
        try writeSampleUSD(to: sourceURL)

        let stamp = try fixedStamp()
        let outcome = try SplatSampleGenerator.generate(
            usdzURL: sourceURL,
            outputURL: plyURL,
            targetCount: 30_000,
            watermark: stamp
        )

        let geometry = try XCTUnwrap(outcome.geometry)
        let positions = try PLYReader.readPositions(url: plyURL)
        XCTAssertGreaterThanOrEqual(positions.count, 30_000)

        let detection = GeometryWatermarker.detect(
            positions: positions,
            key: stamp.key,
            parameters: geometry.detectionParameters
        )
        XCTAssertLessThan(detection.pValue, 1e-6)

        let wrongDetection = GeometryWatermarker.detect(
            positions: positions,
            key: try wrongKey(),
            parameters: geometry.detectionParameters
        )
        XCTAssertGreaterThan(wrongDetection.pValue, 1e-3)
    }

    // MARK: - Key reuse

    func testSharedKeyMarksEveryFileWithTheSameKey() {
        let shared = SharedWatermarkKey(key: .random(), label: "Client A")

        let first = WatermarkStamp.next(sharedKey: shared)
        let second = WatermarkStamp.next(sharedKey: shared)
        XCTAssertEqual(first.key, second.key)
        XCTAssertEqual(first.key, shared.key)
        XCTAssertEqual(first.keyLabel, "Client A")

        // The default stays per-copy: every file gets unique, unlabeled keys.
        let fresh = WatermarkStamp.next(sharedKey: nil)
        let anotherFresh = WatermarkStamp.next(sharedKey: nil)
        XCTAssertNotEqual(fresh.key, anotherFresh.key)
        XCTAssertNil(fresh.keyLabel)
    }

    func testOneSharedKeyVerifiesAcrossDifferentExportedFormats() throws {
        let directory = try makeTemporaryDirectory()
        let sourceURL = directory.appending(path: "box.usdc")
        let glbURL = directory.appending(path: "box.glb")
        let plyURL = directory.appending(path: "box.ply")
        try writeSampleUSD(to: sourceURL)

        // The same saved key stamps two different exports of the same model.
        let shared = SharedWatermarkKey(
            key: try WatermarkKey(data: Data(repeating: 0x3A, count: WatermarkKey.byteCount)),
            label: "Client A"
        )
        let glbOutcome = try USDZToGLTFConverter.convert(
            usdzURL: sourceURL, format: .glb, outputURL: glbURL,
            watermark: .next(sharedKey: shared)
        )
        let plyOutcome = try SplatSampleGenerator.generate(
            usdzURL: sourceURL, outputURL: plyURL, targetCount: 30_000,
            watermark: .next(sharedKey: shared)
        )

        // Both files verify under that one key.
        let glbDetection = GeometryWatermarker.detect(
            positions: try MiniGLTFReader.read(url: glbURL).positions,
            key: shared.key,
            parameters: try XCTUnwrap(glbOutcome.geometry).detectionParameters
        )
        XCTAssertLessThan(glbDetection.pValue, 1e-6)

        let plyDetection = GeometryWatermarker.detect(
            positions: try PLYReader.readPositions(url: plyURL),
            key: shared.key,
            parameters: try XCTUnwrap(plyOutcome.geometry).detectionParameters
        )
        XCTAssertLessThan(plyDetection.pValue, 1e-6)
    }

    func testUnwatermarkedConvertLeavesNoOutcome() throws {
        let directory = try makeTemporaryDirectory()
        let sourceURL = directory.appending(path: "box.usdc")
        let glbURL = directory.appending(path: "box.glb")
        try writeSampleUSD(to: sourceURL)

        let outcome = try USDZToGLTFConverter.convert(
            usdzURL: sourceURL,
            format: .glb,
            outputURL: glbURL
        )
        XCTAssertNil(outcome.geometry)
        XCTAssertNil(outcome.texture)
        XCTAssertTrue(outcome.channels.isEmpty)
    }

    // MARK: - Fixtures

    /// Densely segmented box: broad radial-norm distribution (unlike a sphere)
    /// and enough vertices for the binned geometry watermark.
    private func writeSampleUSD(to url: URL) throws {
        let allocator = MDLMeshBufferDataAllocator()
        let mesh = MDLMesh(
            boxWithExtent: SIMD3<Float>(0.2, 0.2, 0.2),
            segments: SIMD3<UInt32>(28, 28, 28),
            inwardNormals: false,
            geometryType: .triangles,
            allocator: allocator
        )

        let asset = MDLAsset()
        asset.add(mesh)
        try asset.export(to: url)
    }

    private func makeTemporaryDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appending(
            path: UUID().uuidString,
            directoryHint: .isDirectory
        )
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        return directory
    }
}
