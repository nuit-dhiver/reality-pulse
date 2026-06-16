import XCTest
@testable import Object_Capture_Reconstruction
import ModelIO

@MainActor
final class ModelExportTests: XCTestCase {
    func testExportFormatsPersistThroughJobStore() throws {
        let container = try JobStore.makeModelContainer(inMemory: true)
        var job = ReconstructionJob(
            imageFolder: URL(fileURLWithPath: "/tmp/images"),
            modelFolder: URL(fileURLWithPath: "/tmp/models"),
            modelName: "Teapot",
            exportFormats: [.gltf, .glb]
        )

        JobStore(modelContainer: container).saveJob(job)
        let reloaded = JobStore(modelContainer: container).loadJobs().first

        XCTAssertEqual(reloaded?.exportFormats, [.gltf, .glb])
    }

    func testExportURLHelpers() {
        let job = ReconstructionJob(
            imageFolder: URL(fileURLWithPath: "/tmp/images"),
            modelFolder: URL(fileURLWithPath: "/tmp/models"),
            modelName: "Vase",
            primaryDetailLevel: .medium
        )

        XCTAssertEqual(
            job.exportFilename(for: .medium, format: .glb),
            "Vase-medium.glb"
        )
        XCTAssertEqual(
            job.exportURL(for: .medium, format: .gltf).lastPathComponent,
            "Vase-medium.gltf"
        )
    }

    func testUSDZToGLBProducesValidGLBHeader() throws {
        let directory = try makeTemporaryDirectory()
        let sourceURL = directory.appending(path: "box.usdc")
        let glbURL = directory.appending(path: "box.glb")

        try writeSampleUSD(to: sourceURL)

        try USDZToGLTFConverter.convert(
            usdzURL: sourceURL,
            format: .glb,
            outputURL: glbURL
        )

        let data = try Data(contentsOf: glbURL)
        XCTAssertGreaterThan(data.count, 20)
        XCTAssertEqual(data.prefix(4), Data([0x67, 0x6C, 0x54, 0x46])) // glTF

        let version = data.subdata(in: 4..<8).withUnsafeBytes {
            $0.load(as: UInt32.self)
        }
        XCTAssertEqual(version, 2)
    }

    func testUSDZToGLTFProducesJSONAndBinarySidecar() throws {
        let directory = try makeTemporaryDirectory()
        let sourceURL = directory.appending(path: "box.usdc")
        let gltfURL = directory.appending(path: "box.gltf")
        let binURL = directory.appending(path: "box.bin")

        try writeSampleUSD(to: sourceURL)

        try USDZToGLTFConverter.convert(
            usdzURL: sourceURL,
            format: .gltf,
            outputURL: gltfURL
        )

        XCTAssertTrue(FileManager.default.fileExists(atPath: gltfURL.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: binURL.path))

        let json = try Data(contentsOf: gltfURL)
        let document = try JSONDecoder().decode(GLTFDocument.self, from: json)
        XCTAssertEqual(document.asset.version, "2.0")
        XCTAssertFalse(document.meshes?.isEmpty ?? true)
    }

    private func writeSampleUSD(to url: URL) throws {
        let allocator = MDLMeshBufferDataAllocator()
        let mesh = MDLMesh(
            boxWithExtent: SIMD3<Float>(0.2, 0.2, 0.2),
            segments: SIMD3<UInt32>(1, 1, 1),
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
