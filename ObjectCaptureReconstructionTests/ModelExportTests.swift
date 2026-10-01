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

    func testConversionJobInputKindPersistsThroughJobStore() throws {
        let container = try JobStore.makeModelContainer(inMemory: true)
        let conversionJob = ReconstructionJob(
            inputKind: .usdzModel,
            imageFolder: URL(fileURLWithPath: "/tmp/source/Chair.usdz"),
            modelFolder: URL(fileURLWithPath: "/tmp/models"),
            modelName: "Chair",
            exportFormats: [.glb]
        )
        let imageJob = ReconstructionJob(
            imageFolder: URL(fileURLWithPath: "/tmp/images"),
            modelFolder: URL(fileURLWithPath: "/tmp/models"),
            modelName: "Teapot"
        )

        let store = JobStore(modelContainer: container)
        store.saveJobs([conversionJob, imageJob])
        let reloaded = JobStore(modelContainer: container).loadJobs()

        XCTAssertEqual(reloaded.map(\.inputKind), [.usdzModel, .images])
        XCTAssertEqual(reloaded.first?.imageFolder.lastPathComponent, "Chair.usdz")
    }

    func testMissingInputKindLoadsAsImages() throws {
        let job = ReconstructionJob(
            imageFolder: URL(fileURLWithPath: "/tmp/images"),
            modelFolder: URL(fileURLWithPath: "/tmp/models"),
            modelName: "Legacy"
        )
        let persistentJob = try PersistentJob(job: job, queueOrder: 0)
        persistentJob.inputKindRawValue = nil

        XCTAssertEqual(try persistentJob.toJob().inputKind, .images)
    }

    func testConversionURLHelpers() {
        let job = ReconstructionJob(
            inputKind: .usdzModel,
            imageFolder: URL(fileURLWithPath: "/tmp/source/Vase.usdz"),
            modelFolder: URL(fileURLWithPath: "/tmp/models"),
            modelName: "Vase",
            exportFormats: [.glb, .gaussianSplat]
        )

        XCTAssertTrue(job.isConversionJob)
        XCTAssertEqual(job.conversionFilename(for: .glb), "Vase.glb")
        XCTAssertEqual(
            job.conversionOutputURLs.map(\.path),
            ["/tmp/models/Vase.ply", "/tmp/models/Vase.glb"]
        )
    }

    func testConversionTextSculptureDoesNotCollideWithGLB() {
        let job = ReconstructionJob(
            inputKind: .usdzModel,
            imageFolder: URL(fileURLWithPath: "/tmp/source/Vase.usdz"),
            modelFolder: URL(fileURLWithPath: "/tmp/models"),
            modelName: "Vase",
            exportFormats: [.glb, .textSculpture]
        )

        XCTAssertEqual(job.conversionFilename(for: .textSculpture), "Vase-text.glb")
        XCTAssertEqual(
            job.conversionOutputURLs.map(\.path),
            ["/tmp/models/Vase.glb", "/tmp/models/Vase-text.glb"]
        )
    }

    func testJobDraftInUSDZModeBuildsConversionJob() {
        let draft = JobDraft()
        draft.inputMode = .usdz
        draft.sourceModelFile = URL(fileURLWithPath: "/tmp/source/Chair.usdz")
        draft.modelFolder = URL(fileURLWithPath: "/tmp/models")
        draft.modelName = "Chair"

        XCTAssertFalse(draft.validate(), "A conversion job needs at least one export format.")
        XCTAssertEqual(draft.alertMessage, "Choose at least one export format")

        draft.hasError = false
        draft.exportFormats = [.gltf]
        XCTAssertTrue(draft.validate())

        let job = draft.toJob()
        XCTAssertEqual(job?.inputKind, .usdzModel)
        XCTAssertEqual(job?.imageFolder.lastPathComponent, "Chair.usdz")

        let reopened = JobDraft(from: job!)
        XCTAssertEqual(reopened.inputMode, .usdz)
        XCTAssertEqual(reopened.sourceModelFile?.lastPathComponent, "Chair.usdz")
        XCTAssertNil(reopened.imageFolder)
    }

    func testModelExportServiceConvertsSingleFile() throws {
        let directory = try makeTemporaryDirectory()
        let sourceURL = directory.appending(path: "box.usdc")
        let glbURL = directory.appending(path: "out.glb")
        let gltfURL = directory.appending(path: "out.gltf")

        try writeSampleUSD(to: sourceURL)

        for (format, outputURL) in [(ModelExportFormat.glb, glbURL), (.gltf, gltfURL)] {
            try ModelExportService.export(
                usdzURL: sourceURL,
                format: format,
                outputURL: outputURL,
                textSculptureOptions: nil,
                fallbackText: "Box"
            )
        }

        XCTAssertEqual(try Data(contentsOf: glbURL).prefix(4), Data([0x67, 0x6C, 0x54, 0x46]))
        XCTAssertTrue(FileManager.default.fileExists(atPath: gltfURL.path))
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
