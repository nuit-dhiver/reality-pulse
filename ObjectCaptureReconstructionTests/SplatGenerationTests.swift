import XCTest
@testable import Object_Capture_Reconstruction
import ModelIO
import simd

@MainActor
final class SplatGenerationTests: XCTestCase {

    // MARK: - Export format metadata

    func testGaussianSplatExportFormatMetadata() {
        XCTAssertEqual(ModelExportFormat.gaussianSplat.fileExtension, "ply")
        XCTAssertTrue(ModelExportFormat.allCases.contains(.gaussianSplat))

        let job = ReconstructionJob(
            imageFolder: URL(fileURLWithPath: "/tmp/images"),
            modelFolder: URL(fileURLWithPath: "/tmp/models"),
            modelName: "Vase",
            primaryDetailLevel: .medium
        )
        XCTAssertEqual(job.exportFilename(for: .medium, format: .gaussianSplat), "Vase-medium.ply")
    }

    // MARK: - Splat encoding math

    func testColorRoundTripsThroughSphericalHarmonicDC() {
        let color = SIMD3<Float>(0.2, 0.6, 0.9)
        let splat = GaussianSplat.surfaceSplat(
            position: .zero,
            normal: SIMD3<Float>(0, 1, 0),
            color: color,
            tangentScale: 0.01
        )
        // Viewers recover color as: rgb = shC0 * f_dc + 0.5.
        let recovered = GaussianSplat.shC0 * splat.shDC + SIMD3<Float>(repeating: 0.5)
        XCTAssertEqual(recovered.x, color.x, accuracy: 1e-4)
        XCTAssertEqual(recovered.y, color.y, accuracy: 1e-4)
        XCTAssertEqual(recovered.z, color.z, accuracy: 1e-4)
    }

    func testOpacityStoredAsLogit() {
        let splat = GaussianSplat.surfaceSplat(
            position: .zero,
            normal: SIMD3<Float>(0, 0, 1),
            color: SIMD3<Float>(repeating: 0.5),
            tangentScale: 0.01,
            opacity: 0.9
        )
        let recoveredAlpha = 1 / (1 + exp(-splat.opacityLogit))
        XCTAssertEqual(recoveredAlpha, 0.9, accuracy: 1e-4)
    }

    func testRotationAlignsLocalZToSurfaceNormal() {
        let normal = simd_normalize(SIMD3<Float>(0.3, 0.7, -0.5))
        let q = GaussianSplat.quaternion(alignedTo: normal) // (w, x, y, z)
        // Reconstruct a simd quaternion (ix, iy, iz, r) and rotate +Z.
        let quat = simd_quatf(ix: q.y, iy: q.z, iz: q.w, r: q.x)
        let rotatedZ = quat.act(SIMD3<Float>(0, 0, 1))
        XCTAssertEqual(rotatedZ.x, normal.x, accuracy: 1e-3)
        XCTAssertEqual(rotatedZ.y, normal.y, accuracy: 1e-3)
        XCTAssertEqual(rotatedZ.z, normal.z, accuracy: 1e-3)
    }

    // MARK: - End-to-end generation

    func testGeneratesValidSplatPLYFromBox() throws {
        let directory = try makeTemporaryDirectory()
        let usdURL = directory.appending(path: "box.usdc")
        let plyURL = directory.appending(path: "box.ply")
        try writeSampleBox(to: usdURL)

        let requestedCount = 5_000
        try SplatSampleGenerator.generate(usdzURL: usdURL, outputURL: plyURL, targetCount: requestedCount)

        let data = try Data(contentsOf: plyURL)
        let parsed = try parsePLY(data)

        XCTAssertEqual(parsed.vertexCount, requestedCount)
        XCTAssertEqual(parsed.points.count, requestedCount)

        // Every sampled point must lie on the 0.2 m cube (±0.1) within epsilon.
        let halfExtent: Float = 0.1
        let epsilon: Float = 1e-3
        for point in parsed.points {
            XCTAssertLessThanOrEqual(abs(point.x), halfExtent + epsilon)
            XCTAssertLessThanOrEqual(abs(point.y), halfExtent + epsilon)
            XCTAssertLessThanOrEqual(abs(point.z), halfExtent + epsilon)
        }
    }

    func testSamplingIsDeterministic() throws {
        let directory = try makeTemporaryDirectory()
        let usdURL = directory.appending(path: "box.usdc")
        try writeSampleBox(to: usdURL)
        let meshes = MeshGeometryReader.loadMeshes(from: usdURL)

        let first = SurfaceSampler.sample(meshes: meshes, targetCount: 1_000)
        let second = SurfaceSampler.sample(meshes: meshes, targetCount: 1_000)

        XCTAssertEqual(first.points.count, second.points.count)
        XCTAssertEqual(first.points.first?.position, second.points.first?.position)
        XCTAssertEqual(first.points.last?.position, second.points.last?.position)
    }

    // MARK: - Helpers

    private struct ParsedPLY {
        var vertexCount: Int
        var points: [SIMD3<Float>]
    }

    private func parsePLY(_ data: Data) throws -> ParsedPLY {
        let marker = Data("end_header\n".utf8)
        guard let headerRange = data.range(of: marker) else {
            throw XCTSkip("PLY header marker not found")
        }
        let header = String(decoding: data[..<headerRange.upperBound], as: UTF8.self)
        var vertexCount = 0
        for line in header.split(separator: "\n") where line.hasPrefix("element vertex ") {
            vertexCount = Int(line.dropFirst("element vertex ".count)) ?? 0
        }

        let body = data.subdata(in: headerRange.upperBound..<data.endIndex)
        let floatsPerVertex = SplatPLYWriter.propertyNames.count
        let stride = floatsPerVertex * MemoryLayout<Float32>.size
        XCTAssertEqual(body.count, vertexCount * stride)

        var points = [SIMD3<Float>]()
        points.reserveCapacity(vertexCount)
        body.withUnsafeBytes { raw in
            let floats = raw.bindMemory(to: Float32.self)
            for i in 0..<vertexCount {
                let base = i * floatsPerVertex
                points.append(SIMD3<Float>(floats[base], floats[base + 1], floats[base + 2]))
            }
        }
        return ParsedPLY(vertexCount: vertexCount, points: points)
    }

    private func writeSampleBox(to url: URL) throws {
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
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }
}
