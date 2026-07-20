import XCTest
@testable import Object_Capture_Reconstruction
import CoreGraphics
import ModelFileIO
import WatermarkCore

@MainActor
final class USDZStamperTests: XCTestCase {

    /// Synthetic photo-like texture PNG.
    private func makeTexturePNG(width: Int, height: Int) throws -> Data {
        var pixels = [UInt8](repeating: 255, count: width * height * 4)
        for y in 0..<height {
            for x in 0..<width {
                let offset = (y * width + x) * 4
                let value = 90.0 + 80.0 * Double(x) / Double(width)
                    + 40.0 * sin(Double(x) * 0.11) * cos(Double(y) * 0.07)
                pixels[offset] = UInt8(min(255, max(0, value)))
                pixels[offset + 1] = UInt8(min(255, max(0, value * 0.8)))
                pixels[offset + 2] = UInt8(min(255, max(0, value * 0.6)))
            }
        }
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
        let image = pixels.withUnsafeMutableBytes { buffer in
            CGContext(
                data: buffer.baseAddress,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: width * 4,
                space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            )!.makeImage()!
        }
        return try ImageCodec.pngData(from: image)
    }

    /// usdz-shaped fixture: a fake `.usdc` payload plus base-color and normal
    /// PNG entries. ModelIO can't parse the fake usdc, so the stamper falls
    /// back to the filename heuristic — which is the fragile path worth testing.
    private func makeFixtureUSDZ(in directory: URL) throws -> (url: URL, usdcBytes: Data, normalBytes: Data) {
        let usdcBytes = Data((0..<50_000).map { UInt8(truncatingIfNeeded: $0 &* 13) })
        let texturePNG = try makeTexturePNG(width: 256, height: 256)
        let normalPNG = try makeTexturePNG(width: 64, height: 64)

        let archive = UsdzArchive(entries: [
            .init(name: "model.usdc", data: usdcBytes),
            .init(name: "0/baked_mesh_tex0.png", data: texturePNG),
            .init(name: "0/baked_mesh_norm0.png", data: normalPNG),
        ])
        let url = directory.appending(path: "fixture.usdz")
        try archive.write(to: url)
        return (url, usdcBytes, normalPNG)
    }

    func testStampMarksBaseColorOnlyAndPreservesUSDC() throws {
        let directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let (usdzURL, usdcBytes, normalBytes) = try makeFixtureUSDZ(in: directory)

        let stamp = WatermarkStamp(
            key: try WatermarkKey(data: Data(repeating: 0x2A, count: WatermarkKey.byteCount)),
            geometryParameters: GeometryWatermarkParameters(),
            textureParameters: TextureWatermarkParameters()
        )
        let result = try USDZStamper.stampTextures(usdzURL: usdzURL, stamp: stamp)

        XCTAssertEqual(result.stampedImages.count, 1)
        XCTAssertEqual(result.stampedImages.first?.name, "0/baked_mesh_tex0.png")
        XCTAssertEqual(try WatermarkingService.sha256Hex(of: usdzURL), result.fileSHA256)

        let reread = try UsdzArchive.read(url: usdzURL)
        XCTAssertEqual(reread.entries[0].data, usdcBytes, ".usdc bytes must be untouched")
        XCTAssertEqual(reread.entries[2].data, normalBytes, "normal map must be untouched")

        let stampedImage = try ImageCodec.decode(reread.entries[1].data)
        let detection = TextureWatermarker.detect(
            image: stampedImage,
            key: stamp.key,
            parameters: stamp.textureParameters
        )
        XCTAssertLessThan(detection.pValue, 1e-6)

        let wrongDetection = TextureWatermarker.detect(
            image: stampedImage,
            key: try WatermarkKey(data: Data(repeating: 0x2B, count: WatermarkKey.byteCount)),
            parameters: stamp.textureParameters
        )
        XCTAssertGreaterThan(wrongDetection.pValue, 1e-3)
    }

    func testStageLeavesOriginalUntouchedUntilCommit() throws {
        let directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let (usdzURL, _, _) = try makeFixtureUSDZ(in: directory)
        let originalBytes = try Data(contentsOf: usdzURL)

        let stamp = WatermarkStamp(
            key: try WatermarkKey(data: Data(repeating: 0x2C, count: WatermarkKey.byteCount)),
            geometryParameters: GeometryWatermarkParameters(),
            textureParameters: TextureWatermarkParameters()
        )

        // Stage: original untouched, staged copy has the promised hash.
        let staged = try USDZStamper.stage(usdzURL: usdzURL, stamp: stamp)
        XCTAssertEqual(try Data(contentsOf: usdzURL), originalBytes)
        XCTAssertEqual(try WatermarkingService.sha256Hex(of: staged.temporaryURL), staged.fileSHA256)

        // Discard: original still untouched, temp gone.
        USDZStamper.discard(staged)
        XCTAssertEqual(try Data(contentsOf: usdzURL), originalBytes)
        XCTAssertFalse(FileManager.default.fileExists(atPath: staged.temporaryURL.path))

        // Stage + commit: destination now carries the staged bytes.
        let secondStaged = try USDZStamper.stage(usdzURL: usdzURL, stamp: stamp)
        try USDZStamper.commit(secondStaged, to: usdzURL)
        XCTAssertEqual(try WatermarkingService.sha256Hex(of: usdzURL), secondStaged.fileSHA256)
        XCTAssertNotEqual(try Data(contentsOf: usdzURL), originalBytes)
    }

    func testStampFailureLeavesOriginalUntouched() throws {
        let directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        // Archive with no image entries at all.
        let archive = UsdzArchive(entries: [
            .init(name: "model.usdc", data: Data(repeating: 0x55, count: 1024)),
        ])
        let url = directory.appending(path: "empty.usdz")
        try archive.write(to: url)
        let originalBytes = try Data(contentsOf: url)

        XCTAssertThrowsError(try USDZStamper.stampTextures(usdzURL: url, stamp: .fresh()))
        XCTAssertEqual(try Data(contentsOf: url), originalBytes)
    }
}
