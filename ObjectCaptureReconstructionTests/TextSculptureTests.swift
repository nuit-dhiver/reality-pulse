import XCTest
@testable import Object_Capture_Reconstruction
import ModelIO
import simd

@MainActor
final class TextSculptureTests: XCTestCase {

    // MARK: - Export format metadata

    func testTextSculptureFilenameDoesNotCollideWithGLB() {
        let job = ReconstructionJob(
            imageFolder: URL(fileURLWithPath: "/tmp/images"),
            modelFolder: URL(fileURLWithPath: "/tmp/models"),
            modelName: "Vase",
            primaryDetailLevel: .medium
        )
        XCTAssertEqual(job.exportFilename(for: .medium, format: .textSculpture), "Vase-medium-text.glb")
        XCTAssertEqual(job.exportFilename(for: .medium, format: .glb), "Vase-medium.glb")
    }

    func testOptionsPersistThroughJobStore() throws {
        let container = try JobStore.makeModelContainer(inMemory: true)
        var options = TextSculptureOptions()
        options.layout = .cloud
        options.letterSize = .large
        options.text = "سلام دنیا"
        options.useInkColor = true
        options.inkColor = SIMD3<Float>(0.8, 0.1, 0.2)

        let job = ReconstructionJob(
            imageFolder: URL(fileURLWithPath: "/tmp/images"),
            modelFolder: URL(fileURLWithPath: "/tmp/models"),
            modelName: "Teapot",
            exportFormats: [.textSculpture],
            textSculptureOptions: options
        )

        JobStore(modelContainer: container).saveJob(job)
        let reloaded = JobStore(modelContainer: container).loadJobs().first

        XCTAssertEqual(reloaded?.exportFormats, [.textSculpture])
        XCTAssertEqual(reloaded?.textSculptureOptions, options)
    }

    func testOptionsDecodeLenientlyFromPartialData() throws {
        let json = Data(#"{"layout":"cloud","letterSize":"huge"}"#.utf8)
        let options = try JSONDecoder().decode(TextSculptureOptions.self, from: json)

        XCTAssertEqual(options.layout, .cloud)
        XCTAssertEqual(options.letterSize, TextSculptureOptions().letterSize)
        XCTAssertEqual(options.text, "")
        XCTAssertFalse(options.useInkColor)
    }

    // MARK: - Atlas

    func testAtlasUVsPointAtUprightUnmirroredGlyph() throws {
        // "▛" fills every quadrant but the bottom-right, so a flip or mirror shows.
        let shaper = GlyphShaper(text: "▛", fallbackText: "x")
        let glyph = try XCTUnwrap(shaper.glyphStream().first)
        let atlas = try XCTUnwrap(GlyphAtlas(
            glyphs: [glyph.key: glyph.bounds],
            fonts: shaper.fonts,
            lineHeight: shaper.lineHeight
        ))
        let entry = try XCTUnwrap(atlas.entries[glyph.key])

        let image = atlas.image
        let width = image.width
        let height = image.height
        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        pixels.withUnsafeMutableBytes { buffer in
            let context = CGContext(
                data: buffer.baseAddress, width: width, height: height, bitsPerComponent: 8,
                bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            )
            context?.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        }
        // Alpha at a texture coordinate (origin top-left, like glTF).
        func alpha(u: Float, v: Float) -> UInt8 {
            let x = min(width - 1, Int(u * Float(width)))
            let y = min(height - 1, Int(v * Float(height)))
            return pixels[(y * width + x) * 4 + 3]
        }

        func probe(_ x: Float, _ y: Float) -> UInt8 {
            alpha(u: entry.uvMin.x + (entry.uvMax.x - entry.uvMin.x) * x,
                  v: entry.uvMin.y + (entry.uvMax.y - entry.uvMin.y) * y)
        }
        XCTAssertGreaterThan(probe(0.25, 0.25), 128, "top-left")
        XCTAssertGreaterThan(probe(0.75, 0.25), 128, "top-right")
        XCTAssertGreaterThan(probe(0.25, 0.75), 128, "bottom-left")
        XCTAssertLessThan(probe(0.75, 0.75), 16, "bottom-right")
    }

    // MARK: - Slicing

    func testSlicingBoxWeldsEachRingIntoOneClosedSquare() throws {
        let meshes = try loadSampleBox()
        let result = ContourSlicer.slice(meshes: meshes, targetSpacing: 0.05)

        XCTAssertEqual(result.spacing, 0.05, accuracy: 1e-5)
        for ring in 0..<4 {
            let loops = result.polylines.filter { $0.ringIndex == ring }
            XCTAssertEqual(loops.count, 1, "ring \(ring)")
            guard let loop = loops.first else { continue }
            XCTAssertTrue(loop.isClosed)

            let height = -0.1 + (Float(ring) + 0.5) * 0.05
            var perimeter: Float = 0
            for (index, vertex) in loop.vertices.enumerated() {
                XCTAssertEqual(vertex.position.y, height, accuracy: 1e-5)
                let next = loop.vertices[(index + 1) % loop.vertices.count]
                perimeter += simd_distance(vertex.position, next.position)
            }
            XCTAssertEqual(perimeter, 0.8, accuracy: 1e-4)
        }
    }

    // MARK: - Shaping

    func testShapedLinesFitRequestedWidth() {
        let shaper = GlyphShaper(text: "The sea remembers every name it has taken", fallbackText: "x")
        var cursor = 0
        for width in [200, 500, 1_000] as [CGFloat] {
            guard let line = shaper.line(startingAt: cursor, maxWidth: width) else {
                XCTFail("No line for width \(width)")
                continue
            }
            XCTAssertGreaterThan(line.characterCount, 0)
            XCTAssertFalse(line.glyphs.isEmpty)
            for glyph in line.glyphs {
                XCTAssertLessThanOrEqual(glyph.position.x, width)
            }
            cursor += line.characterCount
        }
    }

    func testShaperWrapsAroundCyclicText() {
        let shaper = GlyphShaper(text: "one two", fallbackText: "x")
        let length = shaper.text.utf16.count

        let first = shaper.line(startingAt: 0, maxWidth: 10_000)
        let wrapped = shaper.line(startingAt: length, maxWidth: 10_000)
        XCTAssertEqual(first?.characterCount, length)
        XCTAssertEqual(first?.glyphs.map(\.key), wrapped?.glyphs.map(\.key))
    }

    func testAnyScriptProducesGlyphs() {
        for text in ["سلام دنیا", "Hello world", "你好世界", "Привет мир"] {
            let shaper = GlyphShaper(text: text, fallbackText: "x")
            XCTAssertFalse(shaper.glyphStream().isEmpty, text)
        }
        XCTAssertTrue(GlyphShaper(text: "سلام دنیا", fallbackText: "x").isRightToLeft)
        XCTAssertFalse(GlyphShaper(text: "Hello", fallbackText: "x").isRightToLeft)
    }

    func testBlankTextFallsBackToModelName() {
        let shaper = GlyphShaper(text: "  \n\t ", fallbackText: "Teapot")
        XCTAssertEqual(shaper.text, "Teapot" + GlyphShaper.separator)
    }

    // MARK: - End-to-end generation

    func testGeneratesValidGLBForBothLayouts() throws {
        let directory = try makeTemporaryDirectory()
        let usdURL = directory.appending(path: "box.usdc")
        try writeSampleBox(to: usdURL)

        for layout in TextSculptureOptions.Layout.allCases {
            var options = TextSculptureOptions()
            options.layout = layout
            options.text = "Reality Pulse writes the world"

            let glbURL = directory.appending(path: "box-\(layout.rawValue).glb")
            try TextSculptureGenerator.generate(usdzURL: usdURL, outputURL: glbURL, options: options, fallbackText: "Box")
            let glb = try parseGLB(Data(contentsOf: glbURL))

            let attributes = try XCTUnwrap(glb.document.meshes?.first?.primitives.first?.attributes)
            XCTAssertEqual(Set(attributes.keys), ["POSITION", "NORMAL", "TEXCOORD_0", "COLOR_0"])
            XCTAssertEqual(glb.document.materials?.first?.alphaMode, "MASK")
            XCTAssertEqual(glb.document.images?.count, 1)

            // Letters hug the 0.2 m cube (±0.1), give or take their own size.
            let positions = try glb.vec3(accessor: attributes["POSITION"]!)
            XCTAssertGreaterThan(positions.count, 0)
            XCTAssertEqual(positions.count % 4, 0)
            for position in positions {
                XCTAssertLessThanOrEqual(simd_reduce_max(simd_abs(position)), 0.11, "\(layout)")
            }
        }
    }

    func testRingLettersStandUprightAndFaceOutward() throws {
        let directory = try makeTemporaryDirectory()
        let usdURL = directory.appending(path: "box.usdc")
        let glbURL = directory.appending(path: "rings.glb")
        try writeSampleBox(to: usdURL)

        var options = TextSculptureOptions()
        options.letterSize = .large
        try TextSculptureGenerator.generate(usdzURL: usdURL, outputURL: glbURL, options: options, fallbackText: "Box")

        let glb = try parseGLB(Data(contentsOf: glbURL))
        let attributes = try XCTUnwrap(glb.document.meshes?.first?.primitives.first?.attributes)
        let positions = try glb.vec3(accessor: attributes["POSITION"]!)

        // Quads are BL, BR, TR, TL: "up" must point up and the front face outward.
        for quad in stride(from: 0, to: positions.count, by: 4) {
            let bottomLeft = positions[quad]
            let bottomRight = positions[quad + 1]
            let topLeft = positions[quad + 3]
            let up = topLeft - bottomLeft
            XCTAssertGreaterThan(up.y, 0.9 * simd_length(up))

            let front = simd_cross(bottomRight - bottomLeft, up)
            let outward = SIMD3<Float>(bottomLeft.x, 0, bottomLeft.z)
            XCTAssertGreaterThan(simd_dot(front, outward), 0)
        }
    }

    func testInkModeColorsEveryLetterTheSame() throws {
        let directory = try makeTemporaryDirectory()
        let usdURL = directory.appending(path: "box.usdc")
        let glbURL = directory.appending(path: "ink.glb")
        try writeSampleBox(to: usdURL)

        var options = TextSculptureOptions()
        options.layout = .cloud
        options.letterSize = .large
        options.useInkColor = true
        options.inkColor = SIMD3<Float>(1, 0, 0)
        try TextSculptureGenerator.generate(usdzURL: usdURL, outputURL: glbURL, options: options, fallbackText: "Box")

        let glb = try parseGLB(Data(contentsOf: glbURL))
        let attributes = try XCTUnwrap(glb.document.meshes?.first?.primitives.first?.attributes)
        let colors = try glb.vec3(accessor: attributes["COLOR_0"]!)
        XCTAssertFalse(colors.isEmpty)
        for color in colors {
            XCTAssertEqual(color, SIMD3<Float>(1, 0, 0))
        }
    }

    func testModelExportServicePassesJobOptionsToGenerator() throws {
        let directory = try makeTemporaryDirectory()
        let usdURL = directory.appending(path: "box.usdc")
        let glbURL = directory.appending(path: "Box-text.glb")
        try writeSampleBox(to: usdURL)

        var options = TextSculptureOptions()
        options.layout = .cloud
        options.letterSize = .large
        options.useInkColor = true
        options.inkColor = SIMD3<Float>(0, 0, 1)
        try ModelExportService.export(
            usdzURL: usdURL,
            format: .textSculpture,
            outputURL: glbURL,
            textSculptureOptions: options,
            fallbackText: "Box"
        )

        let glb = try parseGLB(Data(contentsOf: glbURL))
        let attributes = try XCTUnwrap(glb.document.meshes?.first?.primitives.first?.attributes)
        let colors = try glb.vec3(accessor: attributes["COLOR_0"]!)
        XCTAssertFalse(colors.isEmpty)
        for color in colors {
            XCTAssertEqual(color, SIMD3<Float>(0, 0, 1))
        }
    }

    func testOptionsSurviveJSONRoundTrip() throws {
        var options = TextSculptureOptions()
        options.text = "hello"
        options.layout = .cloud
        let job = ReconstructionJob(
            imageFolder: URL(fileURLWithPath: "/tmp/images"),
            modelFolder: URL(fileURLWithPath: "/tmp/models"),
            modelName: "Vase",
            exportFormats: [.textSculpture],
            textSculptureOptions: options
        )

        let decoded = try JSONDecoder().decode(ReconstructionJob.self, from: JSONEncoder().encode(job))
        XCTAssertEqual(decoded.textSculptureOptions, options)
    }

    func testGenerationIsDeterministic() throws {
        let directory = try makeTemporaryDirectory()
        let usdURL = directory.appending(path: "box.usdc")
        try writeSampleBox(to: usdURL)

        for layout in TextSculptureOptions.Layout.allCases {
            var options = TextSculptureOptions()
            options.layout = layout
            options.letterSize = .large
            let firstURL = directory.appending(path: "first-\(layout.rawValue).glb")
            let secondURL = directory.appending(path: "second-\(layout.rawValue).glb")
            try TextSculptureGenerator.generate(usdzURL: usdURL, outputURL: firstURL, options: options, fallbackText: "Box")
            try TextSculptureGenerator.generate(usdzURL: usdURL, outputURL: secondURL, options: options, fallbackText: "Box")
            XCTAssertEqual(try Data(contentsOf: firstURL), try Data(contentsOf: secondURL), "\(layout)")
        }
    }

    // MARK: - Helpers

    private struct ParsedGLB {
        var document: GLTFDocument
        var binary: Data

        func vec3(accessor index: Int) throws -> [SIMD3<Float>] {
            let accessor = try XCTUnwrap(document.accessors?[index])
            XCTAssertEqual(accessor.type, "VEC3")
            let view = try XCTUnwrap(document.bufferViews?[try XCTUnwrap(accessor.bufferView)])
            let start = (view.byteOffset ?? 0) + (accessor.byteOffset ?? 0)
            let bytes = binary.subdata(in: start..<(start + accessor.count * 12))
            return bytes.withUnsafeBytes { raw in
                let floats = raw.bindMemory(to: Float32.self)
                return (0..<accessor.count).map { SIMD3<Float>(floats[$0 * 3], floats[$0 * 3 + 1], floats[$0 * 3 + 2]) }
            }
        }
    }

    private func parseGLB(_ data: Data) throws -> ParsedGLB {
        func uint32(at offset: Int) -> UInt32 {
            data.subdata(in: offset..<(offset + 4)).withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) }
        }
        XCTAssertEqual(data.prefix(4), Data("glTF".utf8))
        XCTAssertEqual(uint32(at: 4), 2)
        XCTAssertEqual(Int(uint32(at: 8)), data.count)

        let jsonLength = Int(uint32(at: 12))
        let json = data.subdata(in: 20..<(20 + jsonLength))
        let binaryHeader = 20 + jsonLength
        let binaryLength = Int(uint32(at: binaryHeader))
        let binary = data.subdata(in: (binaryHeader + 8)..<(binaryHeader + 8 + binaryLength))

        return ParsedGLB(document: try JSONDecoder().decode(GLTFDocument.self, from: json), binary: binary)
    }

    private func loadSampleBox() throws -> [MeshGeometryReader.Mesh] {
        let url = try makeTemporaryDirectory().appending(path: "box.usdc")
        try writeSampleBox(to: url)
        return MeshGeometryReader.loadMeshes(from: url)
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
