/*
See the LICENSE file for licensing information.

Abstract:
Rebuilds a reconstructed USDZ out of words: shaped glyphs are laid on the
model's surface, either along horizontal contour rings (readable lines) or one
per sampled surface point (a cloud of letters), and written as a `.glb` of
flat, alpha-cut letter quads colored by the model's texture or a single ink.
*/

import Foundation
import CoreGraphics
import simd
import os

private let logger = Logger(subsystem: ObjectCaptureReconstructionApp.subsystem,
                            category: "TextSculptureGenerator")

enum TextSculptureGenerator {

    enum GeneratorError: LocalizedError {
        case noGeometry(URL)
        case noGlyphs(URL)
        case atlasFailed

        var errorDescription: String? {
            switch self {
            case .noGeometry(let url):
                return "No mesh geometry found in \(url.lastPathComponent)."
            case .noGlyphs(let url):
                return "No letters could be placed on \(url.lastPathComponent)."
            case .atlasFailed:
                return "Failed to render the letter texture for the text sculpture."
            }
        }
    }

    // Fixed layout/appearance defaults (the UI only picks the letter size). Tunable here.
    static let maximumGlyphCount = 400_000
    static let shapingFontSize: CGFloat = 64
    /// Line height (ascent + descent) as a fraction of the ring spacing; below 1
    /// leaves a sliver of air between rings.
    static let ringLineFill: Float = 0.9
    /// Letters on a closed ring are spread apart to close the loop, never by more than this.
    static let maximumRingStretch: CGFloat = 1.5
    /// Cloud line height as a multiple of the average point spacing `sqrt(area / N)`.
    static let cloudLetterScale: Float = 1.8
    /// Random offset of cloud letters along the normal, as a fraction of their
    /// height, so overlapping letters don't z-fight.
    static let cloudDepthJitter: Float = 0.1
    static let alphaCutoff: Float = 0.4

    /// Roughly how many letters the rings layout aims for. Letter size follows
    /// from the surface area, so wide, flat, and tall models all get about the
    /// same count instead of letter size tracking only the model's height.
    static func ringGlyphTarget(for size: TextSculptureOptions.LetterSize) -> Int {
        switch size {
        case .small: return 120_000
        case .medium: return 40_000
        case .large: return 12_000
        }
    }

    /// Average letter advance as a fraction of the ring spacing, used to turn
    /// a letter budget into a spacing: letters ≈ area / (spacing² × this).
    static let ringAdvanceFactor: Float = 0.4

    static func cloudGlyphCount(for size: TextSculptureOptions.LetterSize) -> Int {
        switch size {
        case .small: return 200_000
        case .medium: return 80_000
        case .large: return 25_000
        }
    }

    /// One letter placed in the world. A glyph-local point `(x, y)` in font
    /// units maps to `origin + right * x + up * y`.
    struct Quad {
        var key: GlyphKey
        var origin: SIMD3<Float>
        var right: SIMD3<Float>
        var up: SIMD3<Float>
        var normal: SIMD3<Float>
        /// Linear RGB.
        var color: SIMD3<Float>
    }

    /// Lay the text out on `usdzURL`'s surface and write the sculpture to `outputURL`.
    /// - Parameter fallbackText: used when `options.text` is blank (the model name).
    nonisolated static func generate(
        usdzURL: URL,
        outputURL: URL,
        options: TextSculptureOptions,
        fallbackText: String
    ) throws {
        let meshes = MeshGeometryReader.loadMeshes(from: usdzURL)
        guard !meshes.isEmpty else { throw GeneratorError.noGeometry(usdzURL) }

        let shaper = GlyphShaper(text: options.text, fallbackText: fallbackText, fontSize: shapingFontSize)
        let coloring = Coloring(meshes: meshes, options: options)
        var glyphBounds: [GlyphKey: CGRect] = [:]

        let quads: [Quad]
        switch options.layout {
        case .rings:
            quads = layOutRings(meshes: meshes, shaper: shaper, coloring: coloring,
                                letterSize: options.letterSize, glyphBounds: &glyphBounds)
        case .cloud:
            quads = layOutCloud(meshes: meshes, shaper: shaper, coloring: coloring,
                                letterSize: options.letterSize, glyphBounds: &glyphBounds)
        }
        guard !quads.isEmpty else { throw GeneratorError.noGlyphs(usdzURL) }

        guard let atlas = GlyphAtlas(glyphs: glyphBounds, fonts: shaper.fonts, lineHeight: shaper.lineHeight) else {
            throw GeneratorError.atlasFailed
        }

        try write(quads, atlas: atlas, to: outputURL)
        logger.log("Generated text sculpture \(outputURL.lastPathComponent, privacy: .public) with \(quads.count) letters (\(options.layout.rawValue, privacy: .public)) from \(usdzURL.lastPathComponent, privacy: .public)")
    }

    // MARK: - Rings layout

    private static func layOutRings(
        meshes: [MeshGeometryReader.Mesh],
        shaper: GlyphShaper,
        coloring: Coloring,
        letterSize: TextSculptureOptions.LetterSize,
        glyphBounds: inout [GlyphKey: CGRect]
    ) -> [Quad] {
        // A ring band of height `spacing` holds at most `area / spacing` of
        // contour length, so this spacing keeps the total near the target.
        let area = surfaceArea(of: meshes)
        let target = Float(ringGlyphTarget(for: letterSize))
        guard area > 0 else { return [] }
        let targetSpacing = sqrt(area / (target * ringAdvanceFactor))

        let slices = ContourSlicer.slice(meshes: meshes, targetSpacing: targetSpacing)
        guard slices.spacing > 0 else { return [] }

        // World units per font unit, and the baseline shift that centers the
        // line box (descent…ascent) on the slice plane.
        let scale = slices.spacing * ringLineFill / Float(shaper.lineHeight)
        let baselineOffset = -Float(shaper.ascent - shaper.descent) / 2
        // Skip slivers shorter than about three letters: they're usually scan noise.
        let minimumLength = Float(shaper.fontSize) * 1.5 * scale
        let minimumFragmentWidth = shaper.fontSize * 0.25

        var quads: [Quad] = []
        var cursor = 0

        // Top ring first, so the text reads downward like a page.
        for polyline in slices.polylines.reversed() {
            guard quads.count < maximumGlyphCount else { break }
            let path = RingPath(polyline)
            guard path.length >= minimumLength else { continue }

            // Fill the ring with consecutive fragments of the (cyclic) text.
            let available = CGFloat(path.length / scale)
            var fragments: [(line: ShapedLine, offset: CGFloat)] = []
            var used: CGFloat = 0
            while available - used > minimumFragmentWidth,
                  let line = shaper.line(startingAt: cursor, maxWidth: available - used),
                  line.width > 0 {
                fragments.append((line, used))
                used += line.width
                cursor += line.characterCount
            }
            guard used > 0 else { continue }

            // Right-to-left text continues leftward, so later fragments go first.
            if shaper.isRightToLeft {
                let shift = path.isClosed ? 0 : available - used
                fragments = fragments.map { ($0.line, shift + used - $0.offset - $0.line.width) }
            }
            let stretch = path.isClosed ? min(max(available / used, 1), maximumRingStretch) : 1

            for (line, offset) in fragments {
                for glyph in line.glyphs {
                    let center = (offset + glyph.position.x + glyph.bounds.midX) * stretch
                    let window = max(Float(glyph.bounds.width) * scale, slices.spacing * 0.5)
                    let frame = path.frame(at: Float(center) * scale, window: window)

                    let right = frame.tangent * scale
                    let up = frame.bitangent * scale
                    let origin = frame.position
                        - right * Float(glyph.bounds.midX)
                        + up * (baselineOffset + Float(glyph.position.y))

                    quads.append(Quad(
                        key: glyph.key,
                        origin: origin,
                        right: right,
                        up: up,
                        normal: frame.normal,
                        color: coloring.color(uv: frame.uv, materialIndex: frame.materialIndex)
                    ))
                    glyphBounds[glyph.key] = glyph.bounds
                }
            }
        }

        if quads.count > maximumGlyphCount {
            quads.removeLast(quads.count - maximumGlyphCount)
        }
        return quads
    }

    // MARK: - Cloud layout

    private static func layOutCloud(
        meshes: [MeshGeometryReader.Mesh],
        shaper: GlyphShaper,
        coloring: Coloring,
        letterSize: TextSculptureOptions.LetterSize,
        glyphBounds: inout [GlyphKey: CGRect]
    ) -> [Quad] {
        let stream = shaper.glyphStream()
        guard !stream.isEmpty else { return [] }

        let count = min(cloudGlyphCount(for: letterSize), maximumGlyphCount)
        let result = SurfaceSampler.sample(meshes: meshes, targetCount: count)
        guard !result.points.isEmpty, result.totalArea > 0 else { return [] }

        let letterHeight = sqrt(result.totalArea / Float(result.points.count)) * cloudLetterScale
        let scale = letterHeight / Float(shaper.lineHeight)
        var rng = SplitMix64(seed: 0x5EED_7E47_5C01_9701)

        var quads: [Quad] = []
        quads.reserveCapacity(result.points.count)

        for (index, point) in result.points.enumerated() {
            let glyph = stream[index % stream.count]
            let (tangent, bitangent) = uprightFrame(normal: point.normal)
            let right = tangent * scale
            let up = bitangent * scale
            let depth = (rng.nextUnitFloat() - 0.5) * cloudDepthJitter * letterHeight

            // Center each letter's ink on its sample point.
            let origin = point.position + point.normal * depth
                - right * Float(glyph.bounds.midX)
                - up * Float(glyph.bounds.midY)

            quads.append(Quad(
                key: glyph.key,
                origin: origin,
                right: right,
                up: up,
                normal: point.normal,
                color: coloring.color(uv: point.uv, materialIndex: point.materialIndex)
            ))
            glyphBounds[glyph.key] = glyph.bounds
        }
        return quads
    }

    static func surfaceArea(of meshes: [MeshGeometryReader.Mesh]) -> Float {
        var area: Float = 0
        for mesh in meshes {
            let positions = mesh.positions
            for submesh in mesh.submeshes {
                for triangle in submesh.triangles {
                    guard triangle.i0 < positions.count, triangle.i1 < positions.count,
                          triangle.i2 < positions.count else { continue }
                    let p0 = positions[triangle.i0]
                    let triangleArea = 0.5 * simd_length(simd_cross(positions[triangle.i1] - p0, positions[triangle.i2] - p0))
                    if triangleArea.isFinite { area += triangleArea }
                }
            }
        }
        return area
    }

    /// Tangent frame on the surface with letters upright where the surface
    /// allows it: `tangent` is horizontal, `bitangent` points as far up as it can.
    static func uprightFrame(normal: SIMD3<Float>) -> (tangent: SIMD3<Float>, bitangent: SIMD3<Float>) {
        let up = SIMD3<Float>(0, 1, 0)
        var tangent = simd_cross(up, normal)
        if simd_length(tangent) < 0.2 {
            // Facing straight up or down: pick any in-plane direction.
            let reference = SIMD3<Float>(1, 0, 0)
            tangent = reference - normal * simd_dot(reference, normal)
        }
        tangent = simd_normalize(tangent)
        return (tangent, simd_normalize(simd_cross(normal, tangent)))
    }

    // MARK: - glTF output

    private static func write(_ quads: [Quad], atlas: GlyphAtlas, to url: URL) throws {
        let vertexCount = quads.count * 4
        var positions = [Float](); positions.reserveCapacity(vertexCount * 3)
        var normals = [Float](); normals.reserveCapacity(vertexCount * 3)
        var texCoords = [Float](); texCoords.reserveCapacity(vertexCount * 2)
        var colors = [Float](); colors.reserveCapacity(vertexCount * 3)
        var indices = [UInt32](); indices.reserveCapacity(quads.count * 6)
        var lower = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var upper = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)

        func appendCorner(_ quad: Quad, x: CGFloat, y: CGFloat, u: Float, v: Float) {
            let position = quad.origin + quad.right * Float(x) + quad.up * Float(y)
            lower = simd_min(lower, position)
            upper = simd_max(upper, position)
            positions.append(position.x); positions.append(position.y); positions.append(position.z)
            normals.append(quad.normal.x); normals.append(quad.normal.y); normals.append(quad.normal.z)
            texCoords.append(u); texCoords.append(v)
            colors.append(quad.color.x); colors.append(quad.color.y); colors.append(quad.color.z)
        }

        for quad in quads {
            guard let entry = atlas.entries[quad.key] else { continue }
            let bounds = entry.bounds
            let base = UInt32(positions.count / 3)

            // Counter-clockwise seen from the normal side: BL, BR, TR, TL.
            appendCorner(quad, x: bounds.minX, y: bounds.minY, u: entry.uvMin.x, v: entry.uvMax.y)
            appendCorner(quad, x: bounds.maxX, y: bounds.minY, u: entry.uvMax.x, v: entry.uvMax.y)
            appendCorner(quad, x: bounds.maxX, y: bounds.maxY, u: entry.uvMax.x, v: entry.uvMin.y)
            appendCorner(quad, x: bounds.minX, y: bounds.maxY, u: entry.uvMin.x, v: entry.uvMin.y)

            indices.append(base); indices.append(base + 1); indices.append(base + 2)
            indices.append(base); indices.append(base + 2); indices.append(base + 3)
        }
        guard !indices.isEmpty, let png = USDZToGLTFConverter.pngData(from: atlas.image) else {
            throw GeneratorError.atlasFailed
        }

        let builder = GLTFBinaryBuilder()
        let positionAccessor = builder.appendFloatBuffer(
            positions,
            componentsPerElement: 3,
            min: [lower.x, lower.y, lower.z],
            max: [upper.x, upper.y, upper.z]
        )
        let normalAccessor = builder.appendFloatBuffer(normals, componentsPerElement: 3)
        let texCoordAccessor = builder.appendFloatBuffer(texCoords, componentsPerElement: 2)
        let colorAccessor = builder.appendFloatBuffer(colors, componentsPerElement: 3)
        let indexAccessor = builder.appendBuffer(
            indices,
            target: GLTFConstants.targetElementArrayBuffer,
            componentType: GLTFConstants.componentTypeUnsignedInt,
            type: "SCALAR"
        )
        let (imageBufferView, _) = builder.appendRawData(png)

        let material = GLTFMaterial(
            name: "Letters",
            pbrMetallicRoughness: GLTFPBRMetallicRoughness(
                baseColorFactor: [1, 1, 1, 1],
                metallicFactor: 0,
                roughnessFactor: 1,
                baseColorTexture: GLTFTextureInfo(index: 0)
            ),
            doubleSided: true,
            alphaMode: "MASK",
            alphaCutoff: alphaCutoff
        )

        let document = GLTFDocument(
            asset: GLTFAssetInfo(version: "2.0", generator: "Reality Pulse"),
            scene: 0,
            scenes: [GLTFScene(nodes: [0])],
            nodes: [GLTFNode(mesh: 0, children: nil, matrix: nil)],
            meshes: [GLTFMesh(primitives: [GLTFPrimitive(
                attributes: [
                    "POSITION": positionAccessor,
                    "NORMAL": normalAccessor,
                    "TEXCOORD_0": texCoordAccessor,
                    "COLOR_0": colorAccessor
                ],
                indices: indexAccessor,
                material: 0,
                mode: GLTFConstants.primitiveModeTriangles
            )])],
            accessors: builder.allAccessors,
            bufferViews: builder.allBufferViews,
            buffers: [GLTFBuffer(byteLength: builder.data.count, uri: nil)],
            materials: [material],
            textures: [GLTFTexture(sampler: 0, source: 0)],
            images: [GLTFImage(uri: nil, mimeType: "image/png", bufferView: imageBufferView)],
            // No mipmaps: averaging an alpha-cut atlas erodes thin strokes below
            // the cutoff, so distant letters would fade away. LINEAR keeps them.
            samplers: [GLTFSampler(
                magFilter: GLTFConstants.filterLinear,
                minFilter: GLTFConstants.filterLinear,
                wrapS: GLTFConstants.wrapClampToEdge,
                wrapT: GLTFConstants.wrapClampToEdge
            )]
        )

        try GLTFWriter.writeGLB(document: document, binaryData: builder.data, to: url)
    }
}

// MARK: - Coloring

/// Per-letter color: the model's base-color texture under the letter, or a
/// single ink. Output is linear RGB, as glTF expects for `COLOR_0`.
private struct Coloring {
    private let samplers: [TextureColorSampler]
    private let ink: SIMD3<Float>?

    init(meshes: [MeshGeometryReader.Mesh], options: TextSculptureOptions) {
        if options.useInkColor {
            ink = Self.linear(options.inkColor)
            samplers = []
        } else {
            // One sampler per flattened material, in the order used for `materialIndex`.
            ink = nil
            samplers = meshes.flatMap { mesh in
                mesh.submeshes.map { TextureColorSampler(material: $0.material) }
            }
        }
    }

    func color(uv: SIMD2<Float>, materialIndex: Int) -> SIMD3<Float> {
        if let ink { return ink }
        guard samplers.indices.contains(materialIndex) else { return SIMD3<Float>(1, 1, 1) }
        return Self.linear(samplers[materialIndex].color(at: uv))
    }

    private static func linear(_ srgb: SIMD3<Float>) -> SIMD3<Float> {
        func channel(_ c: Float) -> Float {
            let clamped = min(max(c, 0), 1)
            return clamped <= 0.04045 ? clamped / 12.92 : pow((clamped + 0.055) / 1.055, 2.4)
        }
        return SIMD3<Float>(channel(srgb.x), channel(srgb.y), channel(srgb.z))
    }
}

// MARK: - Ring path

/// A contour polyline parameterized by arc length and oriented so letters
/// placed along it stand upright and read left to right from outside.
private struct RingPath {

    struct Frame {
        var position: SIMD3<Float>
        var tangent: SIMD3<Float>
        var bitangent: SIMD3<Float>
        var normal: SIMD3<Float>
        var uv: SIMD2<Float>
        var materialIndex: Int
    }

    private let vertices: [ContourSlicer.Vertex]
    /// Arc length at each vertex; closed paths get one extra entry for the closing edge.
    private let cumulative: [Float]
    let isClosed: Bool
    let length: Float

    init(_ polyline: ContourSlicer.Polyline) {
        var vertices = polyline.vertices
        let edgeCount = polyline.isClosed ? vertices.count : vertices.count - 1

        // Letter "up" is normal × direction; flip the path if that points down on average.
        var upness: Float = 0
        for index in 0..<max(edgeCount, 0) {
            let a = vertices[index]
            let b = vertices[(index + 1) % vertices.count]
            upness += simd_cross(a.normal + b.normal, b.position - a.position).y
        }
        if upness < 0 { vertices.reverse() }

        var cumulative: [Float] = [0]
        cumulative.reserveCapacity(vertices.count + 1)
        for index in 0..<max(edgeCount, 0) {
            let step = simd_distance(vertices[index].position, vertices[(index + 1) % vertices.count].position)
            cumulative.append(cumulative[cumulative.count - 1] + step)
        }

        self.vertices = vertices
        self.cumulative = cumulative
        self.isClosed = polyline.isClosed
        self.length = cumulative.last ?? 0
    }

    /// Surface frame at arc length `s`, with the tangent averaged over `window`.
    func frame(at s: Float, window: Float) -> Frame {
        let here = sample(at: s)
        let ahead = sample(at: s + window / 2).position
        let behind = sample(at: s - window / 2).position

        let normal = here.normal
        var direction = ahead - behind
        direction -= normal * simd_dot(direction, normal)
        guard simd_length(direction) > 1e-9 else {
            let (tangent, bitangent) = TextSculptureGenerator.uprightFrame(normal: normal)
            return Frame(position: here.position, tangent: tangent, bitangent: bitangent,
                         normal: normal, uv: here.uv, materialIndex: here.materialIndex)
        }
        let tangent = simd_normalize(direction)
        return Frame(
            position: here.position,
            tangent: tangent,
            bitangent: simd_normalize(simd_cross(normal, tangent)),
            normal: normal,
            uv: here.uv,
            materialIndex: here.materialIndex
        )
    }

    private func sample(at s: Float) -> ContourSlicer.Vertex {
        guard vertices.count > 1, length > 0 else { return vertices[0] }

        var distance = s
        if isClosed {
            distance = distance.truncatingRemainder(dividingBy: length)
            if distance < 0 { distance += length }
        } else {
            distance = min(max(distance, 0), length)
        }

        // Last edge whose start is at or before `distance`.
        var low = 0
        var high = cumulative.count - 2
        while low < high {
            let mid = (low + high + 1) / 2
            if cumulative[mid] <= distance { low = mid } else { high = mid - 1 }
        }

        let a = vertices[low]
        let b = vertices[(low + 1) % vertices.count]
        let edge = cumulative[low + 1] - cumulative[low]
        let t = edge > 0 ? (distance - cumulative[low]) / edge : 0

        let blended = simd_mix(a.normal, b.normal, SIMD3<Float>(repeating: t))
        let normalLength = simd_length(blended)
        return ContourSlicer.Vertex(
            position: simd_mix(a.position, b.position, SIMD3<Float>(repeating: t)),
            normal: normalLength > 0 ? blended / normalLength : a.normal,
            uv: simd_mix(a.uv, b.uv, SIMD2<Float>(repeating: t)),
            materialIndex: t < 0.5 ? a.materialIndex : b.materialIndex
        )
    }
}
