/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Generates a mesh-derived Gaussian-splat `.ply` from a reconstructed USDZ: samples
points across the surface, matches each to the base-color texture, and writes
them as flat, surface-aligned Gaussian splats. This is a direct geometric
conversion (training-free), not a photometrically optimized 3DGS.
*/

import Foundation
import WatermarkCore
import simd
import os

private let logger = Logger(subsystem: ObjectCaptureReconstructionApp.subsystem,
                            category: "SplatSampleGenerator")

enum SplatSampleGenerator {

    enum GeneratorError: LocalizedError {
        case noGeometry(URL)

        var errorDescription: String? {
            switch self {
            case .noGeometry(let url):
                return "No sampleable mesh geometry found in \(url.lastPathComponent)."
            }
        }
    }

    // Fixed sampling/appearance defaults (no UI). Tunable here.
    //
    // `defaultPointCount` drives density: more points are both more numerous and
    // smaller, since each splat's radius is derived from the average surface
    // spacing `sqrt(area / N)`. `splatSizeFactor` shrinks the disks below that
    // spacing for a finer look (keep it near 1.0 — too small leaves visible gaps).
    static let defaultPointCount = 1_000_000
    static let minimumPointCount = 1_000
    static let splatSizeFactor: Float = 0.85
    static let flatten: Float = 0.1
    static let opacity: Float = 0.9

    /// Sample `usdzURL`'s surface and write a Gaussian-splat `.ply` to `outputURL`.
    /// - Parameter targetCount: number of splats to generate (clamped to a floor).
    ///   Exposed for testing; production callers use the fixed default.
    /// - Parameter watermark: when set, the splat positions are stamped with
    ///   the per-copy key (geometry channel only — splat color carries no mark).
    @discardableResult
    nonisolated static func generate(
        usdzURL: URL,
        outputURL: URL,
        targetCount: Int = defaultPointCount,
        watermark: WatermarkStamp? = nil
    ) throws -> WatermarkStampOutcome {
        let meshes = MeshGeometryReader.loadMeshes(from: usdzURL)
        guard !meshes.isEmpty else { throw GeneratorError.noGeometry(usdzURL) }

        let count = max(minimumPointCount, targetCount)
        let result = SurfaceSampler.sample(meshes: meshes, targetCount: count)
        guard !result.points.isEmpty, result.totalArea > 0 else {
            throw GeneratorError.noGeometry(usdzURL)
        }

        var positions = result.points.map(\.position)
        var geometryInfo: WatermarkRecord.GeometryChannelInfo?
        if let watermark {
            let embedResult = GeometryWatermarker.embed(
                positions: &positions,
                key: watermark.key,
                parameters: watermark.geometryParameters
            )
            if embedResult.isEmbedded {
                geometryInfo = WatermarkRecord.GeometryChannelInfo(
                    parameters: watermark.geometryParameters,
                    effectiveBinCount: embedResult.effectiveBinCount,
                    embeddedBits: embedResult.embeddedBits
                )
            } else {
                logger.log("Geometry watermark skipped for \(outputURL.lastPathComponent, privacy: .public): too few points.")
            }
        }

        // One color sampler per flattened material, in the same mesh-major/submesh
        // order the sampler uses for `materialIndex`.
        var colorSamplers: [TextureColorSampler] = []
        for mesh in meshes {
            for submesh in mesh.submeshes {
                colorSamplers.append(TextureColorSampler(material: submesh.material))
            }
        }

        // Tangential splat radius from the average surface spacing: sqrt(area / N),
        // shrunk by `splatSizeFactor` for finer, less blurry coverage.
        let tangentScale = sqrt(result.totalArea / Float(result.points.count)) * splatSizeFactor

        var splats = [GaussianSplat]()
        splats.reserveCapacity(result.points.count)
        for (index, point) in result.points.enumerated() {
            let color: SIMD3<Float>
            if point.materialIndex >= 0, point.materialIndex < colorSamplers.count {
                color = colorSamplers[point.materialIndex].color(at: point.uv)
            } else {
                color = SIMD3<Float>(1, 1, 1)
            }

            splats.append(GaussianSplat.surfaceSplat(
                position: positions[index],
                normal: point.normal,
                color: color,
                tangentScale: tangentScale,
                flatten: flatten,
                opacity: opacity
            ))
        }

        try SplatPLYWriter.write(splats, to: outputURL)
        logger.log("Generated splat \(outputURL.lastPathComponent, privacy: .public) with \(splats.count) points from \(usdzURL.lastPathComponent, privacy: .public)")
        return WatermarkStampOutcome(geometry: geometryInfo, texture: nil)
    }
}
