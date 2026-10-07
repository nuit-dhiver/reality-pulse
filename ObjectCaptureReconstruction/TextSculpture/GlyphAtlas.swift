/*
See the LICENSE file for licensing information.

Abstract:
Rasterizes every distinct glyph used by a text sculpture into one texture:
white letters on a transparent background, so a material's alpha cutout
shapes each quad into its letter and per-vertex color tints it.
*/

import Foundation
import CoreText
import CoreGraphics
import simd

final class GlyphAtlas {

    struct Entry {
        /// Padded ink bounds in shaping font units, relative to the pen position.
        /// The quad drawn for the glyph spans exactly this rectangle.
        var bounds: CGRect
        /// Top-left texture coordinate (glTF convention: origin top-left, v down).
        var uvMin: SIMD2<Float>
        /// Bottom-right texture coordinate.
        var uvMax: SIMD2<Float>
    }

    let image: CGImage
    let entries: [GlyphKey: Entry]

    /// Raster height, in pixels, of one full line (ascent + descent).
    static let targetLinePixels: CGFloat = 72
    /// Smallest line height the atlas will shrink to when glyphs don't fit.
    static let minimumLinePixels: CGFloat = 12
    /// Transparent border around each glyph so mipmaps don't bleed neighbors in.
    static let padding: CGFloat = 3
    static let atlasWidths = [1024, 2048, 4096]

    /// - Parameters:
    ///   - glyphs: each distinct glyph with its unpadded ink bounds in font units.
    ///   - fonts: the shaping fonts, keyed by `GlyphKey.fontName`.
    ///   - lineHeight: ascent + descent of the base font, in font units.
    init?(glyphs: [GlyphKey: CGRect], fonts: [String: CTFont], lineHeight: CGFloat) {
        guard !glyphs.isEmpty, lineHeight > 0 else { return nil }

        // Tallest first for tight shelves; name/glyph tie-breaks keep output deterministic.
        let ordered = glyphs.sorted { lhs, rhs in
            if lhs.value.height != rhs.value.height { return lhs.value.height > rhs.value.height }
            if lhs.key.fontName != rhs.key.fontName { return lhs.key.fontName < rhs.key.fontName }
            return lhs.key.glyph < rhs.key.glyph
        }

        var scale = Self.targetLinePixels / lineHeight
        var layout: (width: Int, height: Int, origins: [GlyphKey: (x: Int, y: Int)])?
        while layout == nil, scale * lineHeight >= Self.minimumLinePixels {
            for width in Self.atlasWidths {
                if let packed = Self.pack(ordered, scale: scale, width: width) {
                    layout = (width, packed.height, packed.origins)
                    break
                }
            }
            if layout == nil { scale *= 0.75 }
        }
        guard let layout else { return nil }

        let width = layout.width
        let height = layout.height
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
              let context = CGContext(
                data: nil,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: width * 4,
                space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
              ) else { return nil }

        context.clear(CGRect(x: 0, y: 0, width: width, height: height))
        context.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
        context.setShouldSmoothFonts(false)

        var rasterFonts: [String: CTFont] = [:]
        var entries: [GlyphKey: Entry] = [:]
        let pad = Self.padding
        let atlasSize = SIMD2<Float>(Float(width), Float(height))

        for (key, bounds) in ordered {
            guard let origin = layout.origins[key], let font = fonts[key.fontName] else { continue }
            let rasterFont: CTFont
            if let cached = rasterFonts[key.fontName] {
                rasterFont = cached
            } else {
                rasterFont = CTFontCreateCopyWithAttributes(font, CTFontGetSize(font) * scale, nil, nil)
                rasterFonts[key.fontName] = rasterFont
            }

            // Anchor the padded glyph at its cell's bottom-left corner.
            let cellHeight = ceil(bounds.height * scale + 2 * pad)
            let paddedWidth = bounds.width * scale + 2 * pad
            let paddedHeight = bounds.height * scale + 2 * pad
            let cellBottomFromTop = CGFloat(origin.y) + cellHeight

            // Core Graphics has a bottom-left origin.
            var pen = CGPoint(
                x: CGFloat(origin.x) + pad - bounds.minX * scale,
                y: CGFloat(height) - cellBottomFromTop + pad - bounds.minY * scale
            )
            var glyph = key.glyph
            CTFontDrawGlyphs(rasterFont, &glyph, &pen, 1, context)

            entries[key] = Entry(
                bounds: bounds.insetBy(dx: -pad / scale, dy: -pad / scale),
                uvMin: SIMD2<Float>(Float(origin.x), Float(cellBottomFromTop - paddedHeight)) / atlasSize,
                uvMax: SIMD2<Float>(Float(CGFloat(origin.x) + paddedWidth), Float(cellBottomFromTop)) / atlasSize
            )
        }

        guard let image = context.makeImage() else { return nil }
        self.image = image
        self.entries = entries
    }

    /// Shelf-pack glyph cells left to right, top to bottom. Returns top-left
    /// cell origins and a power-of-two atlas height, or `nil` if they don't fit.
    private static func pack(
        _ glyphs: [(key: GlyphKey, value: CGRect)],
        scale: CGFloat,
        width: Int
    ) -> (height: Int, origins: [GlyphKey: (x: Int, y: Int)])? {
        var origins: [GlyphKey: (x: Int, y: Int)] = [:]
        var x = 0
        var y = 0
        var shelfHeight = 0

        for (key, bounds) in glyphs {
            let cellWidth = Int(ceil(bounds.width * scale + 2 * padding))
            let cellHeight = Int(ceil(bounds.height * scale + 2 * padding))
            guard cellWidth <= width else { return nil }
            if x + cellWidth > width {
                y += shelfHeight
                x = 0
                shelfHeight = 0
            }
            origins[key] = (x, y)
            x += cellWidth
            shelfHeight = max(shelfHeight, cellHeight)
        }

        let usedHeight = y + shelfHeight
        var height = 64
        while height < usedHeight { height *= 2 }
        guard height <= width else { return nil }
        return (height, origins)
    }
}
