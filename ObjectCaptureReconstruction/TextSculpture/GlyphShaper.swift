/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Shapes the sculpture's text with Core Text, so any script works: Latin, CJK, and
right-to-left joined scripts like Persian and Arabic. The text is treated as
cyclic: when a line reaches the end, the next line starts over from the top.
*/

import Foundation
import AppKit
import CoreText

/// Identifies one rasterizable glyph: Core Text may substitute fallback fonts
/// for scripts the base font lacks, so the font is part of the identity.
struct GlyphKey: Hashable {
    var fontName: String
    var glyph: CGGlyph
}

/// A visible (non-whitespace) glyph on a shaped line, in the shaper's font units.
struct ShapedGlyph {
    var key: GlyphKey
    /// Pen position relative to the start of the line's baseline.
    var position: CGPoint
    /// Ink bounds relative to the pen position.
    var bounds: CGRect
}

struct ShapedLine {
    var glyphs: [ShapedGlyph]
    /// Typographic width, including any trailing whitespace.
    var width: CGFloat
    /// Number of UTF-16 units consumed from the text.
    var characterCount: Int
}

final class GlyphShaper {

    /// Appended between repeats of the text so the loop reads as a sentence break.
    static let separator = " · "

    let fontSize: CGFloat
    let ascent: CGFloat
    let descent: CGFloat
    var lineHeight: CGFloat { ascent + descent }

    /// The normalized text actually typeset, including the trailing separator.
    let text: String
    /// Whether the text's first letter is right-to-left (Arabic, Persian, Hebrew, …).
    /// Successive line fragments on a ring then run right to left as well.
    let isRightToLeft: Bool

    /// Every font Core Text used so far, keyed by `GlyphKey.fontName`.
    private(set) var fonts: [String: CTFont] = [:]

    private let baseFont: CTFont
    private let length: Int
    private let typesetter: CTTypesetter

    init(text: String, fallbackText: String, fontSize: CGFloat = 64) {
        var body = Self.normalize(text)
        if body.isEmpty { body = Self.normalize(fallbackText) }
        if body.isEmpty { body = "·" }

        self.fontSize = fontSize
        self.text = body + Self.separator
        self.isRightToLeft = Self.startsRightToLeft(body)

        baseFont = Self.defaultFont(size: fontSize)
        ascent = CTFontGetAscent(baseFont)
        descent = CTFontGetDescent(baseFont)

        let attributed = NSAttributedString(string: self.text, attributes: [.font: baseFont])
        length = attributed.length
        typesetter = CTTypesetterCreateWithAttributedString(attributed)
    }

    /// The next line of text starting at `index` (wrapped into the text) that
    /// fits in `maxWidth` font units, breaking between words when possible.
    /// Returns `nil` when not even a single character fits.
    func line(startingAt index: Int, maxWidth: CGFloat) -> ShapedLine? {
        guard maxWidth > 0, length > 0 else { return nil }
        let start = ((index % length) + length) % length
        let width = Double(maxWidth)

        // Prefer breaking between words; split inside a word only if it alone is too wide.
        if let line = fittingLine(start: start, count: CTTypesetterSuggestLineBreak(typesetter, start, width), maxWidth: maxWidth) {
            return line
        }
        return fittingLine(start: start, count: CTTypesetterSuggestClusterBreak(typesetter, start, width), maxWidth: maxWidth)
    }

    /// The whole text as one line of visible glyphs, for layouts that consume
    /// characters one at a time.
    func glyphStream() -> [ShapedGlyph] {
        glyphs(in: CTTypesetterCreateLine(typesetter, CFRange(location: 0, length: length)))
    }

    // MARK: - Helpers

    private func fittingLine(start: Int, count: Int, maxWidth: CGFloat) -> ShapedLine? {
        guard count > 0 else { return nil }
        let line = CTTypesetterCreateLine(typesetter, CFRange(location: start, length: count))
        let total = CGFloat(CTLineGetTypographicBounds(line, nil, nil, nil))
        let trailing = CGFloat(CTLineGetTrailingWhitespaceWidth(line))
        guard total - trailing <= maxWidth + 0.5 else { return nil }
        return ShapedLine(glyphs: glyphs(in: line), width: total, characterCount: count)
    }

    private func glyphs(in line: CTLine) -> [ShapedGlyph] {
        var result: [ShapedGlyph] = []
        let runs = CTLineGetGlyphRuns(line) as? [CTRun] ?? []

        for run in runs {
            let count = CTRunGetGlyphCount(run)
            guard count > 0 else { continue }

            let attributes = CTRunGetAttributes(run) as NSDictionary
            let font = attributes[kCTFontAttributeName as String].map { $0 as! CTFont } ?? baseFont
            let fontName = CTFontCopyPostScriptName(font) as String
            fonts[fontName] = font

            var glyphs = [CGGlyph](repeating: 0, count: count)
            var positions = [CGPoint](repeating: .zero, count: count)
            var bounds = [CGRect](repeating: .zero, count: count)
            CTRunGetGlyphs(run, CFRange(location: 0, length: 0), &glyphs)
            CTRunGetPositions(run, CFRange(location: 0, length: 0), &positions)
            CTFontGetBoundingRectsForGlyphs(font, .horizontal, glyphs, &bounds, count)

            for index in 0..<count {
                let rect = bounds[index]
                guard rect.width > 0, rect.height > 0, rect.origin.x.isFinite else { continue }
                result.append(ShapedGlyph(
                    key: GlyphKey(fontName: fontName, glyph: glyphs[index]),
                    position: positions[index],
                    bounds: rect
                ))
            }
        }
        return result
    }

    /// Collapse every whitespace run (including newlines) into a single space.
    static func normalize(_ text: String) -> String {
        text.split(whereSeparator: { $0.isWhitespace }).joined(separator: " ")
    }

    private static func startsRightToLeft(_ text: String) -> Bool {
        for scalar in text.unicodeScalars where scalar.properties.isAlphabetic {
            switch scalar.value {
            case 0x0590...0x08FF, 0xFB1D...0xFDFF, 0xFE70...0xFEFF, 0x10800...0x10FFF, 0x1E800...0x1EFFF:
                return true
            default:
                return false
            }
        }
        return false
    }

    /// The system serif (New York) in a semibold weight, which stays legible
    /// when letters are small. Core Text falls back per script automatically.
    private static func defaultFont(size: CGFloat) -> CTFont {
        let system = NSFont.systemFont(ofSize: size, weight: .semibold)
        if let serif = system.fontDescriptor.withDesign(.serif),
           let font = NSFont(descriptor: serif, size: size) {
            return font as CTFont
        }
        return system as CTFont
    }
}
