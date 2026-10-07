/*
See the LICENSE file for licensing information.

Abstract:
Per-job settings for the text-sculpture export: how letters are laid out on the
model, how large they are, which text they spell, and how they are colored.
*/

import Foundation
import simd

struct TextSculptureOptions: Codable, Equatable {

    enum Layout: String, Codable, CaseIterable {
        /// Readable lines of text wrapping the model along horizontal slices.
        case rings
        /// One character per sampled surface point, like a point cloud.
        case cloud

        var displayName: String {
            switch self {
            case .rings: return "Rings"
            case .cloud: return "Cloud"
            }
        }
    }

    enum LetterSize: String, Codable, CaseIterable {
        case small
        case medium
        case large

        var displayName: String {
            switch self {
            case .small: return "Small"
            case .medium: return "Medium"
            case .large: return "Large"
            }
        }
    }

    var layout: Layout = .rings
    var letterSize: LetterSize = .medium
    /// The words to write. Empty means "use the model name".
    var text: String = ""
    /// When `true`, every letter uses `inkColor` instead of the model's texture.
    var useInkColor: Bool = false
    /// sRGB components in `[0, 1]`, as picked in the UI.
    var inkColor: SIMD3<Float> = SIMD3<Float>(0.08, 0.08, 0.1)

    init() {}

    // Lenient decoding: a missing or unrecognized value falls back to its
    // default, so saved jobs keep loading when options are added or renamed.
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let defaults = TextSculptureOptions()
        layout = (try? container.decodeIfPresent(Layout.self, forKey: .layout)) ?? defaults.layout
        letterSize = (try? container.decodeIfPresent(LetterSize.self, forKey: .letterSize)) ?? defaults.letterSize
        text = (try? container.decodeIfPresent(String.self, forKey: .text)) ?? defaults.text
        useInkColor = (try? container.decodeIfPresent(Bool.self, forKey: .useInkColor)) ?? defaults.useInkColor
        inkColor = (try? container.decodeIfPresent(SIMD3<Float>.self, forKey: .inkColor)) ?? defaults.inkColor
    }
}
