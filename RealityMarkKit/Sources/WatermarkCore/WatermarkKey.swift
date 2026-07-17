import CryptoKit
import Foundation

public enum WatermarkError: Error, Equatable {
    case invalidKeyLength
    case unsupportedImage
}

/// A per-copy 256-bit secret key. Following Kerckhoffs's principle, this key is
/// the only secret in the system: the embedding and detection algorithms are
/// public, and every keyed decision (bit sequence, bin permutation, chip signs)
/// is derived from this key through an HMAC-SHA256 PRF.
public struct WatermarkKey: Sendable, Equatable {
    public static let byteCount = 32

    public let data: Data

    public init(data: Data) throws {
        guard data.count == Self.byteCount else { throw WatermarkError.invalidKeyLength }
        self.data = data
    }

    public static func random() -> WatermarkKey {
        let key = SymmetricKey(size: .bits256)
        let data = key.withUnsafeBytes { Data($0) }
        return try! WatermarkKey(data: data)
    }
}
