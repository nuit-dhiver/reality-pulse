import CryptoKit
import Foundation

/// Deterministic keyed pseudorandom stream: block_i = HMAC-SHA256(key, context ‖ LE64(i)).
///
/// Domain separation comes from the context string, so independent decisions
/// (geometry bits, bin permutation, texture chips) draw from independent
/// streams even under the same key.
public struct KeyedPRF {
    private let key: SymmetricKey
    private let contextTag: [UInt8]
    private var blockIndex: UInt64 = 0
    private var buffer: [UInt8] = []
    private var bufferOffset = 0
    private var bitBuffer: UInt8 = 0
    private var bitsRemaining = 0

    public init(key: WatermarkKey, context: String) {
        self.key = SymmetricKey(data: key.data)
        self.contextTag = Array(context.utf8)
    }

    private mutating func refill() {
        var message = contextTag
        withUnsafeBytes(of: blockIndex.littleEndian) { message.append(contentsOf: $0) }
        let mac = HMAC<SHA256>.authenticationCode(for: Data(message), using: key)
        buffer = Array(mac)
        bufferOffset = 0
        blockIndex += 1
    }

    public mutating func nextByte() -> UInt8 {
        if bufferOffset >= buffer.count { refill() }
        defer { bufferOffset += 1 }
        return buffer[bufferOffset]
    }

    public mutating func nextBit() -> Bool {
        if bitsRemaining == 0 {
            bitBuffer = nextByte()
            bitsRemaining = 8
        }
        let bit = bitBuffer & 1
        bitBuffer >>= 1
        bitsRemaining -= 1
        return bit == 1
    }

    /// ±1 chip for spread-spectrum embedding.
    public mutating func nextChip() -> Float {
        nextBit() ? 1 : -1
    }

    public mutating func nextUInt64() -> UInt64 {
        var value: UInt64 = 0
        for _ in 0..<8 {
            value = (value << 8) | UInt64(nextByte())
        }
        return value
    }

    /// Unbiased integer in 0..<upperBound via rejection sampling.
    public mutating func next(upperBound: Int) -> Int {
        precondition(upperBound > 0)
        let bound = UInt64(upperBound)
        let limit = UInt64.max - UInt64.max % bound
        while true {
            let value = nextUInt64()
            if value < limit { return Int(value % bound) }
        }
    }

    /// Keyed Fisher–Yates permutation of 0..<count.
    public mutating func permutation(count: Int) -> [Int] {
        var permutation = Array(0..<count)
        guard count > 1 else { return permutation }
        for i in stride(from: count - 1, through: 1, by: -1) {
            permutation.swapAt(i, next(upperBound: i + 1))
        }
        return permutation
    }
}
