import Foundation

public enum UsdzArchiveError: Error, Equatable {
    case notAZipArchive
    case corruptArchive(String)
    case compressedEntryUnsupported(String)
    case entryTooLarge(String)
}

/// Minimal reader/writer for the zip subset usdz uses: stored (uncompressed)
/// entries whose file data starts on a 64-byte boundary (padding lives in a
/// local-header extra field, the same technique OpenUSD's writer uses).
///
/// Reading tolerates any stored zip; writing always produces spec-conformant
/// alignment, so read → replace texture bytes → write is a safe round trip
/// that leaves untouched entries byte-identical.
public struct UsdzArchive {
    public struct Entry {
        public var name: String
        public var data: Data
        /// DOS mod time/date preserved from the source archive.
        public var modTime: UInt16
        public var modDate: UInt16

        public init(name: String, data: Data, modTime: UInt16 = 0, modDate: UInt16 = 0) {
            self.name = name
            self.data = data
            self.modTime = modTime
            self.modDate = modDate
        }
    }

    public var entries: [Entry]

    public init(entries: [Entry]) {
        self.entries = entries
    }

    // MARK: - Reading

    public static func read(url: URL) throws -> UsdzArchive {
        try read(data: Data(contentsOf: url))
    }

    public static func read(data: Data) throws -> UsdzArchive {
        guard let eocdOffset = findEndOfCentralDirectory(data) else {
            throw UsdzArchiveError.notAZipArchive
        }

        let entryCount = Int(readUInt16(data, at: eocdOffset + 10))
        let centralDirectoryOffset = Int(readUInt32(data, at: eocdOffset + 16))

        var entries: [Entry] = []
        var cursor = centralDirectoryOffset
        for _ in 0..<entryCount {
            guard cursor + 46 <= data.count,
                  readUInt32(data, at: cursor) == 0x0201_4B50 else {
                throw UsdzArchiveError.corruptArchive("central directory entry")
            }

            let method = readUInt16(data, at: cursor + 10)
            let modTime = readUInt16(data, at: cursor + 12)
            let modDate = readUInt16(data, at: cursor + 14)
            let compressedSize = Int(readUInt32(data, at: cursor + 20))
            let nameLength = Int(readUInt16(data, at: cursor + 28))
            let extraLength = Int(readUInt16(data, at: cursor + 30))
            let commentLength = Int(readUInt16(data, at: cursor + 32))
            let localHeaderOffset = Int(readUInt32(data, at: cursor + 42))

            guard cursor + 46 + nameLength <= data.count else {
                throw UsdzArchiveError.corruptArchive("entry name")
            }
            let name = String(
                decoding: data.subdata(in: (cursor + 46)..<(cursor + 46 + nameLength)),
                as: UTF8.self
            )

            guard method == 0 else {
                throw UsdzArchiveError.compressedEntryUnsupported(name)
            }

            guard localHeaderOffset + 30 <= data.count,
                  readUInt32(data, at: localHeaderOffset) == 0x0403_4B50 else {
                throw UsdzArchiveError.corruptArchive("local header for \(name)")
            }
            let localNameLength = Int(readUInt16(data, at: localHeaderOffset + 26))
            let localExtraLength = Int(readUInt16(data, at: localHeaderOffset + 28))
            let dataOffset = localHeaderOffset + 30 + localNameLength + localExtraLength
            guard dataOffset + compressedSize <= data.count else {
                throw UsdzArchiveError.corruptArchive("entry data for \(name)")
            }

            entries.append(Entry(
                name: name,
                data: data.subdata(in: dataOffset..<(dataOffset + compressedSize)),
                modTime: modTime,
                modDate: modDate
            ))

            cursor += 46 + nameLength + extraLength + commentLength
        }

        return UsdzArchive(entries: entries)
    }

    private static func findEndOfCentralDirectory(_ data: Data) -> Int? {
        // EOCD is 22 bytes plus an up-to-64KB comment; scan backward.
        let minOffset = max(0, data.count - 22 - 65_535)
        guard data.count >= 22 else { return nil }
        var offset = data.count - 22
        while offset >= minOffset {
            if readUInt32(data, at: offset) == 0x0605_4B50 {
                return offset
            }
            offset -= 1
        }
        return nil
    }

    // MARK: - Writing

    public func write(to url: URL) throws {
        try serialized().write(to: url, options: .atomic)
    }

    public func serialized() throws -> Data {
        var output = Data()
        var centralRecords: [(entry: Entry, crc: UInt32, localHeaderOffset: Int)] = []

        for entry in entries {
            guard entry.data.count <= UInt32.max else {
                throw UsdzArchiveError.entryTooLarge(entry.name)
            }
            let nameBytes = Array(entry.name.utf8)
            let localHeaderOffset = output.count
            let crc = CRC32.checksum(entry.data)

            // Extra-field padding so the entry's data starts 64-byte aligned
            // (usdz requirement). The 4-byte extra header itself counts.
            let baseOffset = localHeaderOffset + 30 + nameBytes.count
            let padding = (64 - (baseOffset + 4) % 64) % 64
            let extraLength = 4 + padding

            appendUInt32(&output, 0x0403_4B50)
            appendUInt16(&output, 20)                       // version needed
            appendUInt16(&output, 0)                        // flags
            appendUInt16(&output, 0)                        // method: stored
            appendUInt16(&output, entry.modTime)
            appendUInt16(&output, entry.modDate)
            appendUInt32(&output, crc)
            appendUInt32(&output, UInt32(entry.data.count)) // compressed
            appendUInt32(&output, UInt32(entry.data.count)) // uncompressed
            appendUInt16(&output, UInt16(nameBytes.count))
            appendUInt16(&output, UInt16(extraLength))
            output.append(contentsOf: nameBytes)
            appendUInt16(&output, 0x1986)                   // OpenUSD padding field ID
            appendUInt16(&output, UInt16(padding))
            output.append(Data(count: padding))

            assert(output.count % 64 == 0, "usdz entry data must be 64-byte aligned")
            output.append(entry.data)

            centralRecords.append((entry, crc, localHeaderOffset))
        }

        let centralDirectoryOffset = output.count
        for (entry, crc, localHeaderOffset) in centralRecords {
            let nameBytes = Array(entry.name.utf8)
            appendUInt32(&output, 0x0201_4B50)
            appendUInt16(&output, 20)                       // version made by
            appendUInt16(&output, 20)                       // version needed
            appendUInt16(&output, 0)                        // flags
            appendUInt16(&output, 0)                        // method: stored
            appendUInt16(&output, entry.modTime)
            appendUInt16(&output, entry.modDate)
            appendUInt32(&output, crc)
            appendUInt32(&output, UInt32(entry.data.count))
            appendUInt32(&output, UInt32(entry.data.count))
            appendUInt16(&output, UInt16(nameBytes.count))
            appendUInt16(&output, 0)                        // extra length
            appendUInt16(&output, 0)                        // comment length
            appendUInt16(&output, 0)                        // disk number
            appendUInt16(&output, 0)                        // internal attrs
            appendUInt32(&output, 0)                        // external attrs
            appendUInt32(&output, UInt32(localHeaderOffset))
            output.append(contentsOf: nameBytes)
        }
        let centralDirectorySize = output.count - centralDirectoryOffset

        appendUInt32(&output, 0x0605_4B50)
        appendUInt16(&output, 0)                            // disk number
        appendUInt16(&output, 0)                            // central directory disk
        appendUInt16(&output, UInt16(centralRecords.count))
        appendUInt16(&output, UInt16(centralRecords.count))
        appendUInt32(&output, UInt32(centralDirectorySize))
        appendUInt32(&output, UInt32(centralDirectoryOffset))
        appendUInt16(&output, 0)                            // comment length

        return output
    }

    // MARK: - Byte helpers

    private static func readUInt16(_ data: Data, at offset: Int) -> UInt16 {
        var value: UInt16 = 0
        withUnsafeMutableBytes(of: &value) { destination in
            data.copyBytes(to: destination, from: offset..<(offset + 2))
        }
        return UInt16(littleEndian: value)
    }

    private static func readUInt32(_ data: Data, at offset: Int) -> UInt32 {
        var value: UInt32 = 0
        withUnsafeMutableBytes(of: &value) { destination in
            data.copyBytes(to: destination, from: offset..<(offset + 4))
        }
        return UInt32(littleEndian: value)
    }

    private func appendUInt16(_ data: inout Data, _ value: UInt16) {
        withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
    }

    private func appendUInt32(_ data: inout Data, _ value: UInt32) {
        withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
    }
}

/// Standard CRC-32 (IEEE 802.3), table-driven — avoids linking zlib.
enum CRC32 {
    private static let table: [UInt32] = (0..<256).map { index -> UInt32 in
        var value = UInt32(index)
        for _ in 0..<8 {
            value = (value & 1) == 1 ? (0xEDB8_8320 ^ (value >> 1)) : (value >> 1)
        }
        return value
    }

    static func checksum(_ data: Data) -> UInt32 {
        var crc: UInt32 = 0xFFFF_FFFF
        for byte in data {
            crc = table[Int((crc ^ UInt32(byte)) & 0xFF)] ^ (crc >> 8)
        }
        return crc ^ 0xFFFF_FFFF
    }
}
