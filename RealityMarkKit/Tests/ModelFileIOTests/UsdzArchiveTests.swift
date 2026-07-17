import XCTest
@testable import ModelFileIO

final class UsdzArchiveTests: XCTestCase {
    private func makeArchive() -> UsdzArchive {
        var payload = Data()
        for index in 0..<10_000 {
            payload.append(UInt8(truncatingIfNeeded: index &* 31 &+ 7))
        }
        return UsdzArchive(entries: [
            .init(name: "model.usdc", data: payload, modTime: 0x6BCD, modDate: 0x58F1),
            .init(name: "0/baked_mesh_tex0.png", data: Data([0x89, 0x50, 0x4E, 0x47]) + Data(count: 300)),
            .init(name: "0/baked_mesh_norm0.png", data: Data([0x89, 0x50, 0x4E, 0x47]) + Data(count: 77)),
        ])
    }

    func testRoundTripPreservesEntriesExactly() throws {
        let archive = makeArchive()
        let serialized = try archive.serialized()
        let reread = try UsdzArchive.read(data: serialized)

        XCTAssertEqual(reread.entries.count, 3)
        for (original, roundTripped) in zip(archive.entries, reread.entries) {
            XCTAssertEqual(roundTripped.name, original.name)
            XCTAssertEqual(roundTripped.data, original.data)
            XCTAssertEqual(roundTripped.modTime, original.modTime)
            XCTAssertEqual(roundTripped.modDate, original.modDate)
        }
    }

    func testEveryEntryDataOffsetIs64ByteAligned() throws {
        let serialized = try makeArchive().serialized()

        // Walk the local headers directly and check each data offset.
        var offset = 0
        var checked = 0
        while offset + 30 <= serialized.count {
            let signature = serialized.subdata(in: offset..<(offset + 4))
            guard signature == Data([0x50, 0x4B, 0x03, 0x04]) else { break }
            let nameLength = Int(serialized[offset + 26]) | (Int(serialized[offset + 27]) << 8)
            let extraLength = Int(serialized[offset + 28]) | (Int(serialized[offset + 29]) << 8)
            let sizeBytes = serialized.subdata(in: (offset + 18)..<(offset + 22))
            let size = sizeBytes.withUnsafeBytes { $0.load(as: UInt32.self) }
            let dataOffset = offset + 30 + nameLength + extraLength
            XCTAssertEqual(dataOffset % 64, 0, "entry \(checked) data offset \(dataOffset)")
            offset = dataOffset + Int(UInt32(littleEndian: size))
            checked += 1
        }
        XCTAssertEqual(checked, 3)
    }

    func testUntouchedEntriesStayByteIdenticalAfterSelectiveEdit() throws {
        var archive = makeArchive()
        let originalUSDC = archive.entries[0].data
        archive.entries[1].data = Data([0x89, 0x50, 0x4E, 0x47]) + Data(repeating: 0xAB, count: 500)

        let reread = try UsdzArchive.read(data: try archive.serialized())
        XCTAssertEqual(reread.entries[0].data, originalUSDC)
        XCTAssertEqual(reread.entries[2].data, makeArchive().entries[2].data)
        XCTAssertEqual(reread.entries[1].data.count, 504)
    }

    func testSystemUnzipAcceptsSerializedArchive() throws {
        let directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let zipURL = directory.appending(path: "fixture.usdz")
        try makeArchive().write(to: zipURL)

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/unzip")
        process.arguments = ["-t", zipURL.path]
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        try process.run()
        process.waitUntilExit()
        let output = String(decoding: pipe.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
        XCTAssertEqual(process.terminationStatus, 0, "unzip -t rejected the archive: \(output)")
    }

    func testReadRejectsCompressedZip() throws {
        // Build a zip with `zip` default (deflate) and confirm we refuse it.
        let directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let fileURL = directory.appending(path: "payload.txt")
        try Data(repeating: 0x41, count: 4096).write(to: fileURL)
        let zipURL = directory.appending(path: "compressed.zip")

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/zip")
        process.arguments = ["-j", "-9", zipURL.path, fileURL.path]
        process.standardOutput = Pipe()
        process.standardError = Pipe()
        try process.run()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            throw XCTSkip("zip tool unavailable")
        }

        XCTAssertThrowsError(try UsdzArchive.read(url: zipURL)) { error in
            guard case UsdzArchiveError.compressedEntryUnsupported = error else {
                return XCTFail("unexpected error: \(error)")
            }
        }
    }

    func testReadAcceptsStoredZipFromSystemTool() throws {
        let directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let fileURL = directory.appending(path: "payload.bin")
        let payload = Data((0..<2048).map { UInt8(truncatingIfNeeded: $0) })
        try payload.write(to: fileURL)
        let zipURL = directory.appending(path: "stored.zip")

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/zip")
        process.arguments = ["-j", "-0", "-X", zipURL.path, fileURL.path]
        process.standardOutput = Pipe()
        process.standardError = Pipe()
        try process.run()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            throw XCTSkip("zip tool unavailable")
        }

        let archive = try UsdzArchive.read(url: zipURL)
        XCTAssertEqual(archive.entries.first?.name, "payload.bin")
        XCTAssertEqual(archive.entries.first?.data, payload)
    }

    func testCRC32MatchesKnownVector() {
        XCTAssertEqual(CRC32.checksum(Data("123456789".utf8)), 0xCBF4_3926)
    }
}
