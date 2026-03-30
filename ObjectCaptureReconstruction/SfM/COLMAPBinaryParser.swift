/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Parses COLMAP sparse reconstruction binary files into SfMResult.
*/

import Foundation
import simd
import os

private let logger = Logger(subsystem: ObjectCaptureReconstructionApp.subsystem,
                            category: "COLMAPBinaryParser")

/// Reads COLMAP binary output files and produces an `SfMResult`.
struct COLMAPBinaryParser {

    /// Parse all three binary files from a COLMAP sparse reconstruction directory.
    static func parse(directory: URL) throws -> SfMResult {
        let cameras = try parseCameras(from: directory.appending(path: "cameras.bin"))
        let images = try parseImages(from: directory.appending(path: "images.bin"))
        let points3D = try parsePoints3D(from: directory.appending(path: "points3D.bin"))

        logger.log("Parsed COLMAP output: \(cameras.count) camera(s), \(images.count) image(s), \(points3D.count) point(s)")
        return SfMResult(cameras: cameras, images: images, points3D: points3D)
    }

    // MARK: - cameras.bin

    static func parseCameras(from url: URL) throws -> [CameraIntrinsics] {
        let data = try Data(contentsOf: url)
        var reader = BinaryReader(data: data)

        let numCameras = try reader.readUInt64()
        var cameras: [CameraIntrinsics] = []
        cameras.reserveCapacity(Int(numCameras))

        for _ in 0..<numCameras {
            let cameraId = try reader.readUInt32()
            let modelId = try reader.readInt32()
            let width = try reader.readUInt64()
            let height = try reader.readUInt64()

            guard let model = COLMAPCameraModel(rawValue: modelId) else {
                throw COLMAPError.invalidOutput("Unknown camera model ID: \(modelId)")
            }

            var params: [Double] = []
            for _ in 0..<model.paramCount {
                params.append(try reader.readDouble())
            }

            cameras.append(CameraIntrinsics(
                id: cameraId,
                model: model,
                width: width,
                height: height,
                params: params
            ))
        }

        return cameras
    }

    // MARK: - images.bin

    static func parseImages(from url: URL) throws -> [CameraPose] {
        let data = try Data(contentsOf: url)
        var reader = BinaryReader(data: data)

        let numImages = try reader.readUInt64()
        var images: [CameraPose] = []
        images.reserveCapacity(Int(numImages))

        for _ in 0..<numImages {
            let imageId = try reader.readUInt32()

            let qw = try reader.readDouble()
            let qx = try reader.readDouble()
            let qy = try reader.readDouble()
            let qz = try reader.readDouble()

            let tx = try reader.readDouble()
            let ty = try reader.readDouble()
            let tz = try reader.readDouble()

            let cameraId = try reader.readUInt32()
            let name = try reader.readNullTerminatedString()

            let numPoints2D = try reader.readUInt64()
            var points2D: [Point2D] = []
            points2D.reserveCapacity(Int(numPoints2D))

            for _ in 0..<numPoints2D {
                let x = try reader.readDouble()
                let y = try reader.readDouble()
                let point3DId = try reader.readUInt64()

                points2D.append(Point2D(
                    x: x,
                    y: y,
                    point3DId: point3DId == UInt64.max ? nil : point3DId
                ))
            }

            images.append(CameraPose(
                id: imageId,
                cameraId: cameraId,
                imageName: name,
                rotation: simd_quatd(ix: qx, iy: qy, iz: qz, r: qw),
                translation: SIMD3<Double>(tx, ty, tz),
                points2D: points2D
            ))
        }

        return images
    }

    // MARK: - points3D.bin

    static func parsePoints3D(from url: URL) throws -> [SparsePoint3D] {
        let data = try Data(contentsOf: url)
        var reader = BinaryReader(data: data)

        let numPoints = try reader.readUInt64()
        var points: [SparsePoint3D] = []
        points.reserveCapacity(Int(numPoints))

        for _ in 0..<numPoints {
            let pointId = try reader.readUInt64()

            let x = try reader.readDouble()
            let y = try reader.readDouble()
            let z = try reader.readDouble()

            let r = try reader.readUInt8()
            let g = try reader.readUInt8()
            let b = try reader.readUInt8()

            let error = try reader.readDouble()

            let trackLength = try reader.readUInt64()
            var track: [TrackElement] = []
            track.reserveCapacity(Int(trackLength))

            for _ in 0..<trackLength {
                let imageId = try reader.readUInt32()
                let point2DIdx = try reader.readUInt32()
                track.append(TrackElement(imageId: imageId, point2DIndex: point2DIdx))
            }

            points.append(SparsePoint3D(
                id: pointId,
                position: SIMD3<Double>(x, y, z),
                color: SIMD3<UInt8>(r, g, b),
                reprojectionError: error,
                track: track
            ))
        }

        return points
    }
}

// MARK: - Binary Reader

/// Sequential little-endian binary reader.
struct BinaryReader {
    let data: Data
    private(set) var offset: Int = 0

    init(data: Data) {
        self.data = data
    }

    var bytesRemaining: Int { data.count - offset }

    mutating func readUInt8() throws -> UInt8 {
        guard offset + 1 <= data.count else { throw BinaryReaderError.unexpectedEOF }
        let value = data[data.startIndex + offset]
        offset += 1
        return value
    }

    mutating func readUInt32() throws -> UInt32 {
        try readFixedWidth()
    }

    mutating func readInt32() throws -> Int32 {
        try readFixedWidth()
    }

    mutating func readUInt64() throws -> UInt64 {
        try readFixedWidth()
    }

    mutating func readDouble() throws -> Double {
        let bits: UInt64 = try readFixedWidth()
        return Double(bitPattern: bits)
    }

    mutating func readNullTerminatedString() throws -> String {
        var bytes: [UInt8] = []
        while offset < data.count {
            let byte = data[data.startIndex + offset]
            offset += 1
            if byte == 0 { break }
            bytes.append(byte)
        }
        guard let str = String(bytes: bytes, encoding: .utf8) else {
            throw BinaryReaderError.invalidString
        }
        return str
    }

    // MARK: - Private

    private mutating func readFixedWidth<T: FixedWidthInteger>() throws -> T {
        let size = MemoryLayout<T>.size
        guard offset + size <= data.count else { throw BinaryReaderError.unexpectedEOF }

        let value = data.withUnsafeBytes { ptr in
            ptr.loadUnaligned(fromByteOffset: offset, as: T.self)
        }
        offset += size
        return T(littleEndian: value)
    }
}

enum BinaryReaderError: LocalizedError {
    case unexpectedEOF
    case invalidString

    var errorDescription: String? {
        switch self {
        case .unexpectedEOF: return "Unexpected end of binary data"
        case .invalidString: return "Invalid UTF-8 string in binary data"
        }
    }
}
