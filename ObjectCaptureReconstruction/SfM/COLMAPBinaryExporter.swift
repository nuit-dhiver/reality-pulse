/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Writes SfM results in COLMAP binary format (cameras.bin, images.bin, points3D.bin).
*/

import Foundation
import os

private let logger = Logger(subsystem: ObjectCaptureReconstructionApp.subsystem,
                            category: "COLMAPBinaryExporter")

/// Exports `SfMResult` data to COLMAP binary files.
struct COLMAPBinaryExporter {

    /// Write all three COLMAP binary files to the given directory.
    static func export(_ result: SfMResult, to directory: URL) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        try writeCameras(result.cameras, to: directory.appending(path: "cameras.bin"))
        try writeImages(result.images, to: directory.appending(path: "images.bin"))
        try writePoints3D(result.points3D, to: directory.appending(path: "points3D.bin"))

        logger.log("Exported COLMAP binary files to \(directory.path())")
    }

    // MARK: - cameras.bin

    /// Format: uint64 num_cameras, then per camera:
    /// uint32 camera_id, int32 model_id, uint64 width, uint64 height, double params[]
    static func writeCameras(_ cameras: [CameraIntrinsics], to url: URL) throws {
        var data = Data()
        data.appendLittleEndian(UInt64(cameras.count))

        for camera in cameras {
            data.appendLittleEndian(camera.id)
            data.appendLittleEndian(camera.model.rawValue)
            data.appendLittleEndian(camera.width)
            data.appendLittleEndian(camera.height)

            for param in camera.params {
                data.appendLittleEndian(param)
            }
        }

        try data.write(to: url)
        logger.log("Wrote cameras.bin: \(cameras.count) camera(s)")
    }

    // MARK: - images.bin

    /// Format: uint64 num_images, then per image:
    /// uint32 image_id, qw qx qy qz (double), tx ty tz (double), uint32 camera_id,
    /// name (null-terminated string), uint64 num_points2d,
    /// then per 2D point: double x, double y, uint64 point3d_id
    static func writeImages(_ images: [CameraPose], to url: URL) throws {
        var data = Data()
        data.appendLittleEndian(UInt64(images.count))

        for image in images {
            data.appendLittleEndian(image.id)

            // Quaternion: w, x, y, z
            data.appendLittleEndian(image.rotation.real)
            data.appendLittleEndian(image.rotation.imag.x)
            data.appendLittleEndian(image.rotation.imag.y)
            data.appendLittleEndian(image.rotation.imag.z)

            // Translation
            data.appendLittleEndian(image.translation.x)
            data.appendLittleEndian(image.translation.y)
            data.appendLittleEndian(image.translation.z)

            // Camera ID
            data.appendLittleEndian(image.cameraId)

            // Image name (null-terminated)
            if let nameData = image.imageName.data(using: .utf8) {
                data.append(nameData)
            }
            data.append(0) // null terminator

            // 2D points
            data.appendLittleEndian(UInt64(image.points2D.count))
            for point in image.points2D {
                data.appendLittleEndian(point.x)
                data.appendLittleEndian(point.y)
                data.appendLittleEndian(point.point3DId ?? UInt64.max)
            }
        }

        try data.write(to: url)
        logger.log("Wrote images.bin: \(images.count) image(s)")
    }

    // MARK: - points3D.bin

    /// Format: uint64 num_points, then per point:
    /// uint64 point3d_id, double x y z, uint8 r g b, double error,
    /// uint64 track_length, then per track: uint32 image_id, uint32 point2d_idx
    static func writePoints3D(_ points: [SparsePoint3D], to url: URL) throws {
        var data = Data()
        data.appendLittleEndian(UInt64(points.count))

        for point in points {
            data.appendLittleEndian(point.id)

            // Position
            data.appendLittleEndian(point.position.x)
            data.appendLittleEndian(point.position.y)
            data.appendLittleEndian(point.position.z)

            // Color
            data.append(point.color.x)
            data.append(point.color.y)
            data.append(point.color.z)

            // Reprojection error
            data.appendLittleEndian(point.reprojectionError)

            // Track
            data.appendLittleEndian(UInt64(point.track.count))
            for element in point.track {
                data.appendLittleEndian(element.imageId)
                data.appendLittleEndian(element.point2DIndex)
            }
        }

        try data.write(to: url)
        logger.log("Wrote points3D.bin: \(points.count) point(s)")
    }
}

// MARK: - Data Extension for Little-Endian Writing

extension Data {
    mutating func appendLittleEndian<T: FixedWidthInteger>(_ value: T) {
        var littleEndian = value.littleEndian
        append(UnsafeBufferPointer(start: &littleEndian, count: 1))
    }

    mutating func appendLittleEndian(_ value: Double) {
        var bits = value.bitPattern.littleEndian
        append(UnsafeBufferPointer(start: &bits, count: 1))
    }
}
