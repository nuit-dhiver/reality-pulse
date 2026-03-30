/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Data models for SfM results: camera poses, intrinsics, and sparse 3D points.
*/

import Foundation
import simd

// MARK: - Camera Intrinsics

/// COLMAP camera model identifiers.
enum COLMAPCameraModel: Int32, Codable, CaseIterable, Identifiable {
    case simplePinhole = 0
    case pinhole = 1
    case simpleRadial = 2
    case radial = 3
    case openCV = 4
    case fullOpenCV = 5
    case simpleRadialFisheye = 6
    case radialFisheye = 7
    case openCVFisheye = 8

    var id: Int32 { rawValue }

    var displayName: String {
        switch self {
        case .simplePinhole:      return "Simple Pinhole"
        case .pinhole:            return "Pinhole"
        case .simpleRadial:       return "Simple Radial"
        case .radial:             return "Radial"
        case .openCV:             return "OpenCV"
        case .fullOpenCV:         return "Full OpenCV"
        case .simpleRadialFisheye: return "Simple Radial Fisheye"
        case .radialFisheye:      return "Radial Fisheye"
        case .openCVFisheye:      return "OpenCV Fisheye"
        }
    }

    /// Number of intrinsic parameters for this camera model.
    var paramCount: Int {
        switch self {
        case .simplePinhole:       return 3  // f, cx, cy
        case .pinhole:             return 4  // fx, fy, cx, cy
        case .simpleRadial:        return 4  // f, cx, cy, k
        case .radial:              return 5  // f, cx, cy, k1, k2
        case .openCV:              return 8  // fx, fy, cx, cy, k1, k2, p1, p2
        case .fullOpenCV:          return 12 // fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6
        case .simpleRadialFisheye: return 4  // f, cx, cy, k
        case .radialFisheye:       return 5  // f, cx, cy, k1, k2
        case .openCVFisheye:       return 8  // fx, fy, cx, cy, k1, k2, k3, k4
        }
    }
}

/// Camera intrinsic parameters as stored in COLMAP.
struct CameraIntrinsics: Codable, Identifiable {
    let id: UInt32
    var model: COLMAPCameraModel
    var width: UInt64
    var height: UInt64
    var params: [Double]

    init(id: UInt32, model: COLMAPCameraModel, width: UInt64, height: UInt64, params: [Double]) {
        self.id = id
        self.model = model
        self.width = width
        self.height = height
        self.params = params
    }
}

// MARK: - Camera Pose (Extrinsics)

/// Camera pose for a single image: rotation (quaternion) + translation.
/// Follows COLMAP convention: world-to-camera transformation.
struct CameraPose: Codable, Identifiable {
    let id: UInt32
    var cameraId: UInt32
    var imageName: String

    /// Quaternion (w, x, y, z) representing the world-to-camera rotation.
    var rotation: simd_quatd

    /// Translation vector (world-to-camera).
    var translation: SIMD3<Double>

    /// 2D keypoints observed in this image with optional 3D point associations.
    var points2D: [Point2D]

    init(
        id: UInt32,
        cameraId: UInt32,
        imageName: String,
        rotation: simd_quatd,
        translation: SIMD3<Double>,
        points2D: [Point2D] = []
    ) {
        self.id = id
        self.cameraId = cameraId
        self.imageName = imageName
        self.rotation = rotation
        self.translation = translation
        self.points2D = points2D
    }

    /// The camera center in world coordinates: C = -R^T * t
    var cameraCenter: SIMD3<Double> {
        let rotMatrix = simd_matrix3x3(rotation)
        return -(rotMatrix.transpose * translation)
    }

    /// The 4x4 world-to-camera transformation matrix.
    var viewMatrix: simd_double4x4 {
        let r = simd_matrix3x3(rotation)
        return simd_double4x4(
            SIMD4(r.columns.0, 0),
            SIMD4(r.columns.1, 0),
            SIMD4(r.columns.2, 0),
            SIMD4(translation, 1)
        )
    }
}

/// A 2D observation in an image, optionally linked to a 3D point.
struct Point2D: Codable {
    var x: Double
    var y: Double
    /// ID of the corresponding 3D point, or `nil` if not triangulated.
    var point3DId: UInt64?
}

// MARK: - Sparse 3D Point

/// A triangulated 3D point from the sparse reconstruction.
struct SparsePoint3D: Codable, Identifiable {
    var id: UInt64
    var position: SIMD3<Double>
    var color: SIMD3<UInt8>
    var reprojectionError: Double
    var track: [TrackElement]
}

/// Links a 3D point to an observation in a specific image.
struct TrackElement: Codable {
    var imageId: UInt32
    var point2DIndex: UInt32
}

// MARK: - Aggregate Result

/// Complete SfM result: cameras, poses, and sparse point cloud.
struct SfMResult: Codable {
    var cameras: [CameraIntrinsics]
    var images: [CameraPose]
    var points3D: [SparsePoint3D]

    init(cameras: [CameraIntrinsics] = [], images: [CameraPose] = [], points3D: [SparsePoint3D] = []) {
        self.cameras = cameras
        self.images = images
        self.points3D = points3D
    }

    var cameraCount: Int { cameras.count }
    var imageCount: Int { images.count }
    var pointCount: Int { points3D.count }

    /// Look up a camera by ID.
    func camera(for id: UInt32) -> CameraIntrinsics? {
        cameras.first { $0.id == id }
    }

    /// Look up an image pose by name.
    func pose(forImage name: String) -> CameraPose? {
        images.first { $0.imageName == name }
    }
}

// MARK: - SIMD Codable Conformance

extension simd_quatd: @retroactive Codable {
    enum CodingKeys: String, CodingKey {
        case w, x, y, z
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let w = try container.decode(Double.self, forKey: .w)
        let x = try container.decode(Double.self, forKey: .x)
        let y = try container.decode(Double.self, forKey: .y)
        let z = try container.decode(Double.self, forKey: .z)
        self.init(ix: x, iy: y, iz: z, r: w)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(real, forKey: .w)
        try container.encode(imag.x, forKey: .x)
        try container.encode(imag.y, forKey: .y)
        try container.encode(imag.z, forKey: .z)
    }
}

extension SIMD3: @retroactive Codable where Scalar: Codable {
    enum CodingKeys: String, CodingKey {
        case x, y, z
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let x = try container.decode(Scalar.self, forKey: .x)
        let y = try container.decode(Scalar.self, forKey: .y)
        let z = try container.decode(Scalar.self, forKey: .z)
        self.init(x, y, z)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(x, forKey: .x)
        try container.encode(y, forKey: .y)
        try container.encode(z, forKey: .z)
    }
}
