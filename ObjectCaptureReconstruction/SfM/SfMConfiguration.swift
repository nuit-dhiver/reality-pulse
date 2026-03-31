/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Configuration options for Structure from Motion processing via COLMAP.
*/

import Foundation

/// Quality presets that map to specific COLMAP parameter combinations.
enum SfMQualityPreset: String, Codable, CaseIterable, Identifiable {
    case quick
    case balanced
    case detailed

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .quick:    return "Quick"
        case .balanced: return "Balanced"
        case .detailed: return "Detailed"
        }
    }

    var maxNumFeatures: Int {
        switch self {
        case .quick:    return 4096
        case .balanced: return 8192
        case .detailed: return 16384
        }
    }

    var description: String {
        switch self {
        case .quick:    return "Fast processing, fewer features. Good for quick previews."
        case .balanced: return "Balanced speed and quality. Recommended for most use cases."
        case .detailed: return "Maximum features and accuracy. Slower but more complete."
        }
    }
}

/// Feature matching strategy.
enum SfMMatcherType: String, Codable, CaseIterable, Identifiable {
    case exhaustive
    case sequential

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .exhaustive: return "Exhaustive"
        case .sequential: return "Sequential"
        }
    }

    var description: String {
        switch self {
        case .exhaustive: return "Match every image pair. Best for unordered collections."
        case .sequential: return "Match nearby images. Faster for video sequences."
        }
    }
}

/// Full SfM configuration for a COLMAP run.
struct SfMConfiguration: Codable, Equatable {

    /// Camera model to use for intrinsic estimation.
    var cameraModel: COLMAPCameraModel = .openCV

    /// Whether all images share the same camera intrinsics.
    var sharedIntrinsics: Bool = true

    /// Feature matching strategy.
    var matcherType: SfMMatcherType = .exhaustive

    /// Quality preset controlling feature count and match quality.
    var qualityPreset: SfMQualityPreset = .balanced

    /// Maximum number of SIFT features per image (overrides preset when > 0).
    var maxNumFeaturesOverride: Int = 0

    /// Effective max features, considering preset and override.
    var effectiveMaxFeatures: Int {
        maxNumFeaturesOverride > 0 ? maxNumFeaturesOverride : qualityPreset.maxNumFeatures
    }

    /// Whether to use GPU for feature extraction (if available).
    var useGPU: Bool = true

    // MARK: - COLMAP Command Arguments

    /// Arguments for `colmap feature_extractor`.
    func featureExtractorArgs(databasePath: String, imagePath: String) -> [String] {
        var args = [
            "feature_extractor",
            "--database_path", databasePath,
            "--image_path", imagePath,
            "--ImageReader.camera_model", cameraModelString,
            "--SiftExtraction.max_num_features", "\(effectiveMaxFeatures)",
            // Bundled COLMAP is built without CUDA; always disable GPU.
            "--FeatureExtraction.use_gpu", "0"
        ]

        if sharedIntrinsics {
            args += ["--ImageReader.single_camera", "1"]
        }

        return args
    }

    /// Arguments for `colmap exhaustive_matcher` or `sequential_matcher`.
    func matcherArgs(databasePath: String) -> [String] {
        [
            matcherType == .exhaustive ? "exhaustive_matcher" : "sequential_matcher",
            "--database_path", databasePath,
            // Bundled COLMAP is built without CUDA; always disable GPU.
            "--FeatureMatching.use_gpu", "0"
        ]
    }

    /// Arguments for `colmap mapper`.
    func mapperArgs(databasePath: String, imagePath: String, outputPath: String) -> [String] {
        [
            "mapper",
            "--database_path", databasePath,
            "--image_path", imagePath,
            "--output_path", outputPath
        ]
    }

    // MARK: - Private

    private var cameraModelString: String {
        switch cameraModel {
        case .simplePinhole:       return "SIMPLE_PINHOLE"
        case .pinhole:             return "PINHOLE"
        case .simpleRadial:        return "SIMPLE_RADIAL"
        case .radial:              return "RADIAL"
        case .openCV:              return "OPENCV"
        case .fullOpenCV:          return "FULL_OPENCV"
        case .simpleRadialFisheye: return "SIMPLE_RADIAL_FISHEYE"
        case .radialFisheye:       return "RADIAL_FISHEYE"
        case .openCVFisheye:       return "OPENCV_FISHEYE"
        }
    }
}

/// Codable wrapper for persistence in job queue.
struct CodableSfMConfiguration: Codable, Equatable {
    var cameraModel: COLMAPCameraModel = .openCV
    var sharedIntrinsics: Bool = true
    var matcherType: SfMMatcherType = .exhaustive
    var qualityPreset: SfMQualityPreset = .balanced
    var maxNumFeaturesOverride: Int = 0
    var useGPU: Bool = true

    init() {}

    init(from config: SfMConfiguration) {
        cameraModel = config.cameraModel
        sharedIntrinsics = config.sharedIntrinsics
        matcherType = config.matcherType
        qualityPreset = config.qualityPreset
        maxNumFeaturesOverride = config.maxNumFeaturesOverride
        useGPU = config.useGPU
    }

    func toSfMConfiguration() -> SfMConfiguration {
        var config = SfMConfiguration()
        config.cameraModel = cameraModel
        config.sharedIntrinsics = sharedIntrinsics
        config.matcherType = matcherType
        config.qualityPreset = qualityPreset
        config.maxNumFeaturesOverride = maxNumFeaturesOverride
        config.useGPU = useGPU
        return config
    }
}
