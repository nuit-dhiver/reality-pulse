import Foundation

/// The export record for a single watermarked file — the single source of
/// truth shared by the app (SwiftData persistence, JSON export) and the
/// internal `watermark-verify` CLI. Contains the per-copy secret key; record
/// JSON files must be handled as secrets.
public struct WatermarkRecord: Codable, Sendable, Equatable {
    public static let currentSchemaVersion = 1
    public static let currentAlgorithmVersion = 1

    public enum Channel {
        public static let geometry = "geometry"
        public static let texture = "texture"
    }

    public struct GeometryChannelInfo: Codable, Sendable, Equatable {
        public var parameters: GeometryWatermarkParameters
        public var effectiveBinCount: Int
        public var embeddedBits: Int

        public init(parameters: GeometryWatermarkParameters, effectiveBinCount: Int, embeddedBits: Int) {
            self.parameters = parameters
            self.effectiveBinCount = effectiveBinCount
            self.embeddedBits = embeddedBits
        }

        /// Embed-time parameters with the bin count the embedder actually used.
        public var detectionParameters: GeometryWatermarkParameters {
            var parameters = parameters
            parameters.binCount = effectiveBinCount
            return parameters
        }
    }

    public struct TextureChannelInfo: Codable, Sendable, Equatable {
        public struct Image: Codable, Sendable, Equatable {
            /// Archive entry name or sidecar filename of the stamped image.
            public var name: String
            /// Material semantic, e.g. "baseColor".
            public var semantic: String
            /// Pixel dimensions at embed time — the registration anchor the
            /// detector resamples a rescaled suspect image back to.
            public var width: Int
            public var height: Int

            public init(name: String, semantic: String, width: Int, height: Int) {
                self.name = name
                self.semantic = semantic
                self.width = width
                self.height = height
            }
        }

        public var parameters: TextureWatermarkParameters
        public var images: [Image]

        public init(parameters: TextureWatermarkParameters, images: [Image]) {
            self.parameters = parameters
            self.images = images
        }
    }

    public var schemaVersion: Int
    public var recordId: UUID
    public var jobId: UUID
    /// "usdz" | "gltf" | "glb" | "ply"
    public var format: String
    public var detailLevel: String
    public var filename: String
    /// Absolute path at export time; informational only.
    public var filePath: String?
    public var createdAt: Date
    public var algorithmVersion: Int
    /// The 32-byte per-copy secret key (base64 in JSON).
    public var key: Data
    /// Channels actually embedded in this file.
    public var channels: [String]
    public var geometry: GeometryChannelInfo?
    public var texture: TextureChannelInfo?
    /// SHA-256 of the final stamped file, for exact-copy short-circuit and
    /// stamp idempotence.
    public var fileSHA256: String

    public init(
        recordId: UUID = UUID(),
        jobId: UUID,
        format: String,
        detailLevel: String,
        filename: String,
        filePath: String?,
        createdAt: Date = Date(),
        key: WatermarkKey,
        channels: [String],
        geometry: GeometryChannelInfo?,
        texture: TextureChannelInfo?,
        fileSHA256: String
    ) {
        self.schemaVersion = Self.currentSchemaVersion
        self.recordId = recordId
        self.jobId = jobId
        self.format = format
        self.detailLevel = detailLevel
        self.filename = filename
        self.filePath = filePath
        self.createdAt = createdAt
        self.algorithmVersion = Self.currentAlgorithmVersion
        self.key = key.data
        self.channels = channels
        self.geometry = geometry
        self.texture = texture
        self.fileSHA256 = fileSHA256
    }

    public var watermarkKey: WatermarkKey {
        get throws { try WatermarkKey(data: key) }
    }
}

public extension WatermarkRecord {
    static func makeEncoder() -> JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        return encoder
    }

    static func makeDecoder() -> JSONDecoder {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }

    func jsonData() throws -> Data {
        try Self.makeEncoder().encode(self)
    }

    init(jsonData: Data) throws {
        self = try Self.makeDecoder().decode(WatermarkRecord.self, from: jsonData)
    }
}
