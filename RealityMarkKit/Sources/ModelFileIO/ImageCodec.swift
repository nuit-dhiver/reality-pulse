import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

public enum ImageCodecError: Error {
    case decodeFailed
    case encodeFailed
}

/// CGImage ⇄ PNG/JPEG via ImageIO.
public enum ImageCodec {
    public static func decode(_ data: Data) throws -> CGImage {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw ImageCodecError.decodeFailed
        }
        return image
    }

    public static func pngData(from image: CGImage) throws -> Data {
        try encode(image, type: UTType.png, options: nil)
    }

    public static func jpegData(from image: CGImage, quality: Double) throws -> Data {
        let options = [kCGImageDestinationLossyCompressionQuality: quality] as CFDictionary
        return try encode(image, type: UTType.jpeg, options: options)
    }

    /// Detects PNG vs JPEG payloads by magic bytes; nil for anything else.
    public static func imageFormat(of data: Data) -> ImageFormat? {
        if data.starts(with: [0x89, 0x50, 0x4E, 0x47]) { return .png }
        if data.starts(with: [0xFF, 0xD8, 0xFF]) { return .jpeg }
        return nil
    }

    public enum ImageFormat {
        case png
        case jpeg
    }

    private static func encode(_ image: CGImage, type: UTType, options: CFDictionary?) throws -> Data {
        let data = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(
            data as CFMutableData, type.identifier as CFString, 1, nil
        ) else {
            throw ImageCodecError.encodeFailed
        }
        CGImageDestinationAddImage(destination, image, options)
        guard CGImageDestinationFinalize(destination) else {
            throw ImageCodecError.encodeFailed
        }
        return data as Data
    }
}
