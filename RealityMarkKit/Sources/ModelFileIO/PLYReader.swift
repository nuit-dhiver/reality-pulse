import Foundation
import simd

public enum PLYReaderError: Error {
    case invalidHeader
    case unsupportedFormat(String)
    case missingPositions
    case truncatedPayload
}

/// Generic binary little-endian PLY vertex reader — extracts x/y/z positions
/// from any property layout (superset of the 17-float layout Reality Pulse's
/// splat writer emits).
public enum PLYReader {
    public static func readPositions(url: URL) throws -> [SIMD3<Float>] {
        try readPositions(data: Data(contentsOf: url))
    }

    public static func readPositions(data: Data) throws -> [SIMD3<Float>] {
        guard let headerEndRange = data.range(of: Data("end_header\n".utf8)) else {
            throw PLYReaderError.invalidHeader
        }
        guard let header = String(data: data.subdata(in: 0..<headerEndRange.upperBound), encoding: .ascii),
              header.hasPrefix("ply") else {
            throw PLYReaderError.invalidHeader
        }

        var vertexCount = 0
        var inVertexElement = false
        var offset = 0
        var stride = 0
        var xOffset: Int?
        var yOffset: Int?
        var zOffset: Int?

        for line in header.split(separator: "\n") {
            let fields = line.split(separator: " ")
            guard let keyword = fields.first else { continue }
            switch keyword {
            case "format":
                guard fields.count >= 2, fields[1] == "binary_little_endian" else {
                    throw PLYReaderError.unsupportedFormat(String(line))
                }
            case "element":
                if fields.count >= 3, fields[1] == "vertex" {
                    inVertexElement = true
                    vertexCount = Int(fields[2]) ?? 0
                } else {
                    inVertexElement = false
                }
            case "property":
                guard inVertexElement, fields.count >= 3 else { continue }
                guard fields[1] != "list" else {
                    throw PLYReaderError.unsupportedFormat("list property in vertex element")
                }
                let size = try propertySize(String(fields[1]))
                let name = fields[2]
                if fields[1] == "float" || fields[1] == "float32" {
                    if name == "x" { xOffset = offset }
                    if name == "y" { yOffset = offset }
                    if name == "z" { zOffset = offset }
                }
                offset += size
                stride = offset
            default:
                continue
            }
        }

        guard let xOffset, let yOffset, let zOffset, vertexCount > 0, stride > 0 else {
            throw PLYReaderError.missingPositions
        }

        let payloadStart = headerEndRange.upperBound
        guard payloadStart + vertexCount * stride <= data.count else {
            throw PLYReaderError.truncatedPayload
        }

        var positions = [SIMD3<Float>]()
        positions.reserveCapacity(vertexCount)
        for vertex in 0..<vertexCount {
            let base = payloadStart + vertex * stride
            positions.append(SIMD3<Float>(
                readFloat(data, at: base + xOffset),
                readFloat(data, at: base + yOffset),
                readFloat(data, at: base + zOffset)
            ))
        }
        return positions
    }

    private static func propertySize(_ type: String) throws -> Int {
        switch type {
        case "char", "uchar", "int8", "uint8": return 1
        case "short", "ushort", "int16", "uint16": return 2
        case "int", "uint", "int32", "uint32", "float", "float32": return 4
        case "double", "float64": return 8
        default: throw PLYReaderError.unsupportedFormat(type)
        }
    }

    private static func readFloat(_ data: Data, at offset: Int) -> Float {
        var bits: UInt32 = 0
        withUnsafeMutableBytes(of: &bits) { destination in
            data.copyBytes(to: destination, from: offset..<(offset + 4))
        }
        return Float(bitPattern: UInt32(littleEndian: bits))
    }
}
