import CryptoKit
import Foundation
import WatermarkCore

// watermark-verify — internal provenance checker for Reality Pulse exports.
//
//   watermark-verify --record <record.json> [--channel geometry|texture|both]
//                    [--verbose] <suspect-file>
//
// Exit codes: 0 MATCH, 1 LIKELY, 2 NO MATCH, 3 usage/IO error.
//
// The algorithm is public; detection is possible only with the per-copy key
// inside the record JSON. Treat record files as secrets.

let usage = """
usage: watermark-verify --record <record.json> [--channel geometry|texture|both] [--verbose] <suspect-file>
"""

var recordPath: String?
var channelFilter = "both"
var verbose = false
var suspectPath: String?

var arguments = Array(CommandLine.arguments.dropFirst())
while !arguments.isEmpty {
    let argument = arguments.removeFirst()
    switch argument {
    case "--record", "-r":
        guard !arguments.isEmpty else { fail("--record requires a path") }
        recordPath = arguments.removeFirst()
    case "--channel", "-c":
        guard !arguments.isEmpty else { fail("--channel requires geometry|texture|both") }
        channelFilter = arguments.removeFirst()
        guard ["geometry", "texture", "both"].contains(channelFilter) else {
            fail("unknown channel '\(channelFilter)'")
        }
    case "--verbose", "-v":
        verbose = true
    case "--help", "-h":
        print(usage)
        exit(0)
    default:
        if argument.hasPrefix("-") { fail("unknown option '\(argument)'") }
        guard suspectPath == nil else { fail("multiple suspect files given") }
        suspectPath = argument
    }
}

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data("watermark-verify: \(message)\n\(usage)\n".utf8))
    exit(3)
}

guard let recordPath, let suspectPath else { fail("missing --record or suspect file") }

let record: WatermarkRecord
do {
    record = try WatermarkRecord(jsonData: Data(contentsOf: URL(fileURLWithPath: recordPath)))
} catch {
    fail("could not read record: \(error)")
}

let key: WatermarkKey
do {
    key = try record.watermarkKey
} catch {
    fail("record contains an invalid key")
}

let suspectURL = URL(fileURLWithPath: suspectPath)
let suspect: Suspect
do {
    suspect = try SuspectLoader.load(url: suspectURL)
} catch {
    fail("could not load suspect: \(error)")
}

print("watermark-verify — provenance check")
print("record:   \(record.recordId) (\(record.format), \(record.detailLevel), \(record.filename))")
if let keyLabel = record.keyLabel {
    print("key:      shared key '\(keyLabel)' — identifies the key, not an individual copy")
} else {
    print("key:      per-copy key — identifies this exact exported file")
}
print("suspect:  \(suspectPath)")

// Exact-copy short-circuit.
if let suspectData = try? Data(contentsOf: suspectURL, options: .mappedIfSafe) {
    let sha = SHA256.hash(data: suspectData).map { String(format: "%02x", $0) }.joined()
    if sha == record.fileSHA256 {
        print("sha256:   EXACT byte-identical copy of the recorded export")
        print("verdict:  MATCH (p = 0, exact copy)")
        exit(0)
    }
    print("sha256:   differs from recorded export (expected for edited copies)")
}

var channelPValues: [Double] = []

// Geometry channel.
if channelFilter != "texture" {
    if let geometry = record.geometry, !suspect.positions.isEmpty {
        let detection = GeometryWatermarker.detect(
            positions: suspect.positions,
            key: key,
            parameters: geometry.detectionParameters
        )
        channelPValues.append(detection.pValue)
        print(String(
            format: "geometry: matched %d/%d bits over %d points, p = %.3g",
            detection.matchedBits, detection.totalBits, suspect.positions.count, detection.pValue
        ))
    } else if record.geometry == nil {
        print("geometry: not embedded in this export (per record)")
    } else {
        print("geometry: suspect has no extractable point set")
    }
}

// Texture channel: try every suspect image against every recorded image size,
// keep the best score, and Bonferroni-adjust for the number of attempts.
if channelFilter != "geometry" {
    if let texture = record.texture, !suspect.images.isEmpty {
        var best: (name: String, result: TextureDetectionResult)?
        var attempts = 0
        for (name, image) in suspect.images {
            for recorded in texture.images {
                attempts += 1
                let result = TextureWatermarker.detect(
                    image: image,
                    key: key,
                    parameters: texture.parameters,
                    originalSize: (width: recorded.width, height: recorded.height)
                )
                if verbose {
                    print(String(
                        format: "  - %@ vs %@ (%dx%d): z = %.2f, p = %.3g",
                        name, recorded.name, recorded.width, recorded.height,
                        result.zScore, result.pValue
                    ))
                }
                if best == nil || result.zScore > best!.result.zScore {
                    best = (name, result)
                }
            }
        }
        if let best {
            let adjusted = min(1, best.result.pValue * Double(attempts))
            channelPValues.append(adjusted)
            print(String(
                format: "texture:  best image '%@': z = %.2f, p = %.3g (adjusted %.3g over %d attempt(s))",
                best.name, best.result.zScore, best.result.pValue, adjusted, attempts
            ))
        }
    } else if record.texture == nil {
        print("texture:  not embedded in this export (per record)")
    } else {
        print("texture:  suspect has no extractable images")
    }
}

guard !channelPValues.isEmpty else {
    fail("no channel of this record is testable against this suspect")
}

// Combined verdict: best channel, Bonferroni-adjusted for channels tested.
let combined = min(1, channelPValues.min()! * Double(channelPValues.count))
let verdict: String
let exitCode: Int32
switch combined {
case ..<1e-6:
    verdict = "MATCH"
    exitCode = 0
case ..<1e-3:
    verdict = "LIKELY"
    exitCode = 1
default:
    verdict = "NO MATCH"
    exitCode = 2
}
print(String(format: "verdict:  %@ (combined p = %.3g)", verdict, combined))
exit(exitCode)
