/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Stages Gaussian Splat training datasets from images and COLMAP outputs.
*/

import Foundation
import os

private let logger = Logger(
    subsystem: ObjectCaptureReconstructionApp.subsystem,
    category: "GaussianSplatDatasetPreparer"
)

struct GaussianSplatPreparedDataset {
    let datasetRoot: URL
    let exportsDirectory: URL
}

enum GaussianSplatDatasetPreparerError: LocalizedError {
    case noImagesFound
    case missingCOLMAPModel(URL)
    case missingCOLMAPFiles(URL)

    var errorDescription: String? {
        switch self {
        case .noImagesFound:
            return "No training images were found in the selected image folder."
        case .missingCOLMAPModel(let url):
            return "No COLMAP model was found under \(url.path())."
        case .missingCOLMAPFiles(let url):
            return "The COLMAP model at \(url.path()) is missing camera or image files."
        }
    }
}

struct GaussianSplatDatasetPreparer {

    func prepareDataset(
        imageFolder: URL,
        colmapDirectory: URL,
        workingDirectory: URL
    ) throws -> GaussianSplatPreparedDataset {
        let fileManager = FileManager.default
        let datasetRoot = workingDirectory.appending(path: "dataset")
        let exportsDirectory = workingDirectory.appending(path: "exports")
        let imagesDirectory = datasetRoot.appending(path: "images")
        let sparseDirectory = datasetRoot.appending(path: "sparse").appending(path: "0")

        if fileManager.fileExists(atPath: datasetRoot.path()) {
            try fileManager.removeItem(at: datasetRoot)
        }
        if fileManager.fileExists(atPath: exportsDirectory.path()) {
            try fileManager.removeItem(at: exportsDirectory)
        }

        try fileManager.createDirectory(at: imagesDirectory, withIntermediateDirectories: true)
        try fileManager.createDirectory(at: sparseDirectory, withIntermediateDirectories: true)
        try fileManager.createDirectory(at: exportsDirectory, withIntermediateDirectories: true)

        let imageFiles = try collectImageFiles(in: imageFolder)
        guard !imageFiles.isEmpty else {
            throw GaussianSplatDatasetPreparerError.noImagesFound
        }

        for imageURL in imageFiles {
            let relativePath = try relativePath(for: imageURL, under: imageFolder)
            let destinationURL = imagesDirectory.appending(path: relativePath)
            try fileManager.createDirectory(
                at: destinationURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )

            if fileManager.fileExists(atPath: destinationURL.path()) {
                try fileManager.removeItem(at: destinationURL)
            }

            do {
                try fileManager.createSymbolicLink(at: destinationURL, withDestinationURL: imageURL)
            } catch {
                try fileManager.copyItem(at: imageURL, to: destinationURL)
            }
        }

        let normalizedCOLMAPDirectory = try locateCOLMAPModelDirectory(in: colmapDirectory)
        let paths = try resolveCOLMAPPaths(in: normalizedCOLMAPDirectory)

        try fileManager.copyItem(at: paths.cameras, to: sparseDirectory.appending(path: paths.cameras.lastPathComponent))
        try fileManager.copyItem(at: paths.images, to: sparseDirectory.appending(path: paths.images.lastPathComponent))

        if let points = paths.points3D {
            try fileManager.copyItem(at: points, to: sparseDirectory.appending(path: points.lastPathComponent))
        }

        logger.log("Prepared Gaussian Splat dataset at \(datasetRoot.path())")
        return GaussianSplatPreparedDataset(datasetRoot: datasetRoot, exportsDirectory: exportsDirectory)
    }

    private func collectImageFiles(in root: URL) throws -> [URL] {
        let fileManager = FileManager.default
        guard let enumerator = fileManager.enumerator(
            at: root,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else {
            return []
        }

        var results: [URL] = []
        for case let fileURL as URL in enumerator {
            guard ImageHelper.validImageSuffixes.contains(fileURL.pathExtension.lowercased()) else {
                continue
            }
            results.append(fileURL)
        }

        return results.sorted { $0.path() < $1.path() }
    }

    private func relativePath(for fileURL: URL, under root: URL) throws -> String {
        let rootComponents = root.standardizedFileURL.pathComponents
        let fileComponents = fileURL.standardizedFileURL.pathComponents
        guard fileComponents.starts(with: rootComponents) else {
            throw CocoaError(.fileReadInvalidFileName)
        }
        let relativeComponents = Array(fileComponents.dropFirst(rootComponents.count))
        return NSString.path(withComponents: relativeComponents)
    }

    private func locateCOLMAPModelDirectory(in root: URL) throws -> URL {
        if hasCOLMAPCameraAndImageFiles(in: root) {
            return root
        }

        let fileManager = FileManager.default
        guard let enumerator = fileManager.enumerator(
            at: root,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ) else {
            throw GaussianSplatDatasetPreparerError.missingCOLMAPModel(root)
        }

        for case let directoryURL as URL in enumerator {
            if hasCOLMAPCameraAndImageFiles(in: directoryURL) {
                return directoryURL
            }
        }

        throw GaussianSplatDatasetPreparerError.missingCOLMAPModel(root)
    }

    private func hasCOLMAPCameraAndImageFiles(in directory: URL) -> Bool {
        do {
            let contents = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            let names = Set(contents.map { $0.lastPathComponent.lowercased() })
            return (names.contains("cameras.bin") || names.contains("cameras.txt"))
                && (names.contains("images.bin") || names.contains("images.txt"))
        } catch {
            return false
        }
    }

    private func resolveCOLMAPPaths(in directory: URL) throws -> (cameras: URL, images: URL, points3D: URL?) {
        let contents = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
        let byName = Dictionary(uniqueKeysWithValues: contents.map { ($0.lastPathComponent.lowercased(), $0) })

        let cameras = byName["cameras.bin"] ?? byName["cameras.txt"]
        let images = byName["images.bin"] ?? byName["images.txt"]
        let points = byName["points3d.bin"] ?? byName["points3d.txt"]

        guard let cameras, let images else {
            throw GaussianSplatDatasetPreparerError.missingCOLMAPFiles(directory)
        }

        return (cameras, images, points)
    }
}