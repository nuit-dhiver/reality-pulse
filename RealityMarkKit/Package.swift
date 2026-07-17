// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "RealityMarkKit",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "WatermarkCore", targets: ["WatermarkCore"]),
        .library(name: "ModelFileIO", targets: ["ModelFileIO"]),
        .executable(name: "watermark-verify", targets: ["watermark-verify"]),
    ],
    targets: [
        .target(name: "WatermarkCore"),
        .target(name: "ModelFileIO"),
        .executableTarget(
            name: "watermark-verify",
            dependencies: ["WatermarkCore", "ModelFileIO"]
        ),
        .testTarget(name: "WatermarkCoreTests", dependencies: ["WatermarkCore"]),
        .testTarget(name: "ModelFileIOTests", dependencies: ["ModelFileIO", "WatermarkCore"]),
    ]
)
