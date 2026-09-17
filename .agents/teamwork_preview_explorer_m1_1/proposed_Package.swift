// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "AsyncMoERouter",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .library(
            name: "AsyncMoERouter",
            targets: ["AsyncMoERouter"]
        ),
    ],
    dependencies: [
        // MLX-Swift can be optionally added for Milestone 5 (Recalibration):
        // .package(url: "https://github.com/ml-explore/mlx-swift.git", exact: "0.31.6")
    ],
    targets: [
        .target(
            name: "AsyncMoERouter",
            dependencies: [],
            path: "Sources/AsyncMoERouter"
        ),
        .testTarget(
            name: "AsyncMoERouterTests",
            dependencies: ["AsyncMoERouter"],
            path: "swift_tests/AsyncMoERouterTests"
        ),
    ]
)
