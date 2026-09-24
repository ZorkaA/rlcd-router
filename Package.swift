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
        .executable(name: "BenchmarkE2E", targets: ["BenchmarkE2E"])
    ],
    dependencies: [],
    targets: [
        .target(
            name: "AsyncMoERouter",
            dependencies: [],
            path: "Sources/AsyncMoERouter"
        ),
        .executableTarget(
            name: "BenchmarkE2E",
            dependencies: ["AsyncMoERouter"],
            path: "Sources/BenchmarkE2E"
        ),
        .testTarget(
            name: "AsyncMoERouterTests",
            dependencies: ["AsyncMoERouter"],
            path: "swift_tests/AsyncMoERouterTests"
        ),
    ]
)
