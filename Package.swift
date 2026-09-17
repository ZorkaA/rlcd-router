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
    dependencies: [],
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
