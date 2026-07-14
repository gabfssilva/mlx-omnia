// swift-tools-version: 6.3

import PackageDescription

let package = Package(
    name: "Sideros",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [
        .library(name: "Sideros", targets: ["Sideros"]),
        .library(name: "SiderosServer", targets: ["SiderosServer"]),
        .executable(name: "sideros-bench", targets: ["sideros-bench"]),
        .executable(name: "sideros-serve", targets: ["sideros-serve"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.31.6"),
        .package(url: "https://github.com/hummingbird-project/hummingbird", from: "2.25.0"),
        .package(url: "https://github.com/johnmai-dev/Jinja", from: "2.4.0"),
    ],
    targets: [
        .target(
            name: "Sideros",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ]
        ),
        .target(
            name: "SiderosServer",
            dependencies: [
                "Sideros",
                .product(name: "Hummingbird", package: "hummingbird"),
                .product(name: "Jinja", package: "Jinja"),
            ]
        ),
        .executableTarget(
            name: "sideros-bench",
            dependencies: ["Sideros"]
        ),
        .executableTarget(
            name: "sideros-serve",
            dependencies: ["SiderosServer"]
        ),
        .testTarget(
            name: "SiderosTests",
            dependencies: ["Sideros", "SiderosServer"],
            resources: [.copy("Fixtures")]
        ),
    ]
)
