// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "slurper",
    platforms: [.macOS("26.0")],
    targets: [
        .target(name: "SlurperKit"),
        .executableTarget(name: "slurper", dependencies: ["SlurperKit"]),
        .testTarget(name: "SlurperKitTests", dependencies: ["SlurperKit"]),
    ]
)
