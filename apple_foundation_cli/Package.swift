// swift-tools-version:5.10
import PackageDescription

let package = Package(
    name: "jitskilled-apple-fm",
    platforms: [.macOS(.v26)],
    targets: [
        .executableTarget(
            name: "jitskilled-apple-fm",
            path: "Sources/jitskilled-apple-fm"
        )
    ]
)
