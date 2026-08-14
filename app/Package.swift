// swift-tools-version: 6.2
import PackageDescription

// The panel is the one part of Omnia that is not Python, and it is not Python for one
// reason: `MenuBarExtra(.window)` is the shape this app wants and macOS gives it away —
// anchoring, dismissal, the notch, every display, Stage Manager. Nothing here imports the
// engine; the daemon is reached over HTTP, which is the same contract `lint-imports`
// holds the window and the CLI to.
//
// Textual is the one dependency, and it is here for the transcript: a model writes markdown
// and the panel was drawing it with a block splitter of its own, which never had syntax
// highlighting and never will have nested lists. It renders through SwiftUI's own `Text`
// over Foundation's `AttributedString` parser — the same parser the splitter already called
// for the inline half, so what changes is the block half and the code fences.
let package = Package(
    name: "OmniaPanel",
    platforms: [.macOS(.v26)],
    dependencies: [
        .package(url: "https://github.com/gonzalezreal/textual", from: "0.4.0"),
        // Sparkle is the updater, and it is a dependency of the panel rather than something
        // laid into the bundle afterwards: it used to be fetched as a tarball and opened
        // through ctypes, because the app it updated was Python. Here it is a framework a
        // Swift target links, which is the shape it was written for. `build.sh` copies it
        // into Contents/Frameworks and the executable carries the rpath to find it there.
        .package(url: "https://github.com/sparkle-project/Sparkle", from: "2.9.0"),
    ],
    targets: [
        .executableTarget(
            name: "OmniaPanel",
            dependencies: [
                .product(name: "Textual", package: "textual"),
                .product(name: "Sparkle", package: "Sparkle"),
            ],
            path: "Sources/OmniaPanel",
            resources: [.process("Resources")],
            // The tools version is 6.2 for `.macOS(.v26)` and for nothing else. Swift 6 is
            // the default language mode at that version, and adopting it is a concurrency
            // migration across the whole panel — a separate change from what a tools version
            // was bumped to reach.
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .testTarget(
            name: "OmniaPanelTests",
            dependencies: ["OmniaPanel"]
        ),
    ]
)
