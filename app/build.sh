#!/bin/bash
# Build the panel and lay it out as an .app.
#
# SwiftPM produces an executable and nothing else; a menu bar app needs a bundle, because
# `LSUIElement` is a key in one — that is what keeps the panel out of the Dock and out of
# ⌘-Tab. Signing is ad-hoc and has to happen last, since the signature covers the plist and
# everything laid in beside it: change one afterwards and the bundle will not launch.
#
# What this does not do is the shipping half — a real identity, the engine's interpreter,
# the disk image. That is `mise run dmg`, which builds on top of this.
set -euo pipefail

cd "$(dirname "$0")"
configuration="${1:-release}"

# The rpath is the bundle's layout and not the package's, which is why it is passed here and
# not declared in Package.swift: SwiftPM links Sparkle from its artifact directory, and the
# copy that ships is the one under Contents/Frameworks.
swift build -c "$configuration" -Xlinker -rpath -Xlinker @executable_path/../Frameworks
bin="$(swift build -c "$configuration" --show-bin-path)"

app="build/MLX Omnia.app"
rm -rf "$app"
mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources" "$app/Contents/Frameworks"
cp "$bin/OmniaPanel" "$app/Contents/MacOS/OmniaPanel"
# A dependency that declares resources is built into a bundle of its own, next to the
# executable — which is where `Bundle.module` looks when the binary is run out of `.build`
# and nowhere near it once the binary has been copied into an app. Textual's is Prism, and
# without it every fence is drawn as one colour.
for held in "$bin"/*.bundle; do
    [ -e "$held" ] && cp -R "$held" "$app/Contents/Resources/"
done

# ditto and not cp -R: a versioned framework is symlinks into Versions/Current, and a copy
# that resolves them is a directory tree that no longer signs as one bundle. The slice is
# globbed rather than spelled, so a Sparkle release that changes which architectures it
# ships fails here instead of shipping an app with no updater in it.
sparkle=$(echo .build/artifacts/sparkle/Sparkle/Sparkle.xcframework/macos-*/Sparkle.framework)
if [ ! -d "$sparkle" ]; then
    echo "no Sparkle.framework under .build/artifacts — run swift package resolve" >&2
    exit 1
fi
ditto "$sparkle" "$app/Contents/Frameworks/Sparkle.framework"

cp Info.plist "$app/Contents/Info.plist"
# The tag is what a reader sees; the commit count is what Sparkle compares, and it moves
# before the first tag exists. Written into the copy, so the checked-in plist stays the
# checkout's own answer.
version=$(git describe --tags --abbrev=0 2>/dev/null || echo v0.0.0)
plist="$app/Contents/Info.plist"
/usr/libexec/PlistBuddy -c "Set :CFBundleShortVersionString ${version#v}" "$plist"
/usr/libexec/PlistBuddy -c "Set :CFBundleVersion $(git rev-list --count HEAD)" "$plist"

codesign --force --sign - "$app" >/dev/null

echo "$app"
