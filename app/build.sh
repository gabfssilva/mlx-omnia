#!/bin/bash
# Build the panel and lay it out as an .app.
#
# SwiftPM produces an executable and nothing else; a menu bar app needs a bundle, because
# `LSUIElement` is a key in one — that is what keeps the panel out of the Dock and out of
# ⌘-Tab. Signing is ad-hoc and has to happen after the plist, since the signature covers
# it: change one and the bundle will not launch.
set -euo pipefail

cd "$(dirname "$0")"
configuration="${1:-release}"

swift build -c "$configuration"
bin="$(swift build -c "$configuration" --show-bin-path)"

app="build/MLX Omnia.app"
rm -rf "$app"
mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources"
cp "$bin/OmniaPanel" "$app/Contents/MacOS/OmniaPanel"
# A dependency that declares resources is built into a bundle of its own, next to the
# executable — which is where `Bundle.module` looks when the binary is run out of `.build`
# and nowhere near it once the binary has been copied into an app. Textual's is Prism, and
# without it every fence is drawn as one colour.
for held in "$bin"/*.bundle; do
    [ -e "$held" ] && cp -R "$held" "$app/Contents/Resources/"
done
cp Info.plist "$app/Contents/Info.plist"
codesign --force --sign - "$app" >/dev/null

echo "$app"
