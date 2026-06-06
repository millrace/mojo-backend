#!/usr/bin/env bash
#
# Build mojo-backend.zip — the engine source bundle the Millrace menu app
# downloads, then `mojo build`s on-device against a separately-fetched Mojo
# compiler (see millrace/app Bootstrapper). The bundle unzips to three siblings:
#
#   mojo-backend/   src + assets + tokenizer fixtures +
#                   build/{libflare_tls.so + libssl.3 + libcrypto.3, rpath-fixed}
#   minja2/src/     vendored minja2 (chat templating / JSON)
#   flare/flare/    vendored flare package (HTTP server)
#
# so the app can run:
#   (cd mojo-backend && mojo build src/server.mojo -I ../minja2/src -I ../flare -o build/server)
#
# We ship the prebuilt libflare_tls.so (building it needs clang + OpenSSL) and
# its OpenSSL dylibs, made relocatable via @loader_path so the server finds them
# at runtime with no pixi. Run via pixi (needs CONDA_PREFIX) AFTER `pixi run
# flare-tls`. Usage: scripts/package_engine.sh [out.zip]
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MINJA2="${MINJA2:-$ROOT/../minja2}"
FLARE="${FLARE:-$ROOT/../flare}"
OUT="${1:-$ROOT/mojo-backend.zip}"
PREFIX="${CONDA_PREFIX:?run via pixi — need CONDA_PREFIX for libflare_tls.so + OpenSSL}"
LIB="$PREFIX/lib/libflare_tls.so"
[[ -f "$LIB" ]] || { echo "error: $LIB missing — run 'pixi run flare-tls' first" >&2; exit 1; }

STAGE="$(mktemp -d)"; trap 'rm -rf "$STAGE"' EXIT
B="$STAGE/mojo-backend"

echo "==> staging mojo-backend source" >&2
mkdir -p "$B/build" "$B/tests/fixtures/tokenizer"
cp -R "$ROOT/src" "$B/src"
cp -R "$ROOT/assets" "$B/assets"
for f in vocab.tsv merges.tsv specials.tsv; do
    cp "$ROOT/tests/fixtures/tokenizer/$f" "$B/tests/fixtures/tokenizer/"
done

echo "==> bundling libflare_tls.so + OpenSSL (relocatable)" >&2
cp "$LIB" "$PREFIX/lib/libssl.3.dylib" "$PREFIX/lib/libcrypto.3.dylib" "$B/build/"
# OpenSSL dylibs are already @rpath-id'd with an @loader_path rpath, so copying
# is enough. Only libflare_tls.so needs fixing: find OpenSSL beside it
# (@loader_path) and take libc++ from the OS instead of the (unshipped) conda one.
install_name_tool -delete_rpath "$PREFIX/lib" "$B/build/libflare_tls.so" 2>/dev/null || true
install_name_tool \
    -id "@rpath/libflare_tls.so" \
    -add_rpath "@loader_path" \
    -change "@rpath/libc++.1.dylib" "/usr/lib/libc++.1.dylib" \
    "$B/build/libflare_tls.so"
codesign --force --sign - "$B/build/libflare_tls.so" 2>/dev/null || true

echo "==> staging minja2 + flare" >&2
mkdir -p "$STAGE/minja2" "$STAGE/flare"
cp -R "$MINJA2/src" "$STAGE/minja2/src"
cp -R "$FLARE/flare" "$STAGE/flare/flare"

echo "==> zipping -> $OUT" >&2
rm -f "$OUT"
( cd "$STAGE" && zip -qr -X "$OUT" mojo-backend minja2 flare )
echo "==> done" >&2
ls -lh "$OUT" >&2
