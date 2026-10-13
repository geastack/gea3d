#!/usr/bin/env bash
set -euo pipefail

# Vendors the gea3d library sources into an app as <app>/src/gea3d/.
#
# Why: the gea build pipeline (build-gea-vite-geatsc.mjs) compat-copies only
# the app directory before bundling, so package deps outside the alias table
# (like @geastack/gea3d) don't resolve there. Vendoring keeps the experiment
# fully contained — no core build-script edits. The copy is a generated
# artifact; gea3d/src is the only place to edit.
#
# Usage: bash gea3d/scripts/sync-into-app.sh examples/apps/gea3d-cube

GEA3D_DIR="$(cd "$(dirname "$0")/.." && pwd)"
APP_DIR="${1:?usage: sync-into-app.sh <app-dir>}"
APP_DIR="$(cd "$APP_DIR" && pwd)"
DEST="$APP_DIR/src/gea3d"

mkdir -p "$DEST"
rsync -a --delete "$GEA3D_DIR/src/" "$DEST/"
cat > "$DEST/VENDORED.md" <<EOF
Generated copy of gea3d/src — do not edit here.
Re-sync with: bash gea3d/scripts/sync-into-app.sh <this app dir>
EOF
echo "synced gea3d/src -> $DEST"
