#!/usr/bin/env bash
#
# bundle-colmap.sh
#
# Creates a self-contained COLMAP bundle by copying the Homebrew-installed
# COLMAP binary and ALL of its dylib dependencies, then rewriting every
# absolute /opt/homebrew path to @loader_path-relative references using
# install_name_tool. The result can run without Homebrew installed.
#
# Usage:
#   ./scripts/bundle-colmap.sh [output_dir]
#
# Default output: ObjectCaptureReconstruction/Resources/colmap-bundle/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OUTPUT_DIR="${1:-$REPO_ROOT/ObjectCaptureReconstruction/Resources/colmap-bundle}"
BIN_DIR="$OUTPUT_DIR/bin"
LIB_DIR="$OUTPUT_DIR/lib"

COLMAP_BIN="$(command -v colmap 2>/dev/null || echo "/opt/homebrew/bin/colmap")"
if [[ ! -x "$COLMAP_BIN" ]]; then
    echo "ERROR: COLMAP not found. Install with: brew install colmap"
    exit 1
fi

COLMAP_REAL="$(python3 -c "import os; print(os.path.realpath('$COLMAP_BIN'))")"
echo "==> COLMAP binary: $COLMAP_REAL"
echo "==> Output directory: $OUTPUT_DIR"

# ─── Step 1: Clean and create output structure ───────────────────────────────

rm -rf "$OUTPUT_DIR"
mkdir -p "$BIN_DIR" "$LIB_DIR"

# ─── Step 2: Copy the COLMAP binary ─────────────────────────────────────────

cp "$COLMAP_REAL" "$BIN_DIR/colmap"
chmod 755 "$BIN_DIR/colmap"
echo "==> Copied colmap binary ($(du -h "$BIN_DIR/colmap" | awk '{print $1}'))"

# ─── Step 3: Trace all runtime dylib dependencies ───────────────────────────

echo "==> Tracing runtime dependencies..."

DYLD_PRINT_LIBRARIES=1 "$COLMAP_BIN" help 2>&1 \
    | grep "^dyld\[" \
    | sed 's/.*> //' \
    | sort -u \
    | grep "^/opt/homebrew" \
    > /tmp/colmap_deps_list.txt

DYLIB_COUNT=$(wc -l < /tmp/colmap_deps_list.txt | tr -d ' ')
echo "==> Found $DYLIB_COUNT Homebrew dylibs loaded at runtime"

# Also add COLMAP's own dylibs from the Cellar
COLMAP_LIB_DIR="$(dirname "$COLMAP_REAL")/../lib"
if [[ -d "$COLMAP_LIB_DIR" ]]; then
    for dylib in "$COLMAP_LIB_DIR"/libcolmap_*.dylib; do
        [[ -f "$dylib" ]] && echo "$dylib" >> /tmp/colmap_deps_list.txt
    done
    sort -u -o /tmp/colmap_deps_list.txt /tmp/colmap_deps_list.txt
fi

# ─── Step 4: Copy all dylibs with flat naming ───────────────────────────────

echo "==> Copying dylibs to lib/..."

# Python script handles the copy + creates a mapping file for path rewriting
python3 - "$LIB_DIR" << 'PYEOF'
import os, shutil, json, sys

deps_file = "/tmp/colmap_deps_list.txt"
lib_dir = sys.argv[1]
mapping_file = "/tmp/colmap_path_mapping.json"

with open(deps_file) as f:
    deps = [l.strip() for l in f if l.strip()]

# Map: original_path → flat_name_in_lib
mapping = {}
used_names = set()

for dep in deps:
    real_path = os.path.realpath(dep)
    if not os.path.exists(real_path):
        print(f"  SKIP: {dep} (not found)")
        continue

    # Determine flat name
    name = os.path.basename(real_path)

    # For framework dylibs (e.g., QtCore.framework/Versions/A/QtCore)
    if ".framework/" in dep:
        parts = dep.split("/")
        for i, part in enumerate(parts):
            if part.endswith(".framework"):
                name = part.replace(".framework", "")
                break

    # Deduplicate
    if name in used_names:
        # Map this path to the existing name
        mapping[dep] = name
        continue

    used_names.add(name)
    dest = os.path.join(lib_dir, name)

    if not os.path.exists(dest):
        shutil.copy2(real_path, dest)
        os.chmod(dest, 0o755)

    mapping[dep] = name

    # Also map the /opt/homebrew/opt/... symlink path if different
    # (otool shows the opt path, DYLD shows the Cellar path)
    # We need both in the mapping
    if "/Cellar/" in dep:
        # Try to find the /opt/ equivalent
        parts = dep.split("/Cellar/")
        if len(parts) == 2:
            pkg_parts = parts[1].split("/")
            if len(pkg_parts) >= 3:
                pkg_name = pkg_parts[0]
                remainder = "/".join(pkg_parts[2:])
                opt_path = f"/opt/homebrew/opt/{pkg_name}/{remainder}"
                mapping[opt_path] = name

# Also create @rpath mappings for colmap's own libs
for dep, name in list(mapping.items()):
    if "libcolmap_" in name:
        mapping[f"@rpath/{os.path.basename(dep)}"] = name
        mapping[f"@rpath/{name}"] = name

# Save mapping
with open(mapping_file, "w") as f:
    json.dump(mapping, f, indent=2)

print(f"  Copied {len(used_names)} unique dylibs")
PYEOF

COPIED_COUNT=$(ls "$LIB_DIR" | wc -l | tr -d ' ')
TOTAL_SIZE=$(du -sh "$LIB_DIR" | awk '{print $1}')
echo "==> $COPIED_COUNT dylibs in lib/ ($TOTAL_SIZE)"

# ─── Step 4.5: Create versioned symlinks ────────────────────────────────────
# Some dylibs reference shortened version names (e.g. libfoo.3.dylib) while
# we copied the full-version file (e.g. libfoo.3.4.5.dylib). Create symlinks.

echo "==> Creating versioned symlinks..."

python3 - "$LIB_DIR" << 'PYEOF'
import os, subprocess, sys

lib_dir = sys.argv[1]
available = set(os.listdir(lib_dir))
created = 0

# Scan all files in lib/ for @rpath and @loader_path refs to names not in lib/
for f in os.listdir(lib_dir):
    fpath = os.path.join(lib_dir, f)
    if not os.path.isfile(fpath):
        continue
    out = subprocess.run(["otool", "-L", fpath], capture_output=True, text=True).stdout
    for line in out.strip().split("\n")[1:]:
        ref = line.strip().split(" (")[0]
        name = ref.split("/")[-1]
        if not name.endswith(".dylib") or name in available:
            continue
        # Try finding the real file via Homebrew
        for search_base in ["/opt/homebrew/lib", "/opt/homebrew/opt/icu4c@78/lib"]:
            candidate = os.path.join(search_base, name)
            if os.path.exists(candidate):
                real = os.path.realpath(candidate)
                realbase = os.path.basename(real)
                if realbase in available:
                    link_path = os.path.join(lib_dir, name)
                    if not os.path.exists(link_path):
                        os.symlink(realbase, link_path)
                        available.add(name)
                        created += 1
                break

# Also scan the binary
binary_path = os.path.join(os.path.dirname(lib_dir), "bin", "colmap")
out = subprocess.run(["otool", "-L", binary_path], capture_output=True, text=True).stdout
for line in out.strip().split("\n")[1:]:
    ref = line.strip().split(" (")[0]
    name = ref.split("/")[-1]
    if not name.endswith(".dylib") or name in available:
        continue
    for search_base in ["/opt/homebrew/lib", "/opt/homebrew/opt/icu4c@78/lib"]:
        candidate = os.path.join(search_base, name)
        if os.path.exists(candidate):
            real = os.path.realpath(candidate)
            realbase = os.path.basename(real)
            if realbase in available:
                link_path = os.path.join(lib_dir, name)
                if not os.path.exists(link_path):
                    os.symlink(realbase, link_path)
                    available.add(name)
                    created += 1
            break

print(f"  Created {created} versioned symlinks")
PYEOF

# ─── Step 5: Rewrite paths using install_name_tool ──────────────────────────

echo "==> Rewriting dylib paths with install_name_tool..."

# Python script drives install_name_tool for all files
python3 - "$LIB_DIR" "$BIN_DIR" << 'PYEOF'
import os, json, subprocess, sys

lib_dir = sys.argv[1]
bin_dir = sys.argv[2]
mapping_file = "/tmp/colmap_path_mapping.json"

with open(mapping_file) as f:
    mapping = json.load(f)

def get_deps(path):
    """Get linked library paths from otool -L"""
    try:
        out = subprocess.check_output(["otool", "-L", path], stderr=subprocess.DEVNULL, text=True)
        deps = []
        for line in out.strip().split("\n")[1:]:
            dep = line.strip().split(" (")[0].strip()
            if dep:
                deps.append(dep)
        return deps
    except:
        return []

def rewrite(target, is_binary):
    """Rewrite all Homebrew/rpath references in target to @loader_path"""
    deps = get_deps(target)
    changes = 0

    for dep in deps:
        new_name = None

        # Direct match in mapping
        if dep in mapping:
            new_name = mapping[dep]
        # Try resolving symlinks
        elif dep.startswith("/opt/homebrew"):
            real = os.path.realpath(dep)
            basename = os.path.basename(real)
            if os.path.exists(os.path.join(lib_dir, basename)):
                new_name = basename
            # Try framework name
            if new_name is None and ".framework/" in dep:
                for part in dep.split("/"):
                    if part.endswith(".framework"):
                        fw_name = part.replace(".framework", "")
                        if os.path.exists(os.path.join(lib_dir, fw_name)):
                            new_name = fw_name
                            break
        elif dep.startswith("@rpath/"):
            basename = os.path.basename(dep)
            if os.path.exists(os.path.join(lib_dir, basename)):
                new_name = basename

        if new_name is None:
            continue

        if is_binary:
            new_path = f"@loader_path/../lib/{new_name}"
        else:
            new_path = f"@loader_path/{new_name}"

        subprocess.run(
            ["install_name_tool", "-change", dep, new_path, target],
            capture_output=True
        )
        changes += 1

    # Update install name (id) for dylibs
    if not is_binary:
        name = os.path.basename(target)
        subprocess.run(
            ["install_name_tool", "-id", f"@rpath/{name}", target],
            capture_output=True
        )

    return changes

# Rewrite binary
binary_path = os.path.join(bin_dir, "colmap")
n = rewrite(binary_path, is_binary=True)
print(f"  Binary: {n} paths rewritten")

# Rewrite all dylibs
total = 0
for name in sorted(os.listdir(lib_dir)):
    path = os.path.join(lib_dir, name)
    if os.path.isfile(path):
        n = rewrite(path, is_binary=False)
        total += n

print(f"  Dylibs: {total} paths rewritten across {len(os.listdir(lib_dir))} files")
PYEOF

echo "==> Path rewriting complete"

# ─── Step 6: Codesign everything ────────────────────────────────────────────

echo "==> Codesigning..."
for f in "$LIB_DIR"/*; do
    codesign --force --sign - "$f" 2>/dev/null || true
done
codesign --force --sign - "$BIN_DIR/colmap" 2>/dev/null || true
echo "==> Codesigning complete"

# ─── Step 7: Verify ─────────────────────────────────────────────────────────

echo ""
echo "==> Verification:"
echo "--- Binary deps (should show @loader_path, NOT /opt/homebrew) ---"
otool -L "$BIN_DIR/colmap" 2>/dev/null | head -10

echo ""
REMAINING=$(otool -L "$BIN_DIR/colmap" 2>/dev/null | grep "/opt/homebrew" | wc -l | tr -d ' ')
if [[ "$REMAINING" -gt 0 ]]; then
    echo "⚠️  WARNING: $REMAINING references to /opt/homebrew remain in binary!"
    otool -L "$BIN_DIR/colmap" | grep "/opt/homebrew"
else
    echo "✅ Binary has no remaining /opt/homebrew references"
fi

echo ""
echo "--- Smoke test: running bundled colmap help ---"
if "$BIN_DIR/colmap" help >/dev/null 2>&1; then
    echo "✅ Smoke test PASSED — bundled COLMAP runs standalone"
else
    echo "⚠️  Smoke test may have issues. Checking with DYLD trace..."
    DYLD_PRINT_LIBRARIES=1 "$BIN_DIR/colmap" help 2>&1 | grep "not found\|error" | head -5
fi

echo ""
BUNDLE_SIZE=$(du -sh "$OUTPUT_DIR" | awk '{print $1}')
echo "==> Done! Bundle created at: $OUTPUT_DIR ($BUNDLE_SIZE)"
