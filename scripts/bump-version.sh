#!/usr/bin/env bash
# Bump Reality Pulse's version numbers in RealityPulse.xcodeproj.
#
#   scripts/bump-version.sh show           # print current version and build
#   scripts/bump-version.sh build          # 1.1.0 (2) -> 1.1.0 (3)
#   scripts/bump-version.sh patch          # 1.1.0 (2) -> 1.1.1 (3)
#   scripts/bump-version.sh minor          # 1.1.0 (2) -> 1.2.0 (3)
#   scripts/bump-version.sh major          # 1.1.0 (2) -> 2.0.0 (3)
#   scripts/bump-version.sh set 1.4.0      # 1.1.0 (2) -> 1.4.0 (3)
#
# App Store Connect rejects an upload whose build number (CFBundleVersion) is
# not higher than every previous upload, so every command except `show` also
# increments the build number.
set -euo pipefail

cd "$(dirname "$0")/.."
PBXPROJ="RealityPulse.xcodeproj/project.pbxproj"

current_setting() {
    grep -m1 "$1 = " "$PBXPROJ" | sed -E "s/.*$1 = \"?([^\";]*)\"?;.*/\1/"
}

version=$(current_setting MARKETING_VERSION)
build=$(current_setting CURRENT_PROJECT_VERSION)

usage() {
    sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
}

[[ $# -ge 1 ]] || usage

IFS=. read -r major minor patch <<< "$version"
patch=${patch:-0}

case "$1" in
    show)
        echo "$version ($build)"
        exit 0
        ;;
    build) new_version=$version ;;
    patch) new_version="$major.$minor.$((patch + 1))" ;;
    minor) new_version="$major.$((minor + 1)).0" ;;
    major) new_version="$((major + 1)).0.0" ;;
    set)
        [[ $# -eq 2 && $2 =~ ^[0-9]+\.[0-9]+(\.[0-9]+)?$ ]] || {
            echo "error: 'set' needs a version like 1.4.0" >&2
            exit 1
        }
        new_version=$2
        ;;
    *) usage ;;
esac

new_build=$((build + 1))

sed -i '' -E \
    -e "s/MARKETING_VERSION = [^;]*;/MARKETING_VERSION = $new_version;/" \
    -e "s/CURRENT_PROJECT_VERSION = [^;]*;/CURRENT_PROJECT_VERSION = $new_build;/" \
    "$PBXPROJ"

echo "$version ($build) -> $new_version ($new_build)"
