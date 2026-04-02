#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
workspace_root="$(cd "$script_dir/.." && pwd)"

# Xcode GUI build phases run with a minimal PATH, so Rustup-managed Cargo is
# often unavailable unless we restore the common install locations here.
if [[ -f "$HOME/.cargo/env" ]]; then
	# shellcheck disable=SC1090
	. "$HOME/.cargo/env"
fi

export PATH="$HOME/.cargo/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:${PATH:-}"

cargo_bin="${CARGO_BIN:-$(command -v cargo || true)}"
if [[ -z "$cargo_bin" ]]; then
	echo "error: cargo not found. Xcode build phases use a limited PATH; install Rust via rustup or set CARGO_BIN." >&2
	exit 127
fi

cd "$workspace_root"
"$cargo_bin" build --release -p brush-training-bridge --lib

echo "Built: $workspace_root/target/release/libbrush_training_bridge.dylib"