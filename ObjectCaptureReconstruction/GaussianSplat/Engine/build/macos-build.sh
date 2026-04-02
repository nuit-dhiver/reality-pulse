#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
workspace_root="$(cd "$script_dir/.." && pwd)"

cd "$workspace_root"
cargo build --release -p brush-training-bridge --lib

echo "Built: $workspace_root/target/release/libbrush_training_bridge.dylib"