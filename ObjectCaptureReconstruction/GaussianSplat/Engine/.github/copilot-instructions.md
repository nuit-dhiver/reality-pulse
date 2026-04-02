# Project Guidelines

## Scope

- This workspace is a training-only extraction of Brush for embedding into a Swift macOS application.
- Do not reintroduce viewer, desktop app, web, or Android entry points unless the host app explicitly needs them.
- The primary product boundary is the C ABI exposed by `crates/brush-training-bridge`.

## Architecture

- Keep the copied Brush crates close to upstream unless there is a clear extraction or embedding reason to change them.
- Put Swift-facing behavior in `crates/brush-training-bridge` instead of scattering FFI code across the copied core crates.
- Preserve the current layering: dataset loading in `brush-dataset`, orchestration in `brush-process`, training in `brush-train`, rendering and kernels in the render and kernel crates.

## Build And Test

- Work from this workspace root when building the extracted engine.
- The main build target is `cargo build --release -p brush-training-bridge --lib`.
- Prefer validating the bridge crate first before touching transitive crates.

## Conventions

- Treat the exported header in `include/brush_training_bridge.h` as part of the public API.
- Prefer additive FFI changes that keep Swift integration stable.
- Keep the Rust bridge blocking and callback-based unless there is a concrete need for session-based or async FFI.
- Error reporting across the FFI boundary should be explicit and textual, not just boolean or enum-only.

## Pointers

- Use `crates/brush-training-bridge/src/ffi.rs` for the Swift-facing ABI.
- Use `crates/brush-process/src/train_stream.rs` for the underlying training loop and emitted events.
- Use `crates/brush-process/src/config.rs` and `crates/brush-train/src/config.rs` for config defaults and behavior.