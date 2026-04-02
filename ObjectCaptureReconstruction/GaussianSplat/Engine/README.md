# Brush Training Engine

This folder is a transplantable Rust workspace for embedding Brush training into a Swift macOS app.

It is intentionally training-only:

- No desktop viewer app
- No web host
- No Android entry point
- No CI or release automation

## What is here

- `crates/brush-training-bridge`: the Swift-facing `cdylib` and C ABI
- The Brush training, dataset, rendering, kernel, serialization, and utility crates needed by the training path
- `include/brush_training_bridge.h`: the header to import into Swift or Objective-C bridging code
- `build/macos-build.sh`: a small helper that builds the dynamic library on macOS

## Build

From this folder:

```sh
cargo build --release -p brush-training-bridge --lib
```

The library will be produced under:

```sh
target/release/libbrush_training_bridge.dylib
```

## Swift integration shape

- Link `libbrush_training_bridge.dylib` into the macOS app
- Import declarations from `include/brush_training_bridge.h`
- Run `brush_training_run` from a background queue because it blocks until training finishes
- Copy strings passed through callback messages immediately; callback message pointers are only valid for the duration of the callback

## API overview

The first version is deliberately simple:

- One blocking training entry point
- One progress callback that reports lifecycle, step, refine, eval, warning, and completion events
- One `last_error_message` function for detailed failures

This keeps the Swift side small while preserving the existing Brush training behavior.