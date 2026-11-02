# SmartVision geometry kernels (Rust → WebAssembly)

This crate provides the optional WebAssembly acceleration used by
`lib/geometry-accel.ts` for point-cloud bounds, normalization, and
million-point geometry work in the browser.

## Build

Install the WebAssembly target and `wasm-pack`:

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-pack
```

Build the raw module:

```bash
cd rust/geometry
cargo build --release --target wasm32-unknown-unknown
mkdir -p ../../public/wasm
cp target/wasm32-unknown-unknown/release/smartvision_geometry.wasm ../../public/wasm/geometry.wasm
```

The TypeScript loader fetches `/wasm/geometry.wasm` at runtime. When the file
is missing or the browser cannot instantiate it, the pure-JavaScript
implementation in `lib/geometry-accel.ts` is used automatically.

## Exported ABI

| Export | Description |
| --- | --- |
| `memory` | Linear memory owned by the module |
| `alloc_f32(length)` | Allocate `length` `f32` values |
| `dealloc_f32(pointer, length)` | Release an allocation |
| `point_cloud_bounds(points, count, output)` | Write `[minX,minY,minZ,maxX,maxY,maxZ]`; returns `1`/`0` |
| `normalize_point_cloud(points, count, radius)` | Center and scale in place; returns `1`/`0` |

## Test

```bash
cargo test
```
