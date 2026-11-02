/**
 * Geometry helpers accelerated by an optional Rust/WASM module.
 *
 * The JavaScript implementation is always available; the WASM module is an
 * optimisation and never a hard dependency. Build it with:
 *   cd rust/geometry && wasm-pack build --release --target web
 * then copy the generated `.wasm` to `public/wasm/geometry.wasm`.
 */

export interface PointCloudBounds {
  min: [number, number, number];
  max: [number, number, number];
  center: [number, number, number];
  radius: number;
}

export interface PointCloudData {
  positions: Float32Array;
  colors: Float32Array;
}

interface GeometryWasmExports {
  memory: WebAssembly.Memory;
  alloc_f32(length: number): number;
  dealloc_f32(pointer: number, length: number): void;
  point_cloud_bounds(pointer: number, pointCount: number, outputPointer: number): number;
  normalize_point_cloud(pointer: number, pointCount: number, targetRadius: number): number;
}

let wasmPromise: Promise<GeometryWasmExports | null> | null = null;

export function computePointCloudBounds(positions: ArrayLike<number>): PointCloudBounds {
  if (positions.length === 0 || positions.length % 3 !== 0) {
    throw new Error("positions must contain a non-zero multiple of three coordinates");
  }
  let minX = Infinity;
  let minY = Infinity;
  let minZ = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  let maxZ = -Infinity;
  for (let index = 0; index < positions.length; index += 3) {
    const x = positions[index];
    const y = positions[index + 1];
    const z = positions[index + 2];
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
      continue;
    }
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (z < minZ) minZ = z;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
    if (z > maxZ) maxZ = z;
  }
  if (!Number.isFinite(minX)) {
    throw new Error("positions contain no finite coordinates");
  }
  const center: [number, number, number] = [
    (minX + maxX) / 2,
    (minY + maxY) / 2,
    (minZ + maxZ) / 2,
  ];
  const radius = Math.max(
    1e-6,
    Math.hypot(maxX - minX, maxY - minY, maxZ - minZ) / 2,
  );
  return {
    min: [minX, minY, minZ],
    max: [maxX, maxY, maxZ],
    center,
    radius,
  };
}

export function normalizePointCloud(
  positions: Float32Array,
  targetRadius = 1,
): Float32Array {
  const bounds = computePointCloudBounds(positions);
  const scale = targetRadius / bounds.radius;
  const normalized = new Float32Array(positions.length);
  for (let index = 0; index < positions.length; index += 3) {
    normalized[index] = (positions[index] - bounds.center[0]) * scale;
    normalized[index + 1] = (positions[index + 1] - bounds.center[1]) * scale;
    normalized[index + 2] = (positions[index + 2] - bounds.center[2]) * scale;
  }
  return normalized;
}

export function generatePointCloud(count: number, seed = 1): PointCloudData {
  if (!Number.isInteger(count) || count < 1 || count > 2_000_000) {
    throw new Error("point count must be an integer between 1 and 2,000,000");
  }
  const positions = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);
  let state = seed >>> 0 || 1;
  const random = () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 0xffffffff;
  };
  for (let index = 0; index < count; index += 1) {
    const radius = 0.65 + random() * 0.35;
    const theta = random() * Math.PI * 2;
    const phi = Math.acos(2 * random() - 1);
    const x = radius * Math.sin(phi) * Math.cos(theta);
    const y = radius * Math.sin(phi) * Math.sin(theta);
    const z = radius * Math.cos(phi);
    const offset = index * 3;
    positions[offset] = x;
    positions[offset + 1] = y;
    positions[offset + 2] = z;
    colors[offset] = 0.25 + 0.75 * (x + 1) * 0.5;
    colors[offset + 1] = 0.35 + 0.65 * (y + 1) * 0.5;
    colors[offset + 2] = 0.55 + 0.45 * (z + 1) * 0.5;
  }
  return { positions, colors };
}

export async function loadGeometryWasm(
  url = "/wasm/geometry.wasm",
): Promise<GeometryWasmExports | null> {
  if (typeof WebAssembly === "undefined" || typeof fetch === "undefined") {
    return null;
  }
  if (wasmPromise === null) {
    wasmPromise = (async () => {
      try {
        const response = await fetch(url);
        if (!response.ok) return null;
        const bytes = await response.arrayBuffer();
        const { instance } = await WebAssembly.instantiate(bytes, {});
        const exports = instance.exports as unknown as Partial<GeometryWasmExports>;
        if (
          !exports.memory ||
          typeof exports.alloc_f32 !== "function" ||
          typeof exports.dealloc_f32 !== "function" ||
          typeof exports.point_cloud_bounds !== "function"
        ) {
          return null;
        }
        return exports as GeometryWasmExports;
      } catch {
        return null;
      }
    })();
  }
  return wasmPromise;
}

export async function computePointCloudBoundsAccelerated(
  positions: Float32Array,
): Promise<PointCloudBounds> {
  const wasm = await loadGeometryWasm();
  if (!wasm) {
    return computePointCloudBounds(positions);
  }
  const floatCount = positions.length;
  const inputPointer = wasm.alloc_f32(floatCount);
  const outputPointer = wasm.alloc_f32(6);
  try {
    new Float32Array(wasm.memory.buffer, inputPointer, floatCount).set(positions);
    const ok = wasm.point_cloud_bounds(
      inputPointer,
      floatCount / 3,
      outputPointer,
    );
    if (!ok) {
      return computePointCloudBounds(positions);
    }
    const out = new Float32Array(wasm.memory.buffer, outputPointer, 6);
    const [minX, minY, minZ, maxX, maxY, maxZ] = Array.from(out);
    const center: [number, number, number] = [
      (minX + maxX) / 2,
      (minY + maxY) / 2,
      (minZ + maxZ) / 2,
    ];
    return {
      min: [minX, minY, minZ],
      max: [maxX, maxY, maxZ],
      center,
      radius: Math.max(
        1e-6,
        Math.hypot(maxX - minX, maxY - minY, maxZ - minZ) / 2,
      ),
    };
  } finally {
    wasm.dealloc_f32(inputPointer, floatCount);
    wasm.dealloc_f32(outputPointer, 6);
  }
}
