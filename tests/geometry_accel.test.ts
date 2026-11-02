import assert from "node:assert/strict";
import test from "node:test";

import { detectGpuBackend } from "../lib/gpu-capabilities";
import {
  computePointCloudBounds,
  generatePointCloud,
  normalizePointCloud,
} from "../lib/geometry-accel";

test("computes bounds and center for a point cloud", () => {
  const bounds = computePointCloudBounds([
    -1, -2, -3,
    1, 2, 3,
    0, 0, 0,
  ]);
  assert.deepEqual(bounds.min, [-1, -2, -3]);
  assert.deepEqual(bounds.max, [1, 2, 3]);
  assert.deepEqual(bounds.center, [0, 0, 0]);
  assert.ok(bounds.radius > 0);
});

test("ignores non-finite coordinates and rejects empty input", () => {
  const bounds = computePointCloudBounds([NaN, 0, 0, 2, 2, 2]);
  assert.deepEqual(bounds.min, [2, 2, 2]);
  assert.throws(() => computePointCloudBounds([]), /non-zero multiple/);
  assert.throws(() => computePointCloudBounds([1, 2]), /non-zero multiple/);
});

test("normalizes a cloud to the requested radius", () => {
  const normalized = normalizePointCloud(
    new Float32Array([-2, 0, 0, 2, 0, 0]),
    1,
  );
  const bounds = computePointCloudBounds(normalized);
  assert.ok(Math.abs(bounds.center[0]) < 1e-6);
  assert.ok(Math.abs(bounds.radius - 1) < 1e-6);
});

test("generates deterministic point buffers", () => {
  const first = generatePointCloud(1000, 7);
  const second = generatePointCloud(1000, 7);
  assert.equal(first.positions.length, 3000);
  assert.equal(first.colors.length, 3000);
  assert.deepEqual(Array.from(first.positions.slice(0, 9)), Array.from(second.positions.slice(0, 9)));
  assert.throws(() => generatePointCloud(0), /between 1 and/);
  assert.throws(() => generatePointCloud(1.5), /integer/);
});

test("detects GPU backends without a browser", () => {
  assert.equal(detectGpuBackend({}), "none");
  assert.equal(detectGpuBackend({ gpu: { requestAdapter() {} } }), "webgpu");

  const webgl2Scope = {
    document: {
      createElement: () => ({
        getContext: (id: string) => (id === "webgl2" ? {} : null),
      }),
    },
  };
  assert.equal(detectGpuBackend(webgl2Scope), "webgl2");
});
