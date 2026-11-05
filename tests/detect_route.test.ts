import assert from "node:assert/strict";
import test from "node:test";
import { NextRequest } from "next/server";
import { POST } from "../app/api/detect/route";

test("forwards captured image bytes and tracking session to inference", async () => {
  const originalUrl = process.env.VISION_API_URL;
  const originalAuthUrl = process.env.AUTH_API_URL;
  const originalInternalToken = process.env.INTERNAL_API_TOKEN;
  const originalFetch = globalThis.fetch;
  const expectedBytes = new Uint8Array([0xff, 0xd8, 0xff, 0xd9]);
  process.env.VISION_API_URL = "http://vision.test/api/detect/";
  process.env.AUTH_API_URL = "http://auth.test";
  process.env.INTERNAL_API_TOKEN = "internal-secret";

  globalThis.fetch = (async (input, init) => {
    if (String(input) === process.env.VISION_API_URL) {
      assert.equal(init?.method, "POST");
      assert.equal(
        new Headers(init?.headers).get("x-tracking-session"),
        "a4f8e98d-c19d-4c84-9d4c-82f4a1f68b08"
      );
      assert.ok(init?.body instanceof FormData);
      const form = init.body as FormData;
      const image = form.get("image");
      assert.ok(image instanceof File);
      assert.deepEqual(new Uint8Array(await image.arrayBuffer()), expectedBytes);
      assert.equal(form.get("include_depth"), "true");
      return Response.json({
        success: true,
        total_detections: 1,
        detections: [{ class: "can", confidence: 0.76, bbox: [1, 2, 3, 4] }],
        timing_ms: { inference_ms: 42.5, total_ms: 45, depth_ms: "ignored" },
      });
    }

    assert.equal(String(input), "http://auth.test/detections/batch");
    assert.equal(new Headers(init?.headers).get("x-internal-token"), "internal-secret");
    const batch = JSON.parse(String(init?.body));
    assert.deepEqual(batch.detections, [{ class_name: "can", confidence: 0.76, has_crop: false }]);
    // Vision-service timings are forwarded; the round trip is measured by the web server.
    assert.equal(batch.frame.inference_ms, 42.5);
    assert.equal(batch.frame.total_ms, 45);
    assert.equal(typeof batch.frame.roundtrip_ms, "number");
    assert.ok(batch.frame.roundtrip_ms >= 0);
    assert.equal("depth_ms" in batch.frame, false);
    return Response.json({ success: true, count: 1 });
  }) as typeof fetch;

  try {
    const form = new FormData();
    form.append("image", new Blob([expectedBytes], { type: "image/jpeg" }), "frame.jpg");
    form.append("include_depth", "true");
    const request = new NextRequest("http://localhost/api/detect", {
      method: "POST",
      headers: { cookie: "vision_session_id=a4f8e98d-c19d-4c84-9d4c-82f4a1f68b08" },
      body: form,
    });

    const response = await POST(request);
    assert.equal(response.status, 200);
    assert.deepEqual(await response.json(), {
      success: true,
      total_detections: 1,
      detections: [{ class: "can", confidence: 0.76, bbox: [1, 2, 3, 4] }],
      timing_ms: { inference_ms: 42.5, total_ms: 45, depth_ms: "ignored" },
      telemetry_persisted: true,
    });
  } finally {
    globalThis.fetch = originalFetch;
    if (originalUrl === undefined) delete process.env.VISION_API_URL;
    else process.env.VISION_API_URL = originalUrl;
    if (originalAuthUrl === undefined) delete process.env.AUTH_API_URL;
    else process.env.AUTH_API_URL = originalAuthUrl;
    if (originalInternalToken === undefined) delete process.env.INTERNAL_API_TOKEN;
    else process.env.INTERNAL_API_TOKEN = originalInternalToken;
  }
});

test("does not fabricate detections when the inference service is unavailable", async () => {
  const originalUrl = process.env.VISION_API_URL;
  const originalNodeEnv = process.env.NODE_ENV;
  const mutableEnv = process.env as Record<string, string | undefined>;
  process.env.VISION_API_URL = "";
  mutableEnv.NODE_ENV = "production";

  try {
    const response = await POST(
      new NextRequest("http://localhost/api/detect", { method: "POST" })
    );
    assert.equal(response.status, 503);
    assert.deepEqual(await response.json(), {
      error: "Vision inference service is not configured",
    });
  } finally {
    if (originalUrl === undefined) delete process.env.VISION_API_URL;
    else process.env.VISION_API_URL = originalUrl;
    if (originalNodeEnv === undefined) delete mutableEnv.NODE_ENV;
    else mutableEnv.NODE_ENV = originalNodeEnv;
  }
});

test("rejects malformed model detections instead of forwarding them", async () => {
  const originalUrl = process.env.VISION_API_URL;
  const originalFetch = globalThis.fetch;
  process.env.VISION_API_URL = "http://vision.test/api/detect/";
  globalThis.fetch = (async () =>
    Response.json({
      success: true,
      detections: [{ class: "can", confidence: 0.76 }],
    })) as typeof fetch;

  try {
    const form = new FormData();
    form.append("image", new Blob([new Uint8Array([0xff, 0xd8, 0xff, 0xd9])], { type: "image/jpeg" }), "frame.jpg");
    const response = await POST(
      new NextRequest("http://localhost/api/detect", { method: "POST", body: form })
    );

    assert.equal(response.status, 502);
    assert.deepEqual(await response.json(), {
      error: "Vision inference returned malformed detections",
    });
  } finally {
    globalThis.fetch = originalFetch;
    if (originalUrl === undefined) delete process.env.VISION_API_URL;
    else process.env.VISION_API_URL = originalUrl;
  }
});