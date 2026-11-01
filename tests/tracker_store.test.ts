import test from "node:test";
import assert from "node:assert/strict";
import { SortTracker } from "../lib/tracker";
import { detector } from "../lib/detector";
import { spatialFuser } from "../lib/spatial-fusion";
import { dataManager } from "../lib/store";
import { hashPassword, verifyPassword } from "../lib/auth-crypto";

test("Detector generates valid detection records", () => {
  const detections = detector.detect(640, 480, ["human_face", "sunglasses"]);
  assert.ok(Array.isArray(detections));
  assert.ok(detections.length > 0);
  for (const d of detections) {
    assert.ok(d.class);
    assert.ok(typeof d.confidence === "number");
    assert.ok(Array.isArray(d.bbox) && d.bbox.length === 4);
  }
});

test("SortTracker assigns and maintains consistent object IDs", () => {
  const tracker = new SortTracker(15, 2, 0.3);
  const frame1 = [
    {
      class: "human_face" as const,
      confidence: 0.95,
      bbox: [100, 100, 200, 200] as [number, number, number, number],
    },
  ];
  const tracked1 = tracker.update(frame1);
  assert.ok(tracked1.length > 0);
  const trackId = tracked1[0].track_id;
  assert.ok(typeof trackId === "number");

  // Move slightly in next frame
  const frame2 = [
    {
      class: "human_face" as const,
      confidence: 0.96,
      bbox: [105, 103, 205, 203] as [number, number, number, number],
    },
  ];
  const tracked2 = tracker.update(frame2);
  assert.ok(tracked2.length > 0);
  assert.equal(tracked2[0].track_id, trackId);
});

test("Spatial metric fuser estimates 3D world coordinates", () => {
  const detections = detector.detect(640, 480);
  const fused = spatialFuser.estimateSpatialMetrics(detections, 640, 480);
  assert.ok(fused.length > 0);
  const with3D = fused.find((d) => d.spatial_3d);
  assert.ok(with3D);
  const distance = with3D.spatial_3d?.distance_m;
  assert.ok(typeof distance === "number");
  assert.ok(distance > 0);
});

test("Auth crypto safely hashes and verifies passwords", async () => {
  const pass = "AntigravitySecurePass123!";
  const hash = hashPassword(pass);
  assert.ok(hash.startsWith("pbkdf2:"));
  const isValid = verifyPassword(pass, hash);
  assert.equal(isValid, true);
  const isInvalid = verifyPassword("WrongPassword", hash);
  assert.equal(isInvalid, false);
});

test("DataManager manages lockouts, sessions, and telemetry", () => {
  const identifier = "test_operator_" + Date.now();
  assert.equal(dataManager.isAccountLocked(identifier), false);

  // Record 5 failed attempts
  for (let i = 0; i < 4; i++) {
    const res = dataManager.recordFailedLogin(identifier);
    assert.equal(res.locked, false);
  }
  const fatalRes = dataManager.recordFailedLogin(identifier);
  assert.equal(fatalRes.locked, true);
  assert.equal(dataManager.isAccountLocked(identifier), true);

  // Clear lockouts
  dataManager.clearFailedLogins(identifier);
  assert.equal(dataManager.isAccountLocked(identifier), false);

  // Create session
  const token = dataManager.createSession("usr_test_123");
  assert.ok(token.startsWith("sess_"));

  // Save crop and verify
  const crop = dataManager.saveCrop({
    object_class: "human_face",
    confidence: 0.98,
    crop_image: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=",
  });
  assert.ok(crop.id.startsWith("crop_"));

  const latest = dataManager.getLatestCrop("human_face");
  assert.ok(latest);
  assert.equal(latest.object_class, "human_face");

  const stats = dataManager.getStats();
  assert.ok(stats.total_detections > 0);
  assert.ok(stats.active_sessions > 0);
});
