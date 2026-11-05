import base64
import threading
import json
import os
import unittest
from unittest.mock import ANY, patch, MagicMock
import numpy as np
import cv2

from django.core.cache import cache
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client, TestCase, override_settings
from django.urls import reverse

from backend.ml_model.tracker import SortTracker, calculate_iou, bbox_to_z, x_to_bbox
from backend.ml_model.spatial_fusion import Spatial3DFusion
from backend.ml_model.scene_graph import SpatialSceneGraph
from backend.repositories.detection_buffer import DetectionBatchBuffer
from backend.ml_model.depth_estimation import depth_estimator
from backend.services.rtsp_ingestion import RTSPIngestionManager, RTSPStreamMetrics

LANDING_URL = "/"
DETECT_URL = "/api/detect/"
CHAT_URL = "/api/chat/"


class BaseTestCase(TestCase):
    def setUp(self):
        cache.clear()


class PublicPageTests(BaseTestCase):
    def test_landing_page_loads(self):
        response = self.client.get(LANDING_URL)
        self.assertEqual(response.status_code, 200)

    def test_login_page_loads(self):
        response = self.client.get(reverse("login"))
        self.assertEqual(response.status_code, 200)

    def test_dashboard_redirects_when_logged_out(self):
        response = self.client.get(reverse("dashboard"))
        self.assertEqual(response.status_code, 302)
        self.assertIn(reverse("login"), response["Location"])

    def test_github_oauth_cancel_returns_login_error(self):
        response = self.client.get(reverse("github_callback"), {"error": "access_denied"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.context["error"],
            "GitHub login was cancelled or failed.",
        )


class LoginTests(BaseTestCase):
    def test_missing_password_shows_error(self):
        response = self.client.post(
            reverse("login"), {"email_or_username": "someone"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("provide both", response.context["error"])

    def test_repeated_attempts_on_one_account_get_rate_limited(self):
        for _ in range(5):
            response = self.client.post(
                reverse("login"), {"email_or_username": "target"}
            )
            self.assertEqual(response.status_code, 200)

        response = self.client.post(
            reverse("login"), {"email_or_username": "target"}
        )
        self.assertEqual(response.status_code, 429)


DETECT_INTERNAL_TOKEN = "detect-internal-token"


@override_settings(INTERNAL_API_TOKEN=DETECT_INTERNAL_TOKEN)
class DetectEndpointTests(BaseTestCase):
    def setUp(self):
        super().setUp()
        self.client = Client(HTTP_X_INTERNAL_TOKEN=DETECT_INTERNAL_TOKEN)

    def test_anonymous_request_without_internal_token_is_rejected(self):
        response = Client().post(
            DETECT_URL,
            {"image": SimpleUploadedFile("frame.jpg", b"jpeg-frame", "image/jpeg")},
        )
        self.assertEqual(response.status_code, 401)

    def test_wrong_internal_token_is_rejected(self):
        response = Client(HTTP_X_INTERNAL_TOKEN="wrong").post(
            DETECT_URL,
            {"image": SimpleUploadedFile("frame.jpg", b"jpeg-frame", "image/jpeg")},
        )
        self.assertEqual(response.status_code, 401)

    def test_get_is_not_allowed(self):
        response = self.client.get(DETECT_URL)
        self.assertEqual(response.status_code, 405)

    def test_post_without_image_returns_400(self):
        response = self.client.post(DETECT_URL)
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json())


    @patch("backend.views.vision_service.process_frame", return_value=[])
    def test_uploaded_image_uses_forwarded_tracking_session(self, process_frame):
        response = self.client.post(
            DETECT_URL,
            {"image": SimpleUploadedFile("frame.jpg", b"jpeg-frame", "image/jpeg")},
            HTTP_X_TRACKING_SESSION="vision-session-1",
        )

        self.assertEqual(response.status_code, 200)
        process_frame.assert_called_once_with(
            b"jpeg-frame",
            session_key="vision-session-1",
            user_id=None,
            include_depth=False,
            timings=ANY,
        )

    def test_real_model_processes_uploaded_camera_frame(self):
        encoded_ok, encoded_image = cv2.imencode(
            ".jpg", np.zeros((480, 640, 3), dtype=np.uint8)
        )
        self.assertTrue(encoded_ok)

        response = self.client.post(
            DETECT_URL,
            {
                "image": SimpleUploadedFile(
                    "camera-frame.jpg", encoded_image.tobytes(), "image/jpeg"
                )
            },
            HTTP_X_TRACKING_SESSION="model-smoke-session",
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["success"])
        self.assertIsInstance(response.json()["detections"], list)

    @patch("backend.services.vision_service.detector")
    def test_base64_image_returns_detections(self, mock_detector):
        mock_detector.detect_from_bytes.return_value = [
            {
                "class": "knife",
                "class_id": 1,
                "confidence": 0.93,
                "bbox": [10.0, 20.0, 110.0, 220.0],
            }
        ]
        payload = "data:image/jpeg;base64," + base64.b64encode(b"fake image").decode()

        response = self.client.post(DETECT_URL, {"image_base64": payload})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["success"])
        self.assertEqual(body["total_detections"], 1)
        self.assertEqual(body["detections"][0]["class"], "knife")
        mock_detector.detect_from_bytes.assert_called_once_with(b"fake image")

    @patch("backend.services.vision_service.detector")
    def test_detector_failure_returns_500(self, mock_detector):
        mock_detector.detect_from_bytes.side_effect = ValueError("bad image")
        payload = base64.b64encode(b"not an image").decode()

        response = self.client.post(DETECT_URL, {"image_base64": payload})

        self.assertEqual(response.status_code, 500)
        self.assertIn("error", response.json())


class RTSPIngestionTests(unittest.TestCase):
    class OneFrameCapture:
        def __init__(self):
            self.released = threading.Event()
            self.sent_frame = False

        def isOpened(self):
            return True

        def read(self):
            if not self.sent_frame:
                self.sent_frame = True
                return True, np.zeros((8, 8, 3), dtype=np.uint8)
            self.released.wait(1)
            return False, None

        def release(self):
            self.released.set()

    def test_registered_stream_processes_frame_and_reports_metrics(self):
        processed = threading.Event()
        received = []
        capture = self.OneFrameCapture()
        manager = RTSPIngestionManager(
            lambda camera_id, sequence, frame: (
                received.append((camera_id, sequence, frame)),
                processed.set(),
            ),
            allowed_hosts={"camera.local:554"},
            capture_factory=lambda _url: capture,
            encode_frame=lambda _frame: b"jpeg-frame",
        )

        try:
            manager.register("front-door", "rtsp://camera.local:554/live")
            self.assertTrue(processed.wait(1))
            metrics = manager.metrics()[0]
            self.assertEqual(received, [("front-door", 1, b"jpeg-frame")])
            self.assertEqual(metrics.frames_received, 1)
            self.assertEqual(metrics.frames_processed, 1)
            self.assertEqual(metrics.frames_dropped, 0)
        finally:
            manager.close()

    def test_rejects_unallowlisted_host_and_excess_streams(self):
        capture = self.OneFrameCapture()
        manager = RTSPIngestionManager(
            lambda *_args: None,
            allowed_hosts={"camera.local:554"},
            maximum_streams=1,
            capture_factory=lambda _url: capture,
            encode_frame=lambda _frame: b"jpeg-frame",
        )

        try:
            with self.assertRaises(PermissionError):
                manager.register("camera-1", "rtsp://other.local/live")
            manager.register("camera-1", "rtsp://camera.local:554/live")
            with self.assertRaises(OverflowError):
                manager.register("camera-2", "rtsp://camera.local:554/live")
        finally:
            manager.close()

    def test_rejects_non_rtsp_urls_and_malformed_camera_ids(self):
        manager = RTSPIngestionManager(lambda *_args: None, allowed_hosts={"camera.local"})

        with self.assertRaises(ValueError):
            manager.register("../camera", "rtsp://camera.local/live")
        with self.assertRaises(ValueError):
            manager.register("camera-1", "http://camera.local/live")


class RTSPStreamEndpointTests(BaseTestCase):
    TOKEN = "internal-test-token"

    def request(self, method, **kwargs):
        path = kwargs.pop("path", reverse("rtsp_streams"))
        return getattr(self.client, method)(
            path,
            HTTP_X_INTERNAL_TOKEN=self.TOKEN,
            **kwargs,
        )

    def test_requires_configured_token_and_authorization_header(self):
        with patch("backend.views.rtsp_stream_manager", MagicMock()):
            response = self.client.get(reverse("rtsp_streams"))
            self.assertEqual(response.status_code, 401)

            response = self.client.get(
                reverse("rtsp_streams"), HTTP_X_INTERNAL_TOKEN="wrong-token"
            )
            self.assertEqual(response.status_code, 401)

    @override_settings(INTERNAL_API_TOKEN=TOKEN)
    def test_rejects_non_ascii_token_without_crashing(self):
        response = self.client.get(
            reverse("rtsp_streams"), HTTP_X_INTERNAL_TOKEN="tökén"
        )
        self.assertEqual(response.status_code, 401)

    @override_settings(INTERNAL_API_TOKEN=TOKEN)
    def test_get_reports_stream_metrics(self):
        manager = MagicMock()
        manager.metrics.return_value = (
            RTSPStreamMetrics(
                "front-door", True, 10, 9, 1, 0, 0, 1234.5
            ),
        )
        with patch("backend.views.rtsp_stream_manager", manager):
            response = self.request("get")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["streams"],
            [
                {
                    "camera_id": "front-door",
                    "connected": True,
                    "frames_received": 10,
                    "frames_processed": 9,
                    "frames_dropped": 1,
                    "reconnects": 0,
                    "processing_errors": 0,
                    "last_frame_at": 1234.5,
                }
            ],
        )

    @override_settings(INTERNAL_API_TOKEN=TOKEN)
    def test_post_registers_stream_and_maps_errors(self):
        manager = MagicMock()
        manager.register.return_value = RTSPStreamMetrics(
            "front-door", True, 0, 0, 0, 0, 0, None
        )
        with patch("backend.views.rtsp_stream_manager", manager):
            response = self.request(
                "post",
                data={
                    "camera_id": "front-door",
                    "stream_url": "rtsp://camera.local:554/live",
                },
                content_type="application/json",
            )
            self.assertEqual(response.status_code, 202)
            self.assertEqual(response.json()["camera_id"], "front-door")
            manager.register.assert_called_once_with(
                "front-door", "rtsp://camera.local:554/live"
            )

            manager.register.side_effect = PermissionError("not allowed")
            response = self.request(
                "post",
                data={"camera_id": "x", "stream_url": "rtsp://other/live"},
                content_type="application/json",
            )
            self.assertEqual(response.status_code, 403)

            manager.register.side_effect = OverflowError("too many")
            response = self.request(
                "post",
                data={"camera_id": "x", "stream_url": "rtsp://other/live"},
                content_type="application/json",
            )
            self.assertEqual(response.status_code, 429)

            manager.register.side_effect = ValueError("bad stream")
            response = self.request("post", data="not json", content_type="application/json")
            self.assertEqual(response.status_code, 400)

    @override_settings(INTERNAL_API_TOKEN=TOKEN)
    def test_delete_unregisters_stream(self):
        manager = MagicMock()
        base_url = reverse("rtsp_streams")
        with patch("backend.views.rtsp_stream_manager", manager):
            response = self.request("delete")
            self.assertEqual(response.status_code, 400)

            manager.unregister.return_value = False
            response = self.request("delete", path=f"{base_url}?camera_id=missing")
            self.assertEqual(response.status_code, 404)

            manager.unregister.return_value = True
            response = self.request("delete", path=f"{base_url}?camera_id=front-door")
            self.assertEqual(response.status_code, 200)
            manager.unregister.assert_called_with("front-door")

    @override_settings(INTERNAL_API_TOKEN=TOKEN)
    def test_returns_503_when_manager_unavailable(self):
        with patch("backend.views.rtsp_stream_manager", None):
            response = self.request("get")
        self.assertEqual(response.status_code, 503)


class ChatEndpointTests(BaseTestCase):
    def test_get_is_not_allowed(self):
        response = self.client.get(CHAT_URL)
        self.assertEqual(response.status_code, 405)

    def test_missing_message_returns_400(self):
        response = self.client.post(
            CHAT_URL, data={}, content_type="application/json"
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"], "Message is required")

    def test_invalid_json_returns_400(self):
        response = self.client.post(
            CHAT_URL, data="not json", content_type="application/json"
        )
        self.assertEqual(response.status_code, 400)

    @override_settings(OFFLINE_MODE=False, GROQ_API_KEY="gsk_test_mock_key")
    @patch("backend.services.chat_service.Groq")
    def test_valid_chat_call_succeeds(self, mock_groq_class):
        mock_client = MagicMock()
        mock_groq_class.return_value = mock_client
        mock_choice = MagicMock()
        mock_choice.message.content = "I see a knife on the table."
        mock_client.chat.completions.create.return_value.choices = [mock_choice]

        payload = {
            "message": "What do you see?",
            "detections": [
                {
                    "track_id": 1,
                    "class": "knife",
                    "confidence": 0.92,
                    "bbox": [10.0, 20.0, 100.0, 200.0],
                }
            ],
        }

        response = self.client.post(
            CHAT_URL, data=payload, content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["success"])
        self.assertEqual(body["response"], "I see a knife on the table.")
        self.assertIn("scene_graph", body)


class TrackingAndMathTests(TestCase):
    def test_bbox_z_roundtrip(self):
        bbox = [50.0, 60.0, 150.0, 260.0]
        z = bbox_to_z(bbox)
        # z is [u, v, s, r]
        self.assertAlmostEqual(float(z[0, 0]), 100.0)  # center x
        self.assertAlmostEqual(float(z[1, 0]), 160.0)  # center y
        self.assertAlmostEqual(float(z[2, 0]), 20000.0)  # area: 100 * 200
        self.assertAlmostEqual(float(z[3, 0]), 0.5)  # aspect ratio: 100 / 200

        recovered = x_to_bbox(z)
        self.assertAlmostEqual(recovered[0], 50.0, places=3)
        self.assertAlmostEqual(recovered[1], 60.0, places=3)
        self.assertAlmostEqual(recovered[2], 150.0, places=3)
        self.assertAlmostEqual(recovered[3], 260.0, places=3)

    def test_iou_calculation(self):
        b1 = [0.0, 0.0, 10.0, 10.0]
        b2 = [5.0, 0.0, 15.0, 10.0]
        iou = calculate_iou(b1, b2)
        # Intersection: 5 * 10 = 50. Union: 100 + 100 - 50 = 150 -> 50/150 = 1/3
        self.assertAlmostEqual(iou, 1.0 / 3.0, places=4)

        b3 = [20.0, 20.0, 30.0, 30.0]
        self.assertEqual(calculate_iou(b1, b3), 0.0)

    def test_sort_tracker_id_preservation(self):
        tracker = SortTracker(max_age=5, min_hits=1, iou_threshold=0.2)

        # Frame 1
        dets_f1 = [{"bbox": [100.0, 100.0, 200.0, 300.0], "class": "pen", "confidence": 0.85}]
        tracks_f1 = tracker.update(dets_f1)
        self.assertEqual(len(tracks_f1), 1)
        t_id = tracks_f1[0]["track_id"]

        # Frame 2: object moves 3 pixels
        dets_f2 = [{"bbox": [103.0, 102.0, 203.0, 302.0], "class": "pen", "confidence": 0.86}]
        tracks_f2 = tracker.update(dets_f2)
        self.assertEqual(len(tracks_f2), 1)
        self.assertEqual(tracks_f2[0]["track_id"], t_id)
        self.assertEqual(tracks_f2[0]["hits"], 2)

    def test_mahalanobis_gating_rejects_spurious_jump(self):
        tracker = SortTracker(max_age=5, min_hits=1, iou_threshold=0.2, gating_threshold=10.0)
        tracker.update([{"bbox": [100.0, 100.0, 150.0, 150.0], "class": "pen", "confidence": 0.9}])
        # Candidate jump far away at (500, 500)
        tracks_jump = tracker.update([{"bbox": [500.0, 500.0, 550.0, 550.0], "class": "pen", "confidence": 0.9}])
        # The distant detection should create a new tracklet rather than matching the previous one
        self.assertGreaterEqual(len(tracks_jump), 1)
        self.assertNotEqual(tracks_jump[0]["track_id"], 1 if len(tracker.trackers) > 1 else 0)


class SpatialFusionAndSceneGraphTests(TestCase):
    def test_spatial_metric_estimation(self):
        fuser = Spatial3DFusion(
            fov_degrees=60.0, min_depth_m=0.5, max_depth_m=3.5, camera_height_m=1.2, camera_pitch_deg=5.0
        )
        depth_map = np.full((100, 100), 0.8, dtype=np.float32)
        dets = [{"bbox": [20.0, 20.0, 80.0, 80.0], "class": "chair", "confidence": 0.8}]
        enriched = fuser.estimate_spatial_metrics(dets, depth_map, (100, 100))

        self.assertEqual(len(enriched), 1)
        spatial = enriched[0]["spatial_3d"]
        self.assertIn("distance_m", spatial)
        self.assertIn("x_m", spatial)
        self.assertIn("y_m", spatial)
        self.assertIn("z_m", spatial)
        self.assertIn("elevation_ground_m", spatial)
        self.assertIn("elevation_deg", spatial)
        self.assertGreater(spatial["distance_m"], 0.0)
        self.assertGreater(spatial["elevation_ground_m"], 0.0)

    def test_scene_graph_builder(self):
        builder = SpatialSceneGraph(proximity_threshold_m=1.0)
        items = [
            {
                "track_id": 1,
                "class": "human_face",
                "confidence": 0.9,
                "spatial_3d": {"x_m": 0.0, "y_m": 0.0, "z_m": 1.0, "distance_m": 1.0},
                "velocity": [0.0, 0.0],
            },
            {
                "track_id": 2,
                "class": "mobile_phone",
                "confidence": 0.85,
                "spatial_3d": {"x_m": 0.1, "y_m": 0.1, "z_m": 1.05, "distance_m": 1.06},
                "velocity": [1.0, 0.5],
            },
        ]
        graph = builder.build(items)
        self.assertEqual(len(graph["nodes"]), 2)
        self.assertEqual(len(graph["edges"]), 1)
        self.assertEqual(graph["edges"][0]["relation"], "interacting_with")
        self.assertEqual(graph["nodes"][0]["degree_centrality"], 1.0)
        self.assertEqual(graph["metrics"]["max_degree"], 1)
        self.assertIn("human_face", graph["summary_prompt"])

    def test_scene_graph_2d_fallback_without_depth(self):
        builder = SpatialSceneGraph()
        items = [
            {"class": "chair", "confidence": 0.8, "bbox": [10.0, 10.0, 50.0, 50.0]},
            {"class": "water_bottle", "confidence": 0.85, "bbox": [20.0, 20.0, 40.0, 60.0]},
        ]
        graph = builder.build(items)
        self.assertEqual(len(graph["nodes"]), 2)
        # Unique track IDs should have been assigned automatically
        self.assertNotEqual(graph["nodes"][0]["track_id"], graph["nodes"][1]["track_id"])
        self.assertGreaterEqual(len(graph["edges"]), 1)
        self.assertIn(graph["edges"][0]["relation"], ["overlapping_with", "adjacent_to"])


class BatchBufferTests(TestCase):
    def test_buffer_accumulates_and_clears(self):
        buffer = DetectionBatchBuffer(flush_interval_sec=10.0, max_batch_size=100)
        buffer.add("test-user-uuid", {"class": "fire", "confidence": 0.95, "bbox": [0, 0, 10, 10]})
        with buffer._lock:
            self.assertEqual(len(buffer._buffer), 1)
            self.assertEqual(buffer._buffer[0]["object_class"], "fire")


class ModelInferenceInternalTests(TestCase):
    def test_sigmoid_numerical_bounds(self):
        from backend.ml_model.inference import detector
        res_zero = detector._sigmoid(np.array([0.0]))
        self.assertAlmostEqual(float(res_zero[0]), 0.5)
        res_large = detector._sigmoid(np.array([1000.0, -1000.0]))
        self.assertAlmostEqual(float(res_large[0]), 1.0)
        self.assertAlmostEqual(float(res_large[1]), 0.0)

    def test_apply_nms_filtering(self):
        from backend.ml_model.inference import detector
        boxes = [
            [0.1, 0.1, 0.3, 0.3],
            [0.11, 0.11, 0.31, 0.31],  # Overlapping box, same class
            [0.6, 0.6, 0.9, 0.9],      # Separate box
        ]
        scores = [0.95, 0.80, 0.88]
        classes = [1, 1, 2]
        res = detector._apply_nms(boxes, scores, classes, iou_threshold=0.3)
        self.assertEqual(len(res["boxes"]), 2)
        self.assertIn(1, res["classes"])
        self.assertIn(2, res["classes"])

    def test_apply_nms_empty(self):
        from backend.ml_model.inference import detector
        res = detector._apply_nms([], [], [])
        self.assertEqual(res["boxes"], [])


class DepthEstimationUnitTests(TestCase):
    @patch.object(depth_estimator, "estimate_depth")
    def test_generate_point_cloud_structure(self, mock_estimate):
        fake_depth = np.full((50, 50), 0.7, dtype=np.float32)
        fake_rgb = np.zeros((50, 50, 3), dtype=np.uint8)
        mock_estimate.return_value = (fake_depth, fake_rgb)

        pc = depth_estimator.generate_point_cloud(b"fake_bytes", max_points=100)
        self.assertIn("positions", pc)
        self.assertIn("colors", pc)
        self.assertGreater(pc["point_count"], 0)
        self.assertEqual(len(pc["positions"]), pc["point_count"] * 3)


class MidasLoaderTests(unittest.TestCase):
    """The loader must never wait for TorchHub's interactive trust prompt."""

    def test_trust_list_is_written_once(self):
        import tempfile
        from backend.ml_model.depth_estimation import TRUSTED_HUB_REPOS, _trust_hub_repos

        with tempfile.TemporaryDirectory() as hub_dir:
            _trust_hub_repos(hub_dir)
            _trust_hub_repos(hub_dir)
            with open(os.path.join(hub_dir, "trusted_list")) as handle:
                lines = [line.strip() for line in handle]
        self.assertEqual(sorted(lines), sorted(TRUSTED_HUB_REPOS))

    def test_repos_are_trusted_before_loading(self):
        import tempfile
        from types import SimpleNamespace
        from backend.ml_model.depth_estimation import MIDAS_REPO, load_midas

        with tempfile.TemporaryDirectory() as hub_dir:
            calls = []

            def fake_load(repo, name, trust_repo):
                with open(os.path.join(hub_dir, "trusted_list")) as handle:
                    trusted = handle.read()
                self.assertIn("rwightman_gen-efficientnet-pytorch", trusted)
                calls.append((repo, name, trust_repo))
                return SimpleNamespace(small_transform="transform") if name == "transforms" else "model"

            fake_torch = SimpleNamespace(hub=SimpleNamespace(get_dir=lambda: hub_dir, load=fake_load))
            model, transform = load_midas(fake_torch)

        self.assertEqual((model, transform), ("model", "transform"))
        self.assertEqual([c[0] for c in calls], [MIDAS_REPO, MIDAS_REPO])
        self.assertNotIn(":master", MIDAS_REPO)


@override_settings(INTERNAL_API_TOKEN=DETECT_INTERNAL_TOKEN)
class PointCloudEndpointTests(BaseTestCase):
    URL = "/api/generate-3d/"

    def test_anonymous_request_is_rejected(self):
        response = Client().post(self.URL, data='{"image_base64": "x"}', content_type="application/json")
        self.assertEqual(response.status_code, 401)

    @patch("backend.ml_model.depth_estimation.depth_estimator.generate_point_cloud")
    def test_depth_scale_and_point_count_are_passed_within_limits(self, generate):
        generate.return_value = {
            "positions": [0.0, 0.0, 1.0], "colors": [1.0, 1.0, 1.0], "point_count": 1,
            "original_width": 2, "original_height": 2, "step": 1,
        }
        client = Client(HTTP_X_INTERNAL_TOKEN=DETECT_INTERNAL_TOKEN)
        payload = base64.b64encode(b"image-bytes").decode()
        response = client.post(
            self.URL,
            data=json.dumps({"image_base64": payload, "depth_scale": 9.0, "max_points": 50}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        generate.assert_called_once_with(b"image-bytes", max_points=1000, depth_scale=5.0)
        self.assertEqual(response.json()["point_count"], 1)

    def test_non_numeric_options_are_rejected(self):
        client = Client(HTTP_X_INTERNAL_TOKEN=DETECT_INTERNAL_TOKEN)
        response = client.post(
            self.URL,
            data=json.dumps({"image_base64": base64.b64encode(b"x").decode(), "depth_scale": "deep"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)


class DepthUnavailableTests(TestCase):
    def setUp(self):
        from backend.ml_model.depth_estimation import DepthEstimator

        self.estimator = DepthEstimator()
        self.saved = (
            self.estimator._initialized,
            self.estimator.model,
            self.estimator.transform,
            self.estimator.unavailable_reason,
        )
        self.estimator._initialized = True
        self.estimator.model = None
        self.estimator.transform = None
        self.estimator.unavailable_reason = "PyTorch is not installed"

    def tearDown(self):
        (
            self.estimator._initialized,
            self.estimator.model,
            self.estimator.transform,
            self.estimator.unavailable_reason,
        ) = self.saved

    def _jpeg(self):
        ok, encoded = cv2.imencode(".jpg", np.zeros((48, 64, 3), dtype=np.uint8))
        self.assertTrue(ok)
        return encoded.tobytes()

    def test_estimate_depth_raises_instead_of_guessing(self):
        from backend.ml_model.depth_estimation import DepthUnavailableError

        with self.assertRaises(DepthUnavailableError):
            self.estimator.estimate_depth(self._jpeg())

    @patch("backend.services.vision_service.detector")
    def test_detections_are_returned_without_3d_values(self, mock_detector):
        from backend.services.vision_service import VisionPipelineService

        mock_detector.detect_from_bytes.return_value = [
            {"class": "can", "class_id": 9, "confidence": 0.9, "bbox": [1.0, 2.0, 30.0, 40.0]}
        ]
        detections = VisionPipelineService().process_frame(
            self._jpeg(), session_key="depth-test", include_depth=True
        )
        self.assertEqual(len(detections), 1)
        self.assertNotIn("spatial_3d", detections[0])


class PointBudgetTests(TestCase):
    @patch.object(depth_estimator, "estimate_depth")
    def test_point_cloud_respects_the_requested_budget(self, mock_estimate):
        mock_estimate.return_value = (np.full((240, 120), 0.7, dtype=np.float32), np.zeros((240, 120, 3), dtype=np.uint8))
        for budget in (1000, 4000, 12000):
            with self.subTest(budget=budget):
                pc = depth_estimator.generate_point_cloud(b"x", max_points=budget)
                self.assertLessEqual(pc["point_count"], budget)
                self.assertGreater(pc["point_count"], budget // 4)


class AuthModuleTests(TestCase):
    def test_get_user_missing_or_bad_bearer(self):
        from backend.auth import get_user_from_request
        from django.test import RequestFactory
        factory = RequestFactory()

        req = factory.get("/")
        user, err = get_user_from_request(req)
        self.assertIsNone(user)
        self.assertEqual(err, "Missing Bearer token")

        req_empty = factory.get("/", HTTP_AUTHORIZATION="Bearer ")
        user, err = get_user_from_request(req_empty)
        self.assertIsNone(user)
        self.assertEqual(err, "Missing Bearer token")

        req_invalid = factory.get("/", HTTP_AUTHORIZATION="Bearer invalid.token.value")
        user, err = get_user_from_request(req_invalid)
        self.assertIsNone(user)
        self.assertIn("Invalid", err)


class ViewsExtendedEndpointsTests(BaseTestCase):
    def test_get_3d_model_success(self):
        response = self.client.get("/api/3d-model/knife/")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["class"], "knife")
        self.assertIn("model_url", body)

    def test_get_3d_model_unknown_class(self):
        response = self.client.get("/api/3d-model/unknown_widget/")
        self.assertEqual(response.status_code, 404)

    def test_camera_feed_redirects_when_logged_out(self):
        response = self.client.get("/camera/")
        self.assertEqual(response.status_code, 302)

    def test_model_viewer_redirects_when_logged_out(self):
        response = self.client.get("/model-viewer/")
        self.assertEqual(response.status_code, 302)

    def test_logout_redirects(self):
        response = self.client.get(reverse("logout"))
        self.assertEqual(response.status_code, 302)

    def test_unauthenticated_protected_endpoints(self):
        # User stats
        resp_stats = self.client.get(reverse("user_stats"))
        self.assertEqual(resp_stats.status_code, 401)

        # Clear history
        resp_clear = self.client.post("/api/clear-history/")
        self.assertEqual(resp_clear.status_code, 401)

        resp_crop = self.client.get("/api/get-crop/?class=sunglasses")
        self.assertEqual(resp_crop.status_code, 401)

        # Save crop
        resp_save = self.client.post("/api/save-crop/", data={})
        self.assertEqual(resp_save.status_code, 401)

    @override_settings(INTERNAL_API_TOKEN=DETECT_INTERNAL_TOKEN)
    def test_generate_3d_view_missing_image(self):
        resp = Client(HTTP_X_INTERNAL_TOKEN=DETECT_INTERNAL_TOKEN).post(
            "/api/generate-3d/",
            data="{}",
            content_type="application/json"
        )
        self.assertEqual(resp.status_code, 400)

    def test_generate_3d_mesh_missing_image(self):
        resp = self.client.post(
            "/api/generate-3d-mesh/",
            data="{}",
            content_type="application/json"
        )
        self.assertEqual(resp.status_code, 400)


class LocalOrmAuthAndDataTests(TestCase):
    def test_local_signup_login_and_telemetry_flow(self):
        import json
        signup_data = {
            "full_name": "Test Engineer",
            "email": "engineer@test.local",
            "password": "SecurePassword123!",
            "confirm_password": "SecurePassword123!",
        }
        resp = self.client.post(reverse("signup"), data=signup_data)
        self.assertEqual(resp.status_code, 302)
        self.assertIn("dashboard", resp["Location"])

        # Authenticated dashboard access
        resp_dash = self.client.get(reverse("dashboard"))
        self.assertEqual(resp_dash.status_code, 200)

        # Authenticated stats endpoint
        resp_stats = self.client.get(reverse("user_stats"))
        self.assertEqual(resp_stats.status_code, 200)
        body = resp_stats.json()
        self.assertEqual(body["total_detections"], 0)

        # Save and retrieve crop
        crop_payload = {
            "object_class": "knife",
            "confidence": 0.94,
            "crop_image": "data:image/png;base64,fakecropdata",
        }
        resp_crop = self.client.post("/api/save-crop/", data=json.dumps(crop_payload), content_type="application/json")
        self.assertEqual(resp_crop.status_code, 200)

        resp_get = self.client.get("/api/get-crop/?object_class=knife")
        self.assertEqual(resp_get.status_code, 200)
        self.assertEqual(resp_get.json()["object_class"], "knife")

        # Clear history
        resp_clear = self.client.post("/api/clear-history/")
        self.assertEqual(resp_clear.status_code, 200)
