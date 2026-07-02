import base64
from unittest.mock import patch, MagicMock
import numpy as np

from django.core.cache import cache
from django.test import TestCase
from django.urls import reverse

from backend.ml_model.tracker import SortTracker, calculate_iou, bbox_to_z, x_to_bbox
from backend.ml_model.spatial_fusion import Spatial3DFusion
from backend.ml_model.scene_graph import SpatialSceneGraph
from backend.repositories.detection_buffer import DetectionBatchBuffer

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


class DetectEndpointTests(BaseTestCase):
    def test_get_is_not_allowed(self):
        response = self.client.get(DETECT_URL)
        self.assertEqual(response.status_code, 405)

    def test_post_without_image_returns_400(self):
        response = self.client.post(DETECT_URL)
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json())

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


class SpatialFusionAndSceneGraphTests(TestCase):
    def test_spatial_metric_estimation(self):
        fuser = Spatial3DFusion(fov_degrees=60.0, min_depth_m=0.5, max_depth_m=3.5)
        depth_map = np.full((100, 100), 0.8, dtype=np.float32)
        dets = [{"bbox": [20.0, 20.0, 80.0, 80.0], "class": "chair", "confidence": 0.8}]
        enriched = fuser.estimate_spatial_metrics(dets, depth_map, (100, 100))

        self.assertEqual(len(enriched), 1)
        spatial = enriched[0]["spatial_3d"]
        self.assertIn("distance_m", spatial)
        self.assertIn("x_m", spatial)
        self.assertIn("y_m", spatial)
        self.assertIn("z_m", spatial)
        self.assertGreater(spatial["distance_m"], 0.0)

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
        self.assertIn("human_face", graph["summary_prompt"])


class BatchBufferTests(TestCase):
    def test_buffer_accumulates_and_clears(self):
        buffer = DetectionBatchBuffer(flush_interval_sec=10.0, max_batch_size=100)
        buffer.add("test-user-uuid", {"class": "fire", "confidence": 0.95, "bbox": [0, 0, 10, 10]})
        with buffer._lock:
            self.assertEqual(len(buffer._buffer), 1)
            self.assertEqual(buffer._buffer[0]["object_class"], "fire")