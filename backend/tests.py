import base64
from unittest.mock import patch

from django.core.cache import cache
from django.test import TestCase
from django.urls import reverse

LANDING_URL = "/"
DETECT_URL = "/detect/"
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

    @patch("backend.views.detector")
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

    @patch("backend.views.detector")
    def test_detector_failure_returns_500(self, mock_detector):
        mock_detector.detect_from_bytes.side_effect = ValueError("bad image")
        payload = base64.b64encode(b"not an image").decode()

        response = self.client.post(DETECT_URL, {"image_base64": payload})

        self.assertEqual(response.status_code, 500)
        self.assertIn("error", response.json())

    def test_rate_limit_returns_429(self):
        for _ in range(90):
            self.assertEqual(self.client.post(DETECT_URL).status_code, 400)

        response = self.client.post(DETECT_URL)
        self.assertEqual(response.status_code, 429)
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

    def test_rate_limit_returns_429(self):
        for _ in range(10):
            response = self.client.post(
                CHAT_URL, data={}, content_type="application/json"
            )
            self.assertEqual(response.status_code, 400)

        response = self.client.post(
            CHAT_URL, data={}, content_type="application/json"
        )
        self.assertEqual(response.status_code, 429)