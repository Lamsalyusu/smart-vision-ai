import base64
import json
import logging
import os
import secrets
import time
from datetime import datetime, timedelta
from functools import wraps
from urllib.parse import urlencode

import requests as http_requests

from django.conf import settings
from django.contrib.auth.hashers import check_password, make_password
from django.core.mail import send_mail
from django.http import JsonResponse
from django.shortcuts import redirect, render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from django_ratelimit.decorators import ratelimit
from groq import Groq

from backend.supabase_client import supabase_admin
from .ml_model.inference import detector
from backend.services.vision_service import vision_service
from backend.services.chat_service import chat_service


logger = logging.getLogger(__name__)



# Session helpers
def get_client_ip(request):
    x_forwarded_for = request.META.get("HTTP_X_FORWARDED_FOR")

    if x_forwarded_for:
        return x_forwarded_for.split(",")[0].strip()

    return request.META.get("REMOTE_ADDR")


def session_or_ip(group, request):
    return (
        request.session.get("session_token")
        or request.META.get("REMOTE_ADDR", "unknown")
    )

from backend.repositories.data_manager import data_manager

# too many request --> 429 code
def too_many_requests_json():
    return JsonResponse(
        {"error": "Too many requests. Please slow down."},
        status=429,
    )


def create_session(user_id, request):
    try:
        session_token = data_manager.create_session(
            user_id=str(user_id),
            ip_address=get_client_ip(request),
            user_agent=request.META.get("HTTP_USER_AGENT", ""),
        )
        logger.info("Session created | user=%s", user_id)
        return session_token
    except Exception:
        logger.exception("Failed to create session | user=%s", user_id)
        raise


def get_user_from_session(request):
    session_token = request.session.get("session_token")
    if not session_token:
        return None

    try:
        return data_manager.get_user_from_session(session_token)
    except Exception:
        logger.exception("Failed to validate session")
        return None


def login_required(view_func):
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        user_session = get_user_from_session(request)

        if not user_session:
            return redirect("login")

        request.user_session = user_session
        return view_func(request, *args, **kwargs)

    return wrapper


def find_user_by_identifier(identifier):
    try:
        return data_manager.find_user_by_identifier(identifier)
    except Exception:
        logger.exception("Failed to query user by identifier")
        return None

# Public and authentication views

def landing_page(request):
    if get_user_from_session(request):
        return redirect("dashboard")

    return render(request, "landingpage.html")


@ratelimit(key="ip", rate="10/m", method="POST", block=False)
@ratelimit(key="post:email_or_username", rate="5/m", method="POST", block=False)
def login_page(request):
    if getattr(request, "limited", False):
        logger.warning("Login rate limit hit")

        return render(
            request,
            "loginpage.html",
            {"error": "Too many login attempts. Please wait a minute."},
            status=429,
        )

    if get_user_from_session(request):
        return redirect("dashboard")

    if request.method == "GET":
        return render(request, "loginpage.html")

    email_or_username = request.POST.get("email_or_username")
    password = request.POST.get("password")

    if not email_or_username or not password:
        logger.warning("Login attempt with missing credentials")

        return render(
            request,
            "loginpage.html",
            {"error": "Please provide both email/username and password"},
        )

    try:
        user = find_user_by_identifier(email_or_username)

        if not user:
            logger.warning("Login failed: user not found")

            return render(
                request,
                "loginpage.html",
                {"error": "Invalid credentials"},
            )

        if not check_password(password, user["password_hash"]):
            logger.warning(
                "Login failed: invalid password | user=%s",
                user["id"],
            )

            return render(
                request,
                "loginpage.html",
                {"error": "Invalid credentials"},
            )

        if not user.get("is_active", True):
            logger.warning(
                "Login rejected: inactive account | user=%s",
                user["id"],
            )

            return render(
                request,
                "loginpage.html",
                {"error": "Your account has been deactivated"},
            )

        session_token = create_session(user["id"], request)

        supabase_admin.table("users").update(
            {"last_login": datetime.now().isoformat()}
        ).eq("id", user["id"]).execute()

        request.session["session_token"] = session_token

        logger.info(
            "User logged in successfully | user=%s",
            user["id"],
        )

        return redirect("dashboard")

    except Exception:
        logger.exception("Login failed unexpectedly")

        return render(
            request,
            "loginpage.html",
            {"error": "Unable to login right now. Please try again."},
        )


@ratelimit(key="ip", rate="5/h", method="POST", block=False)
def signup_page(request):
    if getattr(request, "limited", False):
        logger.warning("Signup rate limit hit")

        return render(
            request,
            "signuppage.html",
            {"error": "Too many signup attempts. Please try again later."},
            status=429,
        )

    if get_user_from_session(request):
        return redirect("dashboard")

    if request.method == "GET":
        return render(request, "signuppage.html")

    full_name = request.POST.get("full_name")
    email = request.POST.get("email")
    password = request.POST.get("password")
    confirm_password = request.POST.get("confirm_password")

    if not all([full_name, email, password, confirm_password]):
        logger.warning("Signup attempt with missing fields")

        return render(
            request,
            "signuppage.html",
            {"error": "All fields are required"},
        )

    if password != confirm_password:
        return render(
            request,
            "signuppage.html",
            {"error": "Passwords do not match"},
        )

    if len(password) < 8:
        return render(
            request,
            "signuppage.html",
            {"error": "Password must be at least 8 characters long"},
        )

    username = email.split("@")[0]

    try:
        existing_user = data_manager.find_user_by_identifier(email)
        if existing_user:
            logger.info("Signup rejected: email already registered")
            return render(
                request,
                "signuppage.html",
                {"error": "Email already registered"},
            )

        user = data_manager.create_user(
            full_name=full_name,
            email=email,
            username=username,
            password_hash=make_password(password),
        )

        session_token = create_session(user["id"], request)
        request.session["session_token"] = session_token

        logger.info("New user registered | user=%s", user["id"])
        return redirect("dashboard")

    except Exception:
        logger.exception("Signup failed")
        return render(
            request,
            "signuppage.html",
            {"error": "Error creating account. Please try again."},
        )


def logout_view(request):
    session_token = request.session.get("session_token")
    if session_token:
        try:
            data_manager.invalidate_session(session_token)
            logger.info("User session deactivated")
        except Exception:
            logger.exception("Failed to deactivate user session")

    request.session.flush()
    return redirect("login")

# Dashboard

@login_required
def dashboard(request):
    user_data = request.user_session.get("users", request.user_session)
    user_id = user_data.get("id") or request.user_session.get("id")

    stats = data_manager.get_user_stats(user_id)
    recent = stats.get("recent_detections", [])
    unique_classes = set(stats.get("class_distribution", {}).keys())

    from backend.models import LLMQuery
    total_queries = LLMQuery.objects.filter(user_id=user_id).count()

    # Format recent detections for template
    formatted_recent = [
        {
            "object_class": r["class"],
            "confidence": r["confidence"],
            "created_at": r.get("timestamp", ""),
        }
        for r in recent
    ]

    context = {
        "user": user_data,
        "total_detections": stats.get("total_detections", 0),
        "total_queries": total_queries,
        "unique_classes": len(unique_classes),
        "recent_detections": formatted_recent,
    }

    return render(request, "dashboard.html", context)


@login_required
def camera_feed(request):
    return render(
        request,
        "camera_feed.html",
        {"user": request.user_session["users"]},
    )


@login_required
def model_viewer(request):
    return render(request, "model_viewer_v2.html")


@csrf_exempt
@require_http_methods(["GET"])
def get_user_stats(request):
    user_session = get_user_from_session(request)

    if not user_session:
        logger.warning("Unauthorized user stats request")
        return JsonResponse({"error": "Unauthorized"}, status=401)

    user_id = user_session.get("users", {}).get("id") or user_session.get("id")

    try:
        stats = data_manager.get_user_stats(user_id)
        total_detections = stats.get("total_detections", 0)

        return JsonResponse({
            "total_sessions": 1 if total_detections > 0 else 0,
            "total_detections": total_detections,
            "total_queries": 0,
            "class_distribution": stats.get("class_distribution", {}),
            "recent_detections": stats.get("recent_detections", []),
        })

    except Exception:
        logger.exception("Failed to load user statistics | user=%s", user_id)
        return JsonResponse({"error": "Unable to load statistics"}, status=500)

# Object detection

@csrf_exempt
@ratelimit(key=session_or_ip, rate="90/m", method="POST", block=False)
def detect_objects(request):
    if getattr(request, "limited", False):
        logger.warning("Detection rate limit hit")
        return too_many_requests_json()

    if request.method != "POST":
        return JsonResponse(
            {"error": "POST method required"},
            status=405,
        )

    user_session = get_user_from_session(request)
    user_id = (
        user_session["users"]["id"]
        if user_session
        else None
    )

    logger.info(
        "Detection request received | user=%s",
        user_id,
    )

    try:
        if "image" in request.FILES:
            image_bytes = request.FILES["image"].read()

            logger.info(
                "Processing uploaded image | user=%s | size=%s bytes",
                user_id,
                len(image_bytes),
            )

        elif "image_base64" in request.POST:
            image_data = request.POST["image_base64"]

            if "base64," in image_data:
                image_data = image_data.split("base64,", 1)[1]

            image_bytes = base64.b64decode(image_data)

            logger.info(
                "Processing base64 image | user=%s | size=%s bytes",
                user_id,
                len(image_bytes),
            )

        else:
            logger.warning(
                "Detection request missing image | user=%s",
                user_id,
            )

            return JsonResponse(
                {"error": "No image provided"},
                status=400,
            )

        session_key = request.session.get("session_token") or get_client_ip(request)
        include_depth = request.POST.get("include_depth", "false").lower() == "true"

        detections = vision_service.process_frame(
            image_bytes,
            session_key=session_key,
            user_id=user_id,
            include_depth=include_depth,
        )

        logger.info(
            "Detection completed | user=%s | objects=%s",
            user_id,
            len(detections),
        )

        return JsonResponse({
            "success": True,
            "total_detections": len(detections),
            "detections": detections,
        })

    except Exception:
        logger.exception(
            "Object detection failed | user=%s",
            user_id,
        )

        return JsonResponse(
            {"error": "Object detection failed. Please try again."},
            status=500,
        )


def webcam_feed(request):
    return render(request, "camera_feed.html")

# Static 3D models

CLASS_TO_MODEL = {
    "sunglasses": "sunglasses.glb",
    "knife": "knife.glb",
    "water_bottle": "water_bottle.glb",
    "pen": "pen.glb",
    "chair": "chair.glb",
    "human_face": "human_face.glb",
    "mobile_phone": "mobile_phone.glb",
    "helmet": "helmet.glb",
    "fire": "fire.glb",
    "can": "can.glb",
}


def get_3d_model(request, class_name):
    model_filename = CLASS_TO_MODEL.get(class_name)

    if not model_filename:
        logger.warning(
            "3D model requested for unknown class | class=%s",
            class_name,
        )

        return JsonResponse(
            {"error": "Class not found"},
            status=404,
        )

    return JsonResponse({
        "class": class_name,
        "model_url": f"/static/models/{model_filename}",
        "format": "glb",
    })


# LLM chat

@csrf_exempt
@require_http_methods(["POST"])
@ratelimit(key=session_or_ip, rate="10/m", method="POST", block=False)
def llm_chat(request):
    if getattr(request, "limited", False):
        logger.warning("AI chat rate limit hit")
        return too_many_requests_json()

    user_session = get_user_from_session(request)
    user_id = (
        user_session["users"]["id"]
        if user_session
        else None
    )

    logger.info(
        "AI chat request received | user=%s",
        user_id,
    )

    try:
        data = json.loads(request.body)

        user_message = data.get("message", "")
        detections = data.get("detections", [])

        if not user_message:
            logger.warning(
                "AI chat request missing message | user=%s",
                user_id,
            )

            return JsonResponse(
                {"error": "Message is required"},
                status=400,
            )

        chat_result = chat_service.generate_response(
            user_message=user_message,
            detections=detections,
            user_id=user_id,
        )

        logger.info(
            "AI chat completed | user=%s",
            user_id,
        )

        return JsonResponse({
            "success": True,
            "response": chat_result["response"],
            "scene_graph": chat_result.get("scene_graph"),
        })

    except json.JSONDecodeError:
        logger.warning(
            "Invalid JSON in AI chat request | user=%s",
            user_id,
        )

        return JsonResponse(
            {"error": "Invalid request data"},
            status=400,
        )

    except Exception:
        logger.exception(
            "AI chat failed | user=%s",
            user_id,
        )

        return JsonResponse(
            {"error": "AI service is currently unavailable"},
            status=500,
        )

# MiDaS 3D point cloud

@csrf_exempt
@require_http_methods(["POST"])
@ratelimit(key=session_or_ip, rate="10/m", method="POST", block=False)
def generate_3d_view(request):
    if getattr(request, "limited", False):
        logger.warning("3D point cloud rate limit hit")
        return too_many_requests_json()

    user_session = get_user_from_session(request)
    user_id = (
        user_session["users"]["id"]
        if user_session
        else None
    )

    logger.info(
        "3D point cloud request received | user=%s",
        user_id,
    )

    try:
        if "image" in request.FILES:
            image_bytes = request.FILES["image"].read()

        elif request.content_type == "application/json":
            data = json.loads(request.body)
            image_data = data.get("image_base64", "")

            if "base64," in image_data:
                image_data = image_data.split("base64,", 1)[1]

            image_bytes = base64.b64decode(image_data)

        else:
            logger.warning(
                "3D point cloud request missing image | user=%s",
                user_id,
            )

            return JsonResponse(
                {"error": "No image provided"},
                status=400,
            )

        if not image_bytes:
            return JsonResponse(
                {"error": "Empty image data"},
                status=400,
            )

        from backend.ml_model.depth_estimation import depth_estimator

        result = depth_estimator.generate_point_cloud(
            image_bytes,
            max_points=12000,
            depth_scale=2.0,
        )

        logger.info(
            "3D point cloud generated | user=%s | points=%s",
            user_id,
            result["point_count"],
        )

        return JsonResponse({
            "success": True,
            "positions": result["positions"],
            "colors": result["colors"],
            "point_count": result["point_count"],
            "width": result["original_width"],
            "height": result["original_height"],
        })

    except ImportError as exc:
        logger.exception(
            "MiDaS dependency is missing | user=%s",
            user_id,
        )

        return JsonResponse({
            "error": (
                "MiDaS not installed. "
                "Run: pip install torch torchvision timm"
            ),
            "detail": str(exc),
        }, status=500)

    except json.JSONDecodeError:
        logger.warning(
            "Invalid JSON in 3D point cloud request | user=%s",
            user_id,
        )

        return JsonResponse(
            {"error": "Invalid request data"},
            status=400,
        )

    except Exception:
        logger.exception(
            "3D point cloud generation failed | user=%s",
            user_id,
        )

        return JsonResponse(
            {"error": "3D generation failed. Please try again."},
            status=500,
        )


# OAuth
def google_login(request):
    params = urlencode({
        "client_id": settings.GOOGLE_CLIENT_ID,
        "redirect_uri": settings.GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope": "email profile",
        "access_type": "offline",
        "prompt": "select_account",
    })

    logger.info("Google OAuth login started")

    return redirect(
        "https://accounts.google.com/o/oauth2/v2/auth?" + params
    )


def google_callback(request):
    code = request.GET.get("code")
    error = request.GET.get("error")

    if error or not code:
        logger.warning("Google OAuth login failed")

        return render(
            request,
            "loginpage.html",
            {"error": "Google login failed."},
        )

    try:
        token_resp = http_requests.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code": code,
                "client_id": settings.GOOGLE_CLIENT_ID,
                "client_secret": settings.GOOGLE_CLIENT_SECRET,
                "redirect_uri": settings.GOOGLE_REDIRECT_URI,
                "grant_type": "authorization_code",
            },
        )

        if token_resp.status_code != 200:
            logger.warning("Google OAuth token request failed")

            return render(
                request,
                "loginpage.html",
                {"error": "Failed to get token from Google."},
            )

        access_token = token_resp.json().get("access_token")

        user_resp = http_requests.get(
            "https://www.googleapis.com/oauth2/v2/userinfo",
            headers={
                "Authorization": "Bearer " + access_token,
            },
        )

        if user_resp.status_code != 200:
            logger.warning("Google user info request failed")

            return render(
                request,
                "loginpage.html",
                {"error": "Failed to get user info from Google."},
            )

        google_user = user_resp.json()
        email = google_user.get("email")

        if not email:
            return render(
                request,
                "loginpage.html",
                {"error": "Google did not provide an email address."},
            )

        name = google_user.get(
            "name",
            email.split("@")[0],
        )

        user = _find_or_create_oauth_user(
            email,
            name,
            "google",
        )

        if not user:
            return render(
                request,
                "loginpage.html",
                {"error": "Failed to create account."},
            )

        session_token = create_session(user["id"], request)
        request.session["session_token"] = session_token

        logger.info(
            "Google OAuth login successful | user=%s",
            user["id"],
        )

        return redirect("dashboard")

    except Exception:
        logger.exception("Google OAuth callback failed")

        return render(
            request,
            "loginpage.html",
            {"error": "Google login failed. Please try again."},
        )


def github_login(request):
    params = urlencode({
        "client_id": settings.GITHUB_CLIENT_ID,
        "redirect_uri": settings.GITHUB_REDIRECT_URI,
        "scope": "user:email",
    })

    logger.info("GitHub OAuth login started")

    return redirect(
        "https://github.com/login/oauth/authorize?" + params
    )


def github_callback(request):
    code = request.GET.get("code")
    error = request.GET.get("error")

    if error or not code:
        logger.warning("GitHub OAuth login cancelled or failed")

        return render(
            request,
            "loginpage.html",
            {"error": "GitHub login was cancelled or failed."},
        )

    try:
        token_resp = http_requests.post(
            "https://github.com/login/oauth/access_token",
            data={
                "code": code,
                "client_id": settings.GITHUB_CLIENT_ID,
                "client_secret": settings.GITHUB_CLIENT_SECRET,
                "redirect_uri": settings.GITHUB_REDIRECT_URI,
            },
            headers={
                "Accept": "application/json",
            },
        )

        if token_resp.status_code != 200:
            logger.warning("GitHub OAuth token request failed")

            return render(
                request,
                "loginpage.html",
                {"error": "Failed to get token from GitHub."},
            )

        access_token = token_resp.json().get("access_token")

        user_resp = http_requests.get(
            "https://api.github.com/user",
            headers={
                "Authorization": "Bearer " + access_token,
            },
        )

        if user_resp.status_code != 200:
            logger.warning("GitHub user info request failed")

            return render(
                request,
                "loginpage.html",
                {"error": "Failed to get user info from GitHub."},
            )

        github_user = user_resp.json()
        email = github_user.get("email")

        if not email:
            emails_resp = http_requests.get(
                "https://api.github.com/user/emails",
                headers={
                    "Authorization": "Bearer " + access_token,
                },
            )

            if emails_resp.status_code == 200:
                for email_data in emails_resp.json():
                    if (
                        email_data.get("primary")
                        and email_data.get("verified")
                    ):
                        email = email_data.get("email")
                        break

        if not email:
            return render(
                request,
                "loginpage.html",
                {
                    "error": (
                        "Could not get email from GitHub. "
                        "Make sure your email is public or verified."
                    )
                },
            )

        name = (
            github_user.get("name")
            or github_user.get("login")
            or email.split("@")[0]
        )

        user = _find_or_create_oauth_user(
            email,
            name,
            "github",
        )

        if not user:
            return render(
                request,
                "loginpage.html",
                {"error": "Failed to create account."},
            )

        session_token = create_session(user["id"], request)
        request.session["session_token"] = session_token

        logger.info(
            "GitHub OAuth login successful | user=%s",
            user["id"],
        )

        return redirect("dashboard")

    except Exception:
        logger.exception("GitHub OAuth callback failed")

        return render(
            request,
            "loginpage.html",
            {"error": "GitHub login failed. Please try again."},
        )


def _find_or_create_oauth_user(email, full_name, provider):
    try:
        existing = (
            supabase_admin.table("users")
            .select("*")
            .eq("email", email)
            .execute()
        )

        if existing.data:
            return existing.data[0]

        username = email.split("@")[0]

        user_data = {
            "full_name": full_name,
            "email": email,
            "username": username,
            "password_hash": "",
            "is_active": True,
            "is_verified": True,
            "auth_provider": provider,
        }

        response = (
            supabase_admin.table("users")
            .insert(user_data)
            .execute()
        )

        if response.data:
            return response.data[0]

        logger.warning(
            "OAuth user creation returned no user | provider=%s",
            provider,
        )

        return None

    except Exception:
        logger.exception(
            "Failed to find or create OAuth user | provider=%s",
            provider,
        )
        return None


# Password reset
@ratelimit(key="ip", rate="3/h", method="POST", block=False)
@ratelimit(key="post:email", rate="3/h", method="POST", block=False)
def forgot_password(request):
    if getattr(request, "limited", False):
        logger.warning("Password reset rate limit hit")

        return render(
            request,
            "forgot_password.html",
            {"error": "Too many reset requests. Please try again later."},
            status=429,
        )

    if request.method == "GET":
        return render(request, "forgot_password.html")

    email = request.POST.get("email", "").strip()

    if not email:
        logger.warning("Password reset requested without email")

        return render(
            request,
            "forgot_password.html",
            {"error": "Please enter your email address."},
        )

    try:
        user_resp = (
            supabase_admin.table("users")
            .select("id, email, full_name")
            .eq("email", email)
            .execute()
        )

        if not user_resp.data:
            logger.info("Password reset requested for unknown email")

            return redirect("/forgot-password/?success=true")

        user = user_resp.data[0]

        token = secrets.token_urlsafe(32)
        expires_at = (
            datetime.now() + timedelta(hours=1)
        ).isoformat()

        (
            supabase_admin.table("password_resets")
            .insert({
                "user_id": str(user["id"]),
                "token": token,
                "expires_at": expires_at,
                "used": False,
            })
            .execute()
        )

        reset_url = request.build_absolute_uri(
            f"/reset-password/{token}/"
        )

        send_mail(
            subject="Smart Vision - Password Reset",
            message=(
                f'Hi {user["full_name"]},\n\n'
                "You requested a password reset. "
                "Click the link below:\n\n"
                f"{reset_url}\n\n"
                "This link expires in 1 hour.\n\n"
                "If you did not request this, ignore this email.\n\n"
                "- Smart Vision AI"
            ),
            from_email=settings.DEFAULT_FROM_EMAIL,
            recipient_list=[email],
            fail_silently=False,
        )

        logger.info(
            "Password reset email sent | user=%s",
            user["id"],
        )

    except Exception:
        logger.exception("Password reset request failed")

        return render(
            request,
            "forgot_password.html",
            {
                "error": (
                    "Could not process your request. "
                    "Please try again later."
                )
            },
        )

    return redirect("/forgot-password/?success=true")


@ratelimit(key="ip", rate="10/h", method="POST", block=False)
def reset_password(request, token):
    if getattr(request, "limited", False):
        logger.warning("Password reset submission rate limit hit")

        return render(
            request,
            "reset_password.html",
            {
                "token": token,
                "token_valid": True,
                "error": "Too many attempts. Please try again later.",
            },
            status=429,
        )

    try:
        token_resp = (
            supabase_admin.table("password_resets")
            .select("*, users(id, email, full_name)")
            .eq("token", token)
            .eq("used", False)
            .execute()
        )

        if not token_resp.data:
            logger.warning("Invalid password reset token")

            return render(
                request,
                "reset_password.html",
                {
                    "error": "Invalid or expired reset link.",
                    "token_valid": False,
                },
            )

        reset_data = token_resp.data[0]

        expires_at = datetime.fromisoformat(
            reset_data["expires_at"].replace("Z", "+00:00")
        )

        if datetime.now(expires_at.tzinfo) > expires_at:
            logger.warning("Expired password reset token")

            return render(
                request,
                "reset_password.html",
                {
                    "error": (
                        "This reset link has expired. "
                        "Please request a new one."
                    ),
                    "token_valid": False,
                },
            )

        if request.method == "GET":
            return render(
                request,
                "reset_password.html",
                {
                    "token": token,
                    "token_valid": True,
                    "email": reset_data["users"]["email"],
                },
            )

        password = request.POST.get("password", "")
        confirm_password = request.POST.get(
            "confirm_password",
            "",
        )

        if not password or len(password) < 8:
            return render(
                request,
                "reset_password.html",
                {
                    "token": token,
                    "token_valid": True,
                    "error": "Password must be at least 8 characters.",
                },
            )

        if password != confirm_password:
            return render(
                request,
                "reset_password.html",
                {
                    "token": token,
                    "token_valid": True,
                    "error": "Passwords do not match.",
                },
            )

        user_id = reset_data["user_id"]

        (
            supabase_admin.table("users")
            .update({
                "password_hash": make_password(password)
            })
            .eq("id", user_id)
            .execute()
        )

        (
            supabase_admin.table("password_resets")
            .update({"used": True})
            .eq("token", token)
            .execute()
        )

        logger.info(
            "Password reset completed | user=%s",
            user_id,
        )

    except Exception:
        logger.exception("Password reset failed")

        return render(
            request,
            "reset_password.html",
            {
                "token": token,
                "token_valid": True,
                "error": (
                    "Something went wrong. "
                    "Please try again."
                ),
            },
        )

    return render(
        request,
        "reset_password.html",
        {
            "success": True,
            "token_valid": False,
        },
    )

# Detection crops

@csrf_exempt
@require_http_methods(["POST"])
def save_crop(request):
    user_session = get_user_from_session(request)

    if not user_session:
        logger.warning("Unauthorized crop save request")
        return JsonResponse(
            {"error": "Unauthorized"},
            status=401,
        )

    user_id = user_session.get("users", {}).get("id") or user_session.get("id")

    try:
        data = json.loads(request.body)
        object_class = data.get("object_class", "")
        confidence = data.get("confidence", 0)
        crop_image = data.get("crop_image", "")

        if not object_class or not crop_image:
            logger.warning("Crop save request missing data | user=%s", user_id)
            return JsonResponse({"error": "Missing data"}, status=400)

        saved = data_manager.save_crop(user_id, object_class, confidence, crop_image)
        logger.info("Detection crop saved | user=%s | class=%s", user_id, object_class)
        return JsonResponse({"success": True, "data": saved})

    except json.JSONDecodeError:
        logger.warning("Invalid JSON in crop save request | user=%s", user_id)
        return JsonResponse({"error": "Invalid request data"}, status=400)

    except Exception:
        logger.exception("Failed to save detection crop | user=%s", user_id)
        return JsonResponse({"error": "Failed to save crop"}, status=500)


@csrf_exempt
@require_http_methods(["GET"])
def get_latest_crop(request):
    user_session = get_user_from_session(request)

    if not user_session:
        logger.warning("Unauthorized crop retrieval request")
        return JsonResponse({"error": "Unauthorized"}, status=401)

    object_class = request.GET.get("object_class", "")
    if not object_class:
        return JsonResponse({"error": "Missing object_class"}, status=400)

    user_id = user_session.get("users", {}).get("id") or user_session.get("id")

    try:
        crop = data_manager.get_latest_crop(user_id)
        if not crop:
            return JsonResponse({"success": False, "error": "No crop found for this class"})

        return JsonResponse({
            "success": True,
            "object_class": crop["object_class"],
            "confidence": crop["confidence"],
            "crop_image": crop["crop_image"],
        })

    except Exception:
        logger.exception("Failed to retrieve latest crop | user=%s | class=%s", user_id, object_class)
        return JsonResponse({"error": "Failed to retrieve crop"}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def clear_detection_history(request):
    user_session = get_user_from_session(request)

    if not user_session:
        logger.warning("Unauthorized history clear request")
        return JsonResponse({"error": "Unauthorized"}, status=401)

    user_id = user_session.get("users", {}).get("id") or user_session.get("id")

    try:
        data_manager.clear_history(user_id)
        logger.info("Detection history cleared | user=%s", user_id)
        return JsonResponse({"success": True})

    except Exception:
        logger.exception("Failed to clear detection history | user=%s", user_id)
        return JsonResponse({"error": "Failed to clear detection history"}, status=500)


# TripoSR 3D mesh generation
@csrf_exempt
@require_http_methods(["POST"])
@ratelimit(key=session_or_ip, rate="5/m", method="POST", block=False)
def generate_3d_mesh(request):
    if getattr(request, "limited", False):
        logger.warning("TripoSR mesh rate limit hit")
        return too_many_requests_json()

    user_session = get_user_from_session(request)
    user_id = (
        user_session["users"]["id"]
        if user_session
        else None
    )

    logger.info(
        "TripoSR mesh request received | user=%s",
        user_id,
    )

    try:
        data = json.loads(request.body)
        image_base64 = data.get("image_base64", "")

        if not image_base64:
            logger.warning(
                "TripoSR request missing image | user=%s",
                user_id,
            )

            return JsonResponse(
                {"error": "No image provided"},
                status=400,
            )

        triposr_url = settings.TRIPOSR_API_URL

        if not triposr_url:
            logger.error("TripoSR API URL is not configured")

            return JsonResponse(
                {
                    "error": (
                        "TripoSR not configured. "
                        "Start the Colab notebook first."
                    )
                },
                status=500,
            )

        response = http_requests.post(
            triposr_url.rstrip("/") + "/generate",
            json={"image_base64": image_base64},
            timeout=120,
            headers={
                "Content-Type": "application/json",
                "ngrok-skip-browser-warning": "true",
            },
        )

        if response.status_code != 200:
            logger.error(
                "TripoSR returned error | status=%s | user=%s",
                response.status_code,
                user_id,
            )

            return JsonResponse(
                {
                    "error": "TripoSR service returned an error."
                },
                status=500,
            )

        glb_path = os.path.join(
            settings.BASE_DIR,
            "static",
            "temp_models",
        )

        os.makedirs(glb_path, exist_ok=True)

        filename = f"model_{int(time.time())}.glb"
        filepath = os.path.join(
            glb_path,
            filename,
        )

        with open(filepath, "wb") as file:
            file.write(response.content)

        glb_url = f"/static/temp_models/{filename}"

        logger.info(
            "TripoSR mesh generated | user=%s | file=%s",
            user_id,
            filename,
        )

        return JsonResponse({
            "success": True,
            "glb_url": glb_url,
            "method": "triposr",
        })

    except json.JSONDecodeError:
        logger.warning(
            "Invalid JSON in TripoSR request | user=%s",
            user_id,
        )

        return JsonResponse(
            {"error": "Invalid request data"},
            status=400,
        )

    except http_requests.exceptions.Timeout:
        logger.exception(
            "TripoSR request timed out | user=%s",
            user_id,
        )

        return JsonResponse(
            {
                "error": (
                    "TripoSR timed out. "
                    "The Colab notebook may have disconnected."
                )
            },
            status=500,
        )

    except http_requests.exceptions.ConnectionError:
        logger.exception(
            "Could not connect to TripoSR | user=%s",
            user_id,
        )

        return JsonResponse(
            {
                "error": (
                    "Cannot reach TripoSR. "
                    "Make sure the Colab notebook is running."
                )
            },
            status=500,
        )

    except Exception:
        logger.exception(
            "TripoSR mesh generation failed | user=%s",
            user_id,
        )

        return JsonResponse(
            {"error": "3D mesh generation failed"},
            status=500,
        )