# backend/views.py

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt


# -----------------------------
# API VIEWS (KEEP AS-IS)
# -----------------------------

@csrf_exempt
def health_check(request):
    return JsonResponse({
        "status": "ok",
        "message": "SmartVision backend is alive 🔥",
    })


@csrf_exempt
def detect_object(request):

    if request.method != "POST":
        return JsonResponse(
            {"error": "Use POST with an image file under 'image' field."},
            status=400
        )

    image_file = request.FILES.get("image", None)

    if image_file is None:
        return JsonResponse(
            {"error": "No image uploaded. Send as form-data with key 'image'."},
            status=400
        )

    return JsonResponse({
        "message": "Image received successfully ✅",
        "filename": image_file.name,
        "size_bytes": image_file.size,
    })


# -----------------------------
# HTML PAGE VIEWS (NEW)
# -----------------------------

def landing_page(request):
    return render(request, "landingpage.html")


def login_page(request):
    return render(request, "loginpage.html")


def signup_page(request):
    return render(request, "signuppage.html")
