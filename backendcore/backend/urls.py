# backend/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path("", views.landing_page, name="landing"),
    path("login/", views.login_page, name="login"),
    path("signup/", views.signup_page, name="signup"),

    # API endpoints (keep separate)
    # path("health/", views.health_check, name="health_check"),
    # path("detect/", views.detect_object, name="detect_object"),
]


