# backend/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path("health/", views.health_check, name="health_check"),
    path("detect/", views.detect_object, name="detect_object"),
]

