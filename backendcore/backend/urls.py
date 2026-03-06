from django.urls import path
from . import views

urlpatterns = [
    # Public pages
    path('', views.landing_page, name='landing'),
    path('login/', views.login_page, name='login'),
    path('signup/', views.signup_page, name='signup'),
    path('logout/', views.logout_view, name='logout'),
    path('forgot-password/', views.forgot_password, name='forgot_password'),
    path('reset-password/<str:token>/', views.reset_password, name='reset_password'),

    # OAuth
    path('auth/google/', views.google_login, name='google_login'),
    path('auth/google/callback/', views.google_callback, name='google_callback'),
    path('auth/github/', views.github_login, name='github_login'),
    path('auth/github/callback/', views.github_callback, name='github_callback'),

    # Protected pages
    path('dashboard/', views.dashboard, name='dashboard'),
    path('camera/', views.camera_feed, name='camera_feed'),
    path('model-viewer/', views.model_viewer, name='model_viewer'),

    # API endpoints
    path('api/user/stats/', views.get_user_stats, name='user_stats'),
    path('api/detect/', views.detect_objects, name='detect'),
    path('api/3d-model/<str:class_name>/', views.get_3d_model, name='get_3d_model'),
    path('api/chat/', views.llm_chat, name='llm_chat'),
    # path('api/generate-3d/', views.generate_3d_view, name='generate_3d'),
]