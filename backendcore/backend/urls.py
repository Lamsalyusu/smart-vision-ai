from django.urls import path
from . import views

urlpatterns = [
    # Public pages
    path('', views.landing_page, name='landing'),
    path('login/', views.login_page, name='login'),
    path('signup/', views.signup_page, name='signup'),
    path('logout/', views.logout_view, name='logout'),
    
    # Protected pages (require login)
    path('dashboard/', views.dashboard, name='dashboard'),
    path('camera/', views.camera_feed, name='camera_feed'),
    path('model-viewer/', views.model_viewer, name='model_viewer'),
    
    # API endpoints
    path('api/user/stats/', views.get_user_stats, name='user_stats'),
    path('forgot-password/', views.forgot_password, name='forgot_password'),
    # path('reset-password/<str:token>/', views.reset_password, name='reset_passwords'),
]