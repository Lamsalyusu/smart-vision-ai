"""
URL configuration for backendcore project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/6.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from django.contrib import admin
from django.urls import path, include
from django.http import JsonResponse   # 👈 IMPORTANT

# 👇 ADD THIS FUNCTION (you forgot it)
def home(request):
    return JsonResponse({
        "message": "Backend running successfully 🚀",
        "api_base": "/api/"
    })


urlpatterns = [
    path('', home, name="home"), 
    path('admin/', admin.site.urls),

    # All API endpoints from your app will start with /api/
    path('api/', include('backend.urls')),
    # path('api/', include('backend.urls')),

]

