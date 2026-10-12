from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# Base configuration
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

def get_env(name, default=None):
    return os.getenv(name, default)

SECRET_KEY = get_env("SECRET_KEY", "django-insecure-smartvision-ai-evaluation-secret-key-2026")
OFFLINE_MODE = get_env("OFFLINE_MODE", "False").lower() == "true"

DEBUG = get_env("DJANGO_DEBUG", "False").lower() == "true"

ALLOWED_HOSTS = [
    host.strip()
    for host in get_env("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")
    if host.strip()
]

SUPABASE_URL = get_env("SUPABASE_URL")
SUPABASE_ANON_KEY = get_env("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_ROLE_KEY = get_env("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_JWT_SECRET = get_env("SUPABASE_JWT_SECRET")

GROQ_API_KEY = get_env("GROQ_API_KEY")
AIML_API_KEY = get_env("AIML_API_KEY")
IMGBB_API_KEY = get_env("IMGBB_API_KEY")
TRIPOSR_API_URL = get_env("TRIPOSR_API_URL")

# Django applications
INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "backend",
]

# Middleware
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]


ROOT_URLCONF = "backendcore.urls"
WSGI_APPLICATION = "backendcore.wsgi.application"


# Templates
TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": ("django.contrib.auth.password_validation.""UserAttributeSimilarityValidator"),
    },
    {
        "NAME": ("django.contrib.auth.password_validation.""MinimumLengthValidator"),
    },
    {
        "NAME": ("django.contrib.auth.password_validation.""CommonPasswordValidator"),
    },
    {
        "NAME": ("django.contrib.auth.password_validation.""NumericPasswordValidator"),
    },
]

# Sessions
SESSION_ENGINE = "django.contrib.sessions.backends.db"
SESSION_COOKIE_AGE = 604800
SESSION_SAVE_EVERY_REQUEST = False

# Internationalization
LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"

USE_I18N = True
USE_TZ = True

# Static files
STATIC_URL = "static/"
STATICFILES_DIRS = [BASE_DIR / "static"]
STATIC_ROOT = BASE_DIR / "staticfiles"  

# Email
EMAIL_BACKEND = "django.core.mail.backends.smtp.EmailBackend"
EMAIL_HOST = "smtp.gmail.com"
EMAIL_PORT = 587
EMAIL_USE_TLS = True

EMAIL_HOST_USER = get_env("EMAIL_HOST_USER")
EMAIL_HOST_PASSWORD = get_env("EMAIL_HOST_PASSWORD")

DEFAULT_FROM_EMAIL = EMAIL_HOST_USER

# Google OAuth
GOOGLE_CLIENT_ID = get_env("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = get_env("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = get_env("GOOGLE_REDIRECT_URI","http://127.0.0.1:8000/auth/google/callback/")

# GitHub OAuth
GITHUB_CLIENT_ID = get_env("GITHUB_CLIENT_ID")
GITHUB_CLIENT_SECRET = get_env("GITHUB_CLIENT_SECRET")
GITHUB_REDIRECT_URI = get_env("GITHUB_REDIRECT_URI","http://127.0.0.1:8000/auth/github/callback/")

# Django defaults
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": ("{levelname} {asctime} ""[{name}] {message}"),
            "style": "{",},
    },
    "handlers": {
        "console": {"class": "logging.StreamHandler","formatter": "standard",
        },
        "file": {"class": "logging.FileHandler","filename": LOG_DIR / "django.log","formatter": "standard",
        },
    },
    "loggers": {
        "django": {"handlers": ["console", "file"],"level": "INFO","propagate": False,
        },
        "backend.views": {"handlers": ["console", "file"],"level": "INFO","propagate": False,
        },
    },
}