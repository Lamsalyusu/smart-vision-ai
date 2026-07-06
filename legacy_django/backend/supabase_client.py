import logging
from django.conf import settings
from supabase import create_client

logger = logging.getLogger(__name__)

supabase_admin = None
supabase_public = None

try:
    if settings.SUPABASE_URL and settings.SUPABASE_SERVICE_ROLE_KEY:
        supabase_admin = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY)
except Exception as exc:
    logger.warning("Supabase admin client not initialized: %s", exc)

try:
    if settings.SUPABASE_URL and settings.SUPABASE_ANON_KEY:
        supabase_public = create_client(settings.SUPABASE_URL, settings.SUPABASE_ANON_KEY)
except Exception as exc:
    logger.warning("Supabase public client not initialized: %s", exc)
