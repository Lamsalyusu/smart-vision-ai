from supabase import create_client
from django.conf import settings

supabase_admin = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY)
# OPTIONAL: for public operations (usually frontend uses anon)
supabase_public = create_client(settings.SUPABASE_URL, settings.SUPABASE_ANON_KEY)

