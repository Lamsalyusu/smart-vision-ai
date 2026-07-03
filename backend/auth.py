import jwt
from jwt.exceptions import PyJWTError as JWTError
from django.conf import settings

def get_user_from_request(request):
    authorization = request.headers.get("Authorization", "")

    if not authorization.startswith("Bearer "):
        return None, "Missing Bearer token"

    token = authorization.removeprefix("Bearer ").strip()

    if not token:
        return None, "Missing Bearer token"

    try:
        payload = jwt.decode(
            token,
            settings.SUPABASE_JWT_SECRET,
            algorithms=["HS256"],
            audience="authenticated",
        )
        return payload, None

    except JWTError:
        return None, "Invalid token"

