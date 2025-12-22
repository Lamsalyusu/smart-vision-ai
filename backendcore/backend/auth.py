from jose import jwt, JWTError
from django.conf import settings

def get_user_from_request(request):
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return None, "Missing Bearer token"

    token = auth.split(" ", 1)[1].strip()

    try:
        payload = jwt.decode(
            token,
            settings.SUPABASE_JWT_SECRET,
            algorithms=["HS256"],
            audience="authenticated",
        )
        # payload contains: sub (user id), email, role, exp...
        return payload, None

    except JWTError as e:
        return None, f"Invalid token: {str(e)}"
