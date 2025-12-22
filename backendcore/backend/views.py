# # backend/views.py

# from django.http import JsonResponse
# from django.shortcuts import render
# from django.views.decorators.csrf import csrf_exempt


# # -----------------------------
# # API VIEWS (KEEP AS-IS)
# # -----------------------------

# @csrf_exempt
# def health_check(request):
#     return JsonResponse({
#         "status": "ok",
#         "message": "SmartVision backend is alive 🔥",
#     })


# @csrf_exempt
# def detect_object(request):

#     if request.method != "POST":
#         return JsonResponse(
#             {"error": "Use POST with an image file under 'image' field."},
#             status=400
#         )

#     image_file = request.FILES.get("image", None)

#     if image_file is None:
#         return JsonResponse(
#             {"error": "No image uploaded. Send as form-data with key 'image'."},
#             status=400
#         )

#     return JsonResponse({
#         "message": "Image received successfully ✅",
#         "filename": image_file.name,
#         "size_bytes": image_file.size,
#     })


# # -----------------------------
# # HTML PAGE VIEWS (NEW)
# # -----------------------------

# def landing_page(request):
#     return render(request, "landingpage.html")


# def login_page(request):
#     return render(request, "loginpage.html")


# def signup_page(request):
#     return render(request, "signuppage.html")


from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.contrib.auth.hashers import make_password, check_password
from backend.supabase_client import supabase_admin, supabase_public
import uuid
import secrets
from datetime import datetime, timedelta
import json


# ============================================
# HELPER FUNCTIONS
# ============================================

def get_client_ip(request):
    """Get client IP address"""
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip


def create_session(user_id, request):
    """Create a new user session"""
    session_token = secrets.token_urlsafe(32)
    expires_at = (datetime.now() + timedelta(days=7)).isoformat()
    
    session_data = {
        'user_id': str(user_id),
        'session_token': session_token,
        'ip_address': get_client_ip(request),
        'user_agent': request.META.get('HTTP_USER_AGENT', ''),
        'expires_at': expires_at,
        'is_active': True
    }
    
    # Insert session into Supabase
    response = supabase_admin.table('user_sessions').insert(session_data).execute()
    
    return session_token


def get_user_from_session(request):
    """Get user from session token"""
    session_token = request.session.get('session_token')
    
    if not session_token:
        return None
    
    # Get session from Supabase
    response = supabase_admin.table('user_sessions')\
        .select('*, users(*)')\
        .eq('session_token', session_token)\
        .eq('is_active', True)\
        .single()\
        .execute()
    
    if not response.data:
        return None
    
    # Check if session is expired
    expires_at = datetime.fromisoformat(response.data['expires_at'].replace('Z', '+00:00'))
    if datetime.now(expires_at.tzinfo) > expires_at:
        return None
    
    return response.data


def login_required(view_func):
    """Decorator to require login"""
    def wrapper(request, *args, **kwargs):
        user_session = get_user_from_session(request)
        if not user_session:
            return redirect('login')
        request.user_session = user_session
        return view_func(request, *args, **kwargs)
    return wrapper


# ============================================
# PUBLIC VIEWS
# ============================================

def landing_page(request):
    """Landing page view"""
    # If user is already logged in, redirect to dashboard
    if get_user_from_session(request):
        return redirect('dashboard')
    return render(request, 'landingpage.html')


def login_page(request):
    """Login page view"""
    # If user is already logged in, redirect to dashboard
    if get_user_from_session(request):
        return redirect('dashboard')
    
    if request.method == 'GET':
        return render(request, 'loginpage.html')
    
    # Handle POST request (login form submission)
    if request.method == 'POST':
        email_or_username = request.POST.get('email_or_username')
        password = request.POST.get('password')
        
        if not email_or_username or not password:
            return render(request, 'loginpage.html', {
                'error': 'Please provide both email/username and password'
            })
        
        # Try to find user by email or username
        user_response = supabase_admin.table('users')\
            .select('*')\
            .or_(f'email.eq.{email_or_username},username.eq.{email_or_username}')\
            .execute()
        
        if not user_response.data or len(user_response.data) == 0:
            return render(request, 'loginpage.html', {
                'error': 'Invalid credentials'
            })
        
        user = user_response.data[0]
        
        # Verify password
        if not check_password(password, user['password_hash']):
            return render(request, 'loginpage.html', {
                'error': 'Invalid credentials'
            })
        
        # Check if user is active
        if not user.get('is_active', True):
            return render(request, 'loginpage.html', {
                'error': 'Your account has been deactivated'
            })
        
        # Create session
        session_token = create_session(user['id'], request)
        
        # Update last login
        supabase_admin.table('users')\
            .update({'last_login': datetime.now().isoformat()})\
            .eq('id', user['id'])\
            .execute()
        
        # Store session token in Django session
        request.session['session_token'] = session_token
        
        # Redirect to dashboard
        return redirect('dashboard')


def signup_page(request):
    """Signup page view"""
    # If user is already logged in, redirect to dashboard
    if get_user_from_session(request):
        return redirect('dashboard')
    
    if request.method == 'GET':
        return render(request, 'signuppage.html')
    
    # Handle POST request (signup form submission)
    if request.method == 'POST':
        full_name = request.POST.get('full_name')
        email = request.POST.get('email')
        password = request.POST.get('password')
        confirm_password = request.POST.get('confirm_password')
        
        # Validation
        if not all([full_name, email, password, confirm_password]):
            return render(request, 'signuppage.html', {
                'error': 'All fields are required'
            })
        
        if password != confirm_password:
            return render(request, 'signuppage.html', {
                'error': 'Passwords do not match'
            })
        
        if len(password) < 8:
            return render(request, 'signuppage.html', {
                'error': 'Password must be at least 8 characters long'
            })
        
        # Generate username from email
        username = email.split('@')[0]
        
        # Check if email already exists
        existing_user = supabase_admin.table('users')\
            .select('id')\
            .eq('email', email)\
            .execute()
        
        if existing_user.data and len(existing_user.data) > 0:
            return render(request, 'signuppage.html', {
                'error': 'Email already registered'
            })
        
        # Create user
        user_data = {
            'full_name': full_name,
            'email': email,
            'username': username,
            'password_hash': make_password(password),
            'is_active': True,
            'is_verified': False
        }
        
        try:
            response = supabase_admin.table('users').insert(user_data).execute()
            user = response.data[0]
            
            # Create session
            session_token = create_session(user['id'], request)
            request.session['session_token'] = session_token
            
            # Redirect to dashboard
            return redirect('dashboard')
        
        except Exception as e:
            return render(request, 'signuppage.html', {
                'error': f'Error creating account: {str(e)}'
            })


def logout_view(request):
    """Logout view"""
    session_token = request.session.get('session_token')
    
    if session_token:
        # Deactivate session in Supabase
        supabase_admin.table('user_sessions')\
            .update({'is_active': False})\
            .eq('session_token', session_token)\
            .execute()
    
    # Clear Django session
    request.session.flush()
    
    return redirect('login')


# ============================================
# PROTECTED VIEWS (Require Login)
# ============================================

@login_required
def dashboard(request):
    """Dashboard view - shows user stats and recent activity"""
    user_data = request.user_session['users']
    user_id = user_data['id']
    
    # Get user statistics
    # Total detection sessions
    sessions_response = supabase_admin.table('detection_sessions')\
        .select('id, session_type, started_at, total_detections')\
        .eq('user_id', user_id)\
        .order('started_at', desc=True)\
        .limit(5)\
        .execute()
    
    # Total detections
    detections_response = supabase_admin.table('detection_results')\
        .select('id, total_objects_detected, created_at')\
        .eq('user_id', user_id)\
        .order('created_at', desc=True)\
        .limit(10)\
        .execute()
    
    # Total LLM queries
    queries_response = supabase_admin.table('llm_queries')\
        .select('id')\
        .eq('user_id', user_id)\
        .execute()
    
    context = {
        'user': user_data,
        'total_sessions': len(sessions_response.data) if sessions_response.data else 0,
        'total_detections': len(detections_response.data) if detections_response.data else 0,
        'total_queries': len(queries_response.data) if queries_response.data else 0,
        'recent_sessions': sessions_response.data if sessions_response.data else [],
        'recent_detections': detections_response.data if detections_response.data else []
    }
    
    return render(request, 'dashboard.html', context)


@login_required
def camera_feed(request):
    """Camera feed view - real-time object detection"""
    user_data = request.user_session['users']
    
    context = {
        'user': user_data,
    }
    
    return render(request, 'camera_feed.html', context)


# ============================================
# API ENDPOINTS (for AJAX/WebSocket)
# ============================================

@csrf_exempt
@require_http_methods(["GET"])
def get_user_stats(request):
    """API endpoint to get user statistics"""
    user_session = get_user_from_session(request)
    
    if not user_session:
        return JsonResponse({'error': 'Unauthorized'}, status=401)
    
    user_id = user_session['users']['id']
    
    # Get statistics from Supabase
    sessions_count = supabase_admin.table('detection_sessions')\
        .select('id', count='exact')\
        .eq('user_id', user_id)\
        .execute()
    
    detections_count = supabase_admin.table('detection_results')\
        .select('id', count='exact')\
        .eq('user_id', user_id)\
        .execute()
    
    queries_count = supabase_admin.table('llm_queries')\
        .select('id', count='exact')\
        .eq('user_id', user_id)\
        .execute()
    
    return JsonResponse({
        'total_sessions': sessions_count.count if hasattr(sessions_count, 'count') else 0,
        'total_detections': detections_count.count if hasattr(detections_count, 'count') else 0,
        'total_queries': queries_count.count if hasattr(queries_count, 'count') else 0
    })