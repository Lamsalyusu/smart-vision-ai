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
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
import json
import base64
import os
import requests as http_requests   
from django.core.mail import send_mail
from django.conf import settings
from urllib.parse import urlencode
from django.shortcuts import render, redirect
from django.contrib import messages
import secrets
from datetime import datetime, timedelta

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

# PUBLIC VIEWS


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



@login_required
def model_viewer(request):
    """3D Model Viewer page"""
    return render(request, 'model_viewer.html')

# API ENDPOINTS (for AJAX/WebSocket)


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




# def forgot_password(request):
#     if request.method == 'POST':
#         email = request.POST.get('email')
        
#         # Check if user exists
#         try:
#             user = User.objects.get(email=email)
            
#             # Generate reset token
#             token = secrets.token_urlsafe(32)
#             expiry = datetime.now() + timedelta(hours=1)
            
#             # Save token to database (you'll need a PasswordReset model)
#             # PasswordReset.objects.create(user=user, token=token, expiry=expiry)
            
#             # Send email (implement this later)
#             # send_password_reset_email(user.email, token)
            
#             # Redirect with success
#             return redirect('/forgot-password/?success=true')
            
#         except User.DoesNotExist:
#             return render(request, 'forgot_password.html', {
#                 'error': 'No account found with this email address.'
#             })
    
#     return render(request, 'forgot_password.html')


from .ml_model.inference import detector
# from .supabase_client import supabase 


@csrf_exempt
def detect_objects(request):
    """
    API endpoint for object detection
    POST /api/detect/
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'POST method required'}, status=405)
    
    try:
        # Handle file upload
        if 'image' in request.FILES:
            image_file = request.FILES['image']
            
            # Save temporarily
            temp_path = default_storage.save(f'temp/{image_file.name}', image_file)
            full_path = os.path.join(settings.MEDIA_ROOT, temp_path)
            
            # Run detection
            detections = detector.detect_from_file(full_path)
            
            # Cleanup
            os.remove(full_path)
        
        # Handle base64 image
        elif 'image_base64' in request.POST:
            image_data = request.POST['image_base64']
            
            # Decode base64
            if 'base64,' in image_data:
                image_data = image_data.split('base64,')[1]
            
            image_bytes = base64.b64decode(image_data)
            detections = detector.detect_from_bytes(image_bytes)
        
        else:
            return JsonResponse({'error': 'No image provided'}, status=400)
        
        # Save to Supabase (optional)
        try:
            supabase_data = {
                'detections': detections,
                'total': len(detections)
            }
            supabase.table('detections').insert(supabase_data).execute()
        except Exception as e:
            print(f"Supabase error: {e}")
        
        return JsonResponse({
            'success': True,
            'total_detections': len(detections),
            'detections': detections
        })
    
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


def webcam_feed(request):
    """Render webcam detection page"""
    return render(request, 'camera_feed.html')


def model_viewer(request):
    """Render 3D model viewer page"""
    context = {
        'detected_class': request.GET.get('class', 'sunglasses')
    }
    return render(request, 'model_viewer_v2.html', context)


def get_3d_model(request, class_name):
    """Get 3D model URL for a class"""
    
    CLASS_TO_MODEL = {
        "sunglasses": "sunglasses.glb",
        "knife": "knife.glb",
        "water_bottle": "water_bottle.glb",
        "pen": "pen.glb",
        "chair": "chair.glb",
        "human_face": "human_face.glb",
        "mobile_phone": "mobile_phone.glb",
        "helmet": "helmet.glb",
        "fire": "fire.glb",
        "can": "can.glb"
    }
    
    if class_name not in CLASS_TO_MODEL:
        return JsonResponse({'error': 'Class not found'}, status=404)
    
    model_url = f"/static/models/{CLASS_TO_MODEL[class_name]}"
    
    return JsonResponse({
        'class': class_name,
        'model_url': model_url,
        'format': 'glb'
    })
from django.conf import settings

from groq import Groq

@csrf_exempt
@require_http_methods(["POST"])
def llm_chat(request):
    try:
        data = json.loads(request.body)
        user_message = data.get('message', '')
        detections = data.get('detections', [])

        detection_context = "No objects currently detected."
        if detections:
            objects = [f"{d['class']} ({d['confidence']*100:.0f}%)" for d in detections]
            detection_context = f"Currently detected objects: {', '.join(objects)}."

        client = Groq(api_key=settings.GROQ_API_KEY)
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{
                "role": "user",
                "content": f"You are SmartVision AI assistant helping users understand objects detected by a real-time camera. {detection_context} Keep responses concise.\n\nUser: {user_message}"
            }]
        )

        ai_response = response.choices[0].message.content

        # Save query to Supabase
        user_session = get_user_from_session(request)
        if user_session:
            try:
                supabase_admin.table('llm_queries').insert({
                    'user_id': str(user_session['users']['id']),
                    'message': user_message[:1000],
                    'response': ai_response[:2000],
                    'detection_context': detection_context[:500],
                }).execute()
            except Exception as e:
                print(f"Save query error: {e}")

        return JsonResponse({
            'success': True,
            'response': ai_response
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def generate_3d_view(request):
    """
    API endpoint: Takes a cropped detected object image,
    runs MiDaS depth estimation, returns 3D point cloud data.
    
    POST /api/generate-3d/
    Body: FormData with 'image' file OR JSON with 'image_base64'
    
    Returns: { positions: [...], colors: [...], point_count: int }
    """
    try:
        # Get image data
        if 'image' in request.FILES:
            image_bytes = request.FILES['image'].read()
        
        elif request.content_type == 'application/json':
            data = json.loads(request.body)
            image_data = data.get('image_base64', '')
            
            if 'base64,' in image_data:
                image_data = image_data.split('base64,')[1]
            
            image_bytes = base64.b64decode(image_data)
        
        else:
            return JsonResponse({'error': 'No image provided'}, status=400)
        
        if len(image_bytes) == 0:
            return JsonResponse({'error': 'Empty image data'}, status=400)
        
        # Run depth estimation and generate point cloud
        from backend.ml_model.depth_estimation import depth_estimator
        
        result = depth_estimator.generate_point_cloud(
            image_bytes,
            max_points=12000,   # Good balance of detail vs performance
            depth_scale=2.0     # How "deep" the 3D effect is
        )
        
        return JsonResponse({
            'success': True,
            'positions': result['positions'],
            'colors': result['colors'],
            'point_count': result['point_count'],
            'width': result['original_width'],
            'height': result['original_height']
        })
    
    except ImportError as e:
        return JsonResponse({
            'error': 'MiDaS not installed. Run: pip install torch torchvision timm',
            'detail': str(e)
        }, status=500)
    
    except Exception as e:
        print(f"3D generation error: {e}")
        return JsonResponse({'error': str(e)}, status=500)

    
# OAUTH: GOOGLE

def google_login(request):
    """Redirect user to Google OAuth consent screen"""
    params = urlencode({
        'client_id': settings.GOOGLE_CLIENT_ID,
        'redirect_uri': settings.GOOGLE_REDIRECT_URI,
        'response_type': 'code',
        'scope': 'email profile',
        'access_type': 'offline',
        'prompt': 'select_account',
    })
    return redirect('https://accounts.google.com/o/oauth2/v2/auth?' + params)


def google_callback(request):
    """Handle Google OAuth callback"""
    code = request.GET.get('code')
    error = request.GET.get('error')

    if error or not code:
        return render(request, 'loginpage.html', {
            'error': 'Google login was cancelled or failed.'
        })

    # Exchange code for access token
    token_resp = http_requests.post('https://oauth2.googleapis.com/token', data={
        'code': code,
        'client_id': settings.GOOGLE_CLIENT_ID,
        'client_secret': settings.GOOGLE_CLIENT_SECRET,
        'redirect_uri': settings.GOOGLE_REDIRECT_URI,
        'grant_type': 'authorization_code',
    })

    if token_resp.status_code != 200:
        return render(request, 'loginpage.html', {
            'error': 'Failed to get token from Google.'
        })

    access_token = token_resp.json().get('access_token')

    # Get user info from Google
    user_resp = http_requests.get('https://www.googleapis.com/oauth2/v2/userinfo', headers={
        'Authorization': 'Bearer ' + access_token
    })

    if user_resp.status_code != 200:
        return render(request, 'loginpage.html', {
            'error': 'Failed to get user info from Google.'
        })

    google_user = user_resp.json()
    email = google_user.get('email')
    name = google_user.get('name', email.split('@')[0])

    # Find or create user in Supabase
    user = _find_or_create_oauth_user(email, name, 'google')
    if not user:
        return render(request, 'loginpage.html', {
            'error': 'Failed to create account.'
        })

    # Create session and redirect
    session_token = create_session(user['id'], request)
    request.session['session_token'] = session_token
    return redirect('dashboard')


# ============================================
# OAUTH: GITHUB
# ============================================

def github_login(request):
    """Redirect user to GitHub OAuth"""
    params = urlencode({
        'client_id': settings.GITHUB_CLIENT_ID,
        'redirect_uri': settings.GITHUB_REDIRECT_URI,
        'scope': 'user:email',
    })
    return redirect('https://github.com/login/oauth/authorize?' + params)


def github_callback(request):
    """Handle GitHub OAuth callback"""
    code = request.GET.get('code')
    error = request.GET.get('error')

    if error or not code:
        return render(request, 'loginpage.html', {
            'error': 'GitHub login was cancelled or failed.'
        })

    # Exchange code for access token
    token_resp = http_requests.post('https://github.com/login/oauth/access_token', data={
        'code': code,
        'client_id': settings.GITHUB_CLIENT_ID,
        'client_secret': settings.GITHUB_CLIENT_SECRET,
        'redirect_uri': settings.GITHUB_REDIRECT_URI,
    }, headers={
        'Accept': 'application/json'
    })

    if token_resp.status_code != 200:
        return render(request, 'loginpage.html', {
            'error': 'Failed to get token from GitHub.'
        })

    access_token = token_resp.json().get('access_token')

    # Get user info
    user_resp = http_requests.get('https://api.github.com/user', headers={
        'Authorization': 'Bearer ' + access_token
    })

    if user_resp.status_code != 200:
        return render(request, 'loginpage.html', {
            'error': 'Failed to get user info from GitHub.'
        })

    github_user = user_resp.json()

    # GitHub might not return email in profile, fetch from emails endpoint
    email = github_user.get('email')
    if not email:
        emails_resp = http_requests.get('https://api.github.com/user/emails', headers={
            'Authorization': 'Bearer ' + access_token
        })
        if emails_resp.status_code == 200:
            for e in emails_resp.json():
                if e.get('primary') and e.get('verified'):
                    email = e.get('email')
                    break

    if not email:
        return render(request, 'loginpage.html', {
            'error': 'Could not get email from GitHub. Make sure your email is public or verified.'
        })

    name = github_user.get('name') or github_user.get('login', email.split('@')[0])

    # Find or create user
    user = _find_or_create_oauth_user(email, name, 'github')
    if not user:
        return render(request, 'loginpage.html', {
            'error': 'Failed to create account.'
        })

    session_token = create_session(user['id'], request)
    request.session['session_token'] = session_token
    return redirect('dashboard')

# OAUTH HELPER

def _find_or_create_oauth_user(email, full_name, provider):
    """Find existing user by email or create a new one for OAuth login"""
    try:
        # Check if user exists
        existing = supabase_admin.table('users')\
            .select('*')\
            .eq('email', email)\
            .execute()

        if existing.data and len(existing.data) > 0:
            return existing.data[0]

        # Create new user (no password needed for OAuth)
        username = email.split('@')[0]
        user_data = {
            'full_name': full_name,
            'email': email,
            'username': username,
            'password_hash': '',  # OAuth users don't have a password
            'is_active': True,
            'is_verified': True,  # OAuth emails are already verified
            'auth_provider': provider,
        }

        resp = supabase_admin.table('users').insert(user_data).execute()
        if resp.data and len(resp.data) > 0:
            return resp.data[0]
        return None

    except Exception as e:
        print(f"OAuth user error: {e}")
        return None

# FORGOT PASSWORD (FIXED - uses Supabase)

def forgot_password(request):
    """Forgot password - sends reset email"""
    if request.method == 'GET':
        return render(request, 'forgot_password.html')

    email = request.POST.get('email', '').strip()

    if not email:
        return render(request, 'forgot_password.html', {
            'error': 'Please enter your email address.'
        })

    # Check if user exists in Supabase
    user_resp = supabase_admin.table('users')\
        .select('id, email, full_name')\
        .eq('email', email)\
        .execute()

    if not user_resp.data or len(user_resp.data) == 0:
        # Don't reveal if email exists or not (security)
        return redirect('/forgot-password/?success=true')

    user = user_resp.data[0]

    # Generate reset token
    token = secrets.token_urlsafe(32)
    expires_at = (datetime.now() + timedelta(hours=1)).isoformat()

    # Save token to Supabase
    try:
        supabase_admin.table('password_resets').insert({
            'user_id': str(user['id']),
            'token': token,
            'expires_at': expires_at,
            'used': False,
        }).execute()
    except Exception as e:
        print(f"Error saving reset token: {e}")
        return render(request, 'forgot_password.html', {
            'error': 'Something went wrong. Please try again.'
        })

    # Send reset email
    reset_url = request.build_absolute_uri(f'/reset-password/{token}/')

    try:
        send_mail(
            subject='Smart Vision - Password Reset',
            message=f'Hi {user["full_name"]},\n\n'
                    f'You requested a password reset. Click the link below:\n\n'
                    f'{reset_url}\n\n'
                    f'This link expires in 1 hour.\n\n'
                    f'If you did not request this, ignore this email.\n\n'
                    f'- Smart Vision AI',
            from_email=settings.DEFAULT_FROM_EMAIL,
            recipient_list=[email],
            fail_silently=False,
        )
    except Exception as e:
        print(f"Email send error: {e}")
        return render(request, 'forgot_password.html', {
            'error': 'Could not send email. Please try again later.'
        })

    return redirect('/forgot-password/?success=true')

# RESET PASSWORD


def reset_password(request, token):
    """Reset password using token from email"""
    # Verify token
    token_resp = supabase_admin.table('password_resets')\
        .select('*, users(id, email, full_name)')\
        .eq('token', token)\
        .eq('used', False)\
        .execute()

    if not token_resp.data or len(token_resp.data) == 0:
        return render(request, 'reset_password.html', {
            'error': 'Invalid or expired reset link.',
            'token_valid': False,
        })

    reset_data = token_resp.data[0]

    # Check expiry
    expires_at = datetime.fromisoformat(reset_data['expires_at'].replace('Z', '+00:00'))
    if datetime.now(expires_at.tzinfo) > expires_at:
        return render(request, 'reset_password.html', {
            'error': 'This reset link has expired. Please request a new one.',
            'token_valid': False,
        })

    if request.method == 'GET':
        return render(request, 'reset_password.html', {
            'token': token,
            'token_valid': True,
            'email': reset_data['users']['email'],
        })

    # Handle POST - update password
    password = request.POST.get('password', '')
    confirm_password = request.POST.get('confirm_password', '')

    if not password or len(password) < 8:
        return render(request, 'reset_password.html', {
            'token': token,
            'token_valid': True,
            'error': 'Password must be at least 8 characters.',
        })

    if password != confirm_password:
        return render(request, 'reset_password.html', {
            'token': token,
            'token_valid': True,
            'error': 'Passwords do not match.',
        })

    # Update password in Supabase
    user_id = reset_data['user_id']
    try:
        supabase_admin.table('users')\
            .update({'password_hash': make_password(password)})\
            .eq('id', user_id)\
            .execute()

        # Mark token as used
        supabase_admin.table('password_resets')\
            .update({'used': True})\
            .eq('token', token)\
            .execute()

    except Exception as e:
        print(f"Password update error: {e}")
        return render(request, 'reset_password.html', {
            'token': token,
            'token_valid': True,
            'error': 'Something went wrong. Please try again.',
        })

    return render(request, 'reset_password.html', {
        'success': True,
        'token_valid': False,
    })


@csrf_exempt
@require_http_methods(["POST"])
def save_crop(request):
    """Save a cropped detection image for later 3D viewing"""
    user_session = get_user_from_session(request)
    if not user_session:
        return JsonResponse({'error': 'Unauthorized'}, status=401)

    try:
        data = json.loads(request.body)
        object_class = data.get('object_class', '')
        confidence = data.get('confidence', 0)
        crop_image = data.get('crop_image', '')

        if not object_class or not crop_image:
            return JsonResponse({'error': 'Missing data'}, status=400)

        resp = supabase_admin.table('detection_crops').insert({
            'user_id': str(user_session['users']['id']),
            'object_class': object_class,
            'confidence': confidence,
            'crop_image': crop_image,
        }).execute()

        return JsonResponse({'success': True})
    except Exception as e:
        print(f"Save crop error: {e}")
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
@require_http_methods(["GET"])
def get_latest_crop(request):
    """Get the latest saved crop for a given object class"""
    user_session = get_user_from_session(request)
    if not user_session:
        return JsonResponse({'error': 'Unauthorized'}, status=401)

    object_class = request.GET.get('object_class', '')
    if not object_class:
        return JsonResponse({'error': 'Missing object_class'}, status=400)

    try:
        resp = supabase_admin.table('detection_crops')\
            .select('object_class, confidence, crop_image')\
            .eq('user_id', str(user_session['users']['id']))\
            .eq('object_class', object_class)\
            .order('created_at', desc=True)\
            .limit(1)\
            .execute()

        if resp.data and len(resp.data) > 0:
            return JsonResponse({
                'success': True,
                'object_class': resp.data[0]['object_class'],
                'confidence': resp.data[0]['confidence'],
                'crop_image': resp.data[0]['crop_image'],
            })
        else:
            return JsonResponse({'success': False, 'error': 'No crop found for this class'})
    except Exception as e:
        print(f"Get crop error: {e}")
        return JsonResponse({'error': str(e)}, status=500)
    
@csrf_exempt
@require_http_methods(["POST"])
def clear_detection_history(request):
    user_session = get_user_from_session(request)
    if not user_session:
        return JsonResponse({'error': 'Unauthorized'}, status=401)

    user_id = user_session['users']['id']

    try:
        supabase_admin.table('detection_history')\
            .delete()\
            .eq('user_id', str(user_id))\
            .execute()

        return JsonResponse({'success': True})
    except Exception as e:
        print(f"Clear history error: {e}")
        return JsonResponse({'error': str(e)}, status=500)
    


@csrf_exempt
@require_http_methods(["POST"])
def generate_3d_mesh(request):
    """Generate 3D mesh using TripoSR via AIML API"""
    try:
        data = json.loads(request.body)
        image_base64 = data.get('image_base64', '')

        if not image_base64:
            return JsonResponse({'error': 'No image provided'}, status=400)

        # Strip data URI prefix if present
        if 'base64,' in image_base64:
            image_base64 = image_base64.split('base64,')[1]

        # Upload to imgbb to get a public URL (free, no key needed for basic)
        imgbb_resp = http_requests.post('https://api.imgbb.com/1/upload', data={
            'key': settings.IMGBB_API_KEY,
            'image': image_base64,
        }, timeout=30)

        if imgbb_resp.status_code != 200:
            return JsonResponse({'error': 'Image upload failed'}, status=500)

        imgbb_data = imgbb_resp.json()
        public_url = imgbb_data.get('data', {}).get('url', '')

        if not public_url:
            return JsonResponse({'error': 'Could not get image URL'}, status=500)

        # Call AIML API for TripoSR
        aiml_response = http_requests.post(
            'https://api.aimlapi.com/v1/images/generations',
            headers={
                'Authorization': 'Bearer ' + settings.AIML_API_KEY,
                'Content-Type': 'application/json',
            },
            json={
                'model': 'triposr',
                'image_url': public_url,
                'output_format': 'glb',
                'do_remove_background': True,
                'foreground_ratio': 0.9,
                'mc_resolution': 256,
            },
            timeout=60
        )

        if aiml_response.status_code != 200:
            return JsonResponse({
                'error': 'TripoSR API error: ' + str(aiml_response.status_code),
                'detail': aiml_response.text[:500]
            }, status=500)

        result = aiml_response.json()

        glb_url = None
        if 'model_mesh' in result and 'url' in result['model_mesh']:
            glb_url = result['model_mesh']['url']

        if not glb_url:
            return JsonResponse({'error': 'No 3D model returned from API'}, status=500)

        return JsonResponse({
            'success': True,
            'glb_url': glb_url,
            'method': 'triposr'
        })

    except Exception as e:
        print(f"3D mesh generation error: {e}")
        import traceback
        traceback.print_exc()
        return JsonResponse({'error': str(e)}, status=500)