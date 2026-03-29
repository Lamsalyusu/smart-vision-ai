# SmartVision AI

Real-time object detection web application built as a Final Year Project at Herald College Kathmandu (University of Wolverhampton).

## About

SmartVision AI detects 10 object classes through a live camera feed using a custom YOLO-inspired model built with TensorFlow/Keras. It includes an AI chatbot for object explanations, 3D depth visualization, and a detection history dashboard.

Detection classes: sunglasses, knife, water bottle, pen, chair, human face, mobile phone, helmet, fire, can.

## Tech Stack
 Backend: Django
 ML Model: TensorFlow/Keras (custom YOLO, 7.9M parameters)
 3D Viewer: TRIPOSR with MiDaS depth estimation
 Database: Supabase
 LLM Chatbot: Groq API (Llama 3)
 Authentication: Email/password, Google OAuth, GitHub OAuth

## Setup

1. Clone the repository and navigate to the project folder.

2. Create a virtual environment and install dependencies:
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

3. Create a `.env` file with your credentials:

```
SUPABASE_URL=your_url
SUPABASE_ANON_KEY=your_key
SUPABASE_SERVICE_ROLE_KEY=your_key
GROQ_API_KEY=your_key
EMAIL_HOST_USER=your_email
EMAIL_HOST_PASSWORD=your_app_password
GOOGLE_CLIENT_ID=your_id
GOOGLE_CLIENT_SECRET=your_secret
GITHUB_CLIENT_ID=your_id
GITHUB_CLIENT_SECRET=your_secret
```

4. Place model weights (`best_weights.weights.h5`) in `backend/ml_model/weights/`.

5. Run the server:

```
python manage.py runserver
```

6. Open `http://127.0.0.1:8000` in your browser.

## Author

Yuyutsu Lamsal (Student ID: 2432214)
Herald College Kathmandu
