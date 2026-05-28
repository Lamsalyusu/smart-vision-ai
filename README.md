# SmartVision AI

Real-time object detection web app — Final Year Project, Herald College Kathmandu (University of Wolverhampton), 2026.

Detects 10 object classes through a live camera feed using a custom YOLO-inspired model, integrated with depth estimation, 3D reconstruction, and an AI chatbot — deployed as a Progressive Web App.

**Detection Classes:** sunglasses · knife · water bottle · pen · chair · human face · mobile phone · helmet · fire · can

---

## Features

- Real-time object detection (custom CSPDarknet-inspired backbone, 7.9M parameters, TF 2.19)
- Monocular depth estimation using MiDaS
- Single-image 3D reconstruction using TripoSR
- Conversational AI chatbot via Groq + Llama 3
- Detection history dashboard (Supabase)
- Authentication — Email/password, Google OAuth, GitHub OAuth
- Deployed as a Progressive Web App (PWA)

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Django |
| ML Model | TensorFlow / Keras (TF 2.19) |
| Depth & 3D | MiDaS + TripoSR |
| LLM Chatbot | Groq API (Llama 3) |
| Database | Supabase |
| Auth | Email, Google OAuth, GitHub OAuth |

---

## Setup

```bash
git clone https://github.com/Lamsalyusu/smartvision-ai
cd smartvision-ai
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env` in root:
SUPABASE_URL=

SUPABASE_ANON_KEY=

SUPABASE_SERVICE_ROLE_KEY=

GROQ_API_KEY=

EMAIL_HOST_USER=

EMAIL_HOST_PASSWORD=

GOOGLE_CLIENT_ID=

GOOGLE_CLIENT_SECRET=

GITHUB_CLIENT_ID=

GITHUB_CLIENT_SECRET=

Place `best_weights.weights.h5` in `backend/ml_model/weights/`, then:

```bash
python manage.py migrate
python manage.py runserver
```

Open `http://127.0.0.1:8000`

---

## Author

Yuyutsu Lamsal — Student ID: 2432214  
Herald College Kathmandu | University of Wolverhampton  
[LinkedIn](https://linkedin.com/in/yuyutsu-lamsal-6250472b3)
