# SmartVision AI (Final Year Project)
A web app that finds objects in your camera feed in real time. This is my final year project at Herald College Kathmandu (University of Wolverhampton), 2026.
It can spot 10 things: sunglasses, knife, water bottle, pen, chair, human face, mobile phone, helmet, fire and can.

*  I faced a lot of problems while doing this project because this was mine first project that related to AI.
* At first i thought that the model trainig part will be much easier and will not be time consuming but after i started i  got to learn many things and i failed at some of the part.  

## What does this project do:
- Detects objects live from your webcam, using a YOLO-style model I built in TensorFlow (about 7.9M parameters)
- Estimates depth with MiDaS and shows it as a 3D point cloud
- Turns a single image into a 3D model with TripoSR (this needs a separate Colab notebook running)
- Has a chat assistant (Groq, Llama 3) that knows what the camera is seeing
- Login with email, Google or GitHub, and a dashboard with your detection history
- Can be used as a PWA

## What is it built with 
I built this project with the Tech stacks like Django, TensorFlow/Keras, MiDaS, TripoSR, Groq, Supabase

## About running the project
Use Python 3.12. Newer versions may not work with TensorFlow.
You can follow the below commands to run the project my project.

git clone https://github.com/Lamsalyusu/object_detection_system
cd object_detection_system
python3.12 -m venv .venv
source .venv/bin/activate        
on Windows: .venv\Scripts\activate
pip install -r requirements.txt

for env file you can take help from .env.example where i have mentioned all of the requirements that are required.
Or
You can see below:

Make a `.env` file in the project folder and add these things:
SECRET_KEY=any-long-random-string
DJANGO_DEBUG=True
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
TRIPOSR_API_URL=

`TRIPOSR_API_URL` is only needed for the 3D model feature but for now it only supports midas depth estimation.
Keep `DJANGO_DEBUG=True` while developing, otherwise the CSS and JS won't load.

The model which you get after training like this -> `best_weights.weights.h5` should be put inside `backend/ml_model/weights/`.
Set up the database by running `db/schema.sql` in the Supabase SQL editor. Then:

python manage.py migrate
python manage.py runserver
Open http://127.0.0.1:8000

The `migrate` step only creates Django's own session tables. My data lives in Supabase.

## Tests
You can use the following command for testing the project 
`python manage.py test`

## Author

Yuyutsu Lamsal  
Herald College Kathmandu