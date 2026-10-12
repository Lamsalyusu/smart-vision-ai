import logging
from django.conf import settings
from groq import Groq
from backend.ml_model.scene_graph import scene_graph_builder
from backend.supabase_client import supabase_admin

logger = logging.getLogger(__name__)


class ChatAssistantService:
    """Provides spatially contextual reasoning via Groq and dynamic scene graphs."""

    def __init__(self, model_name="llama-3.1-8b-instant"):
        self.model_name = model_name

    def generate_response(self, user_message, detections, user_id=None):
        """
        Synthesize spatial scene graph and generate conversational AI response.
        """
        scene_graph = scene_graph_builder.build(detections)
        context_prompt = scene_graph["summary_prompt"]

        client = Groq(api_key=settings.GROQ_API_KEY)
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are SmartVision AI, an intelligent real-time spatial vision assistant. "
                        "You understand what the camera feed sees, including tracked objects, estimated "
                        "distances, velocities, and spatial relationships between objects. "
                        "Answer clearly, accurately, and concisely."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Current Scene Perception: {context_prompt}\n\n"
                        f"User: {user_message}"
                    ),
                },
            ],
            temperature=0.7,
            max_tokens=500,
        )

        ai_text = response.choices[0].message.content

        # Save query history
        if user_id:
            try:
                supabase_admin.table("llm_queries").insert({
                    "user_id": str(user_id),
                    "message": user_message[:1000],
                    "response": ai_text[:2000],
                    "detection_context": context_prompt[:500],
                }).execute()
            except Exception:
                logger.exception("Failed to persist LLM query for user=%s", user_id)

        return {
            "response": ai_text,
            "scene_graph": scene_graph,
        }


chat_service = ChatAssistantService()
