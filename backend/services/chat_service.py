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
        Falls back to local spatial perception engine if API key or network is unavailable.
        """
        scene_graph = scene_graph_builder.build(detections)
        context_prompt = scene_graph["summary_prompt"]

        groq_key = getattr(settings, "GROQ_API_KEY", None)
        offline_mode = getattr(settings, "OFFLINE_MODE", False) or not groq_key or "placeholder" in str(groq_key).lower()

        if offline_mode:
            ai_text = self._generate_offline_response(user_message, scene_graph)
        else:
            try:
                client = Groq(api_key=groq_key)
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
            except Exception as exc:
                logger.warning("Groq API unavailable (%s), falling back to local spatial engine", exc)
                ai_text = self._generate_offline_response(user_message, scene_graph)

        # Save query history
        if user_id:
            try:
                from backend.models import LLMQuery, User
                user_obj = User.objects.filter(id=user_id).first()
                if user_obj:
                    LLMQuery.objects.create(
                        user=user_obj,
                        query_text=user_message[:1000],
                        llm_response=ai_text[:2000],
                        context_objects=scene_graph.get("nodes"),
                    )
            except Exception:
                logger.debug("Failed to persist LLM query to local ORM")

            if supabase_admin and not getattr(settings, "OFFLINE_MODE", False):
                try:
                    supabase_admin.table("llm_queries").insert({
                        "user_id": str(user_id),
                        "query_text": user_message[:1000],
                        "llm_response": ai_text[:2000],
                        "context_objects": scene_graph.get("nodes"),
                    }).execute()
                except Exception:
                    pass

        return {
            "response": ai_text,
            "scene_graph": scene_graph,
        }

    def _generate_offline_response(self, user_message, scene_graph):
        """Local deterministic spatial reasoning engine for offline/evaluation environments."""
        nodes = scene_graph.get("nodes", [])
        edges = scene_graph.get("edges", [])

        if not nodes:
            return "SmartVision Perception (Local Engine): No objects detected in the current camera feed."

        obj_summary = ", ".join([f"{n['class']} (Track #{n['track_id']})" for n in nodes])
        text = f"SmartVision Perception (Local Engine): Detected {len(nodes)} object(s): {obj_summary}."

        if edges:
            rel_str = "; ".join([f"{e['source_class']} is {e['relation']} {e['target_class']} (~{e['distance_m']}m)" for e in edges])
            text += f" Spatial relations: {rel_str}."

        if "distance" in user_message.lower():
            dists = [f"{n['class']} at ~{n.get('spatial', {}).get('distance_m', 'N/A')}m" for n in nodes if 'spatial' in n]
            if dists:
                text += " Distance estimates: " + ", ".join(dists) + "."

        return text


chat_service = ChatAssistantService()
