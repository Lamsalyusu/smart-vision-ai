import { NextRequest, NextResponse } from "next/server";
import { sceneGraphBuilder } from "@/lib/scene-graph";
import { Detection } from "@/lib/types";

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const userMessage = (body.message || "").trim();
    const detections: Detection[] = body.detections || [];

    if (!userMessage) {
      return NextResponse.json({ error: "Message is required" }, { status: 400 });
    }

    // 1. Build Spatial Scene Graph
    const sceneGraph = sceneGraphBuilder.build(detections);

    // 2. Groq LLM integration or smart fallback
    const groqApiKey = process.env.GROQ_API_KEY;
    let aiResponse = "";

    if (groqApiKey) {
      try {
        const groqRes = await fetch("https://api.groq.com/openai/v1/chat/completions", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${groqApiKey}`,
          },
          body: JSON.stringify({
            model: "llama-3.1-8b-instant",
            messages: [
              {
                role: "system",
                content: `You are SmartVision AI, an advanced spatial perception assistant. You have access to real-time 3D detections and the following spatial scene graph:\n\n${sceneGraph.summary_prompt}\n\nAnswer concisely, accurately, and reference spatial coordinates, distances, and object relationships where relevant.`,
              },
              { role: "user", content: userMessage },
            ],
            temperature: 0.2,
            max_tokens: 300,
          }),
        });

        if (groqRes.ok) {
          const data = await groqRes.json();
          aiResponse = data.choices?.[0]?.message?.content || "";
        }
      } catch (err) {
        console.warn("Groq API request failed, using local reasoning engine:", err);
      }
    }

    // Contextual spatial reasoning engine if Groq not configured or errored
    if (!aiResponse) {
      const lower = userMessage.toLowerCase();
      if (detections.length === 0) {
        aiResponse = "I currently do not perceive any objects in the camera frame. Point your camera at objects such as a human face, sunglasses, phone, water bottle, or chair to begin spatial perception.";
      } else if (lower.includes("how many") || lower.includes("count")) {
        aiResponse = `I observe ${detections.length} tracked object(s) in the frame: ${detections.map((d) => d.class.replace("_", " ")).join(", ")}.`;
      } else if (lower.includes("distance") || lower.includes("far") || lower.includes("close")) {
        const distItems = detections
          .filter((d) => d.spatial_3d?.distance_m !== undefined)
          .map((d) => `${d.class.replace("_", " ")} is ~${d.spatial_3d?.distance_m}m away (Z-depth: ${d.spatial_3d?.z}m)`);
        aiResponse = distItems.length > 0
          ? `Metric 3D distances: ${distItems.join("; ")}.`
          : "Depth estimation is currently computing metric 3D ranges.";
      } else if (lower.includes("where") || lower.includes("position") || lower.includes("relation")) {
        if (sceneGraph.edges.length > 0) {
          aiResponse = `Spatial scene graph: ${sceneGraph.edges.map((e) => `${e.source_class.replace("_", " ")} is ${e.relation.replace("_", " ")} ${e.target_class.replace("_", " ")} (${e.directional})`).join(". ")}.`;
        } else {
          aiResponse = `Observed objects: ${detections.map((d) => `${d.class} (ID #${d.track_id || 1})`).join(", ")}. No proximity thresholds are currently intersected.`;
        }
      } else {
        aiResponse = `Perceived scene: ${detections.map((d) => `${d.class.replace("_", " ")} (${Math.round(d.confidence * 100)}%)`).join(", ")}. Central spatial node is ${sceneGraph.graph_metrics.most_central_node || "Human Face"}. How can I assist with this scene?`;
      }
    }

    return NextResponse.json({
      success: true,
      response: aiResponse,
      scene_graph: sceneGraph,
    });
  } catch (err: unknown) {
    console.error("Chat API error:", err);
    return NextResponse.json({ error: "Chat service error" }, { status: 500 });
  }
}
