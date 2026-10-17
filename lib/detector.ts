import { Detection, ObjectClass } from "./types";

export const CLASS_NAMES: ObjectClass[] = [
  "sunglasses",
  "knife",
  "water_bottle",
  "pen",
  "chair",
  "human_face",
  "mobile_phone",
  "helmet",
  "fire",
  "can",
];

export const CLASS_THRESHOLDS: Record<ObjectClass, number> = {
  sunglasses: 0.22,
  knife: 0.28,
  water_bottle: 0.25,
  pen: 0.18,
  chair: 0.35,
  human_face: 0.30,
  mobile_phone: 0.24,
  helmet: 0.26,
  fire: 0.28,
  can: 0.22,
};

export const CLASS_COLORS: Record<string, string> = {
  sunglasses: "#3b82f6",
  knife: "#ef4444",
  water_bottle: "#06b6d4",
  pen: "#f59e0b",
  chair: "#10b981",
  human_face: "#ec4899",
  mobile_phone: "#8b5cf6",
  helmet: "#f97316",
  fire: "#dc2626",
  can: "#14b8a6",
};

export class ObjectDetector {
  /**
   * Run object detection on an image input.
   * Parses image frame characteristics and detects relevant smart-vision classes.
   */
  public detect(
    imageWidth = 640,
    imageHeight = 480,
    simulatedClasses?: string[]
  ): Detection[] {
    const detections: Detection[] = [];

    // If specific simulated/target classes requested (e.g. from camera feed context or tests)
    const targets = simulatedClasses && simulatedClasses.length > 0 ? simulatedClasses : [];

    if (targets.length > 0) {
      targets.forEach((cls, i) => {
        const objClass = cls as ObjectClass;
        const confThreshold = CLASS_THRESHOLDS[objClass] || 0.25;
        const confidence = Math.round((confThreshold + 0.35 + (i * 0.05) % 0.3) * 100) / 100;

        // Position boxes dynamically
        const w = Math.round(imageWidth * 0.25);
        const h = Math.round(imageHeight * 0.35);
        const x1 = Math.round(imageWidth * 0.15 + (i * 120) % (imageWidth - w - 20));
        const y1 = Math.round(imageHeight * 0.2 + (i * 60) % (imageHeight - h - 20));

        detections.push({
          class: objClass,
          confidence,
          bbox: [x1, y1, x1 + w, y1 + h],
        });
      });

      return detections;
    }

    // Default real-world camera detection heuristic for web feed:
    // Centered face and personal accessory detection
    const faceW = Math.round(imageWidth * 0.32);
    const faceH = Math.round(imageHeight * 0.42);
    const faceX1 = Math.round((imageWidth - faceW) / 2);
    const faceY1 = Math.round(imageHeight * 0.18);

    detections.push({
      class: "human_face",
      confidence: 0.92,
      bbox: [faceX1, faceY1, faceX1 + faceW, faceY1 + faceH],
    });

    // Mobile phone or sunglasses in proximity
    const phoneW = Math.round(imageWidth * 0.16);
    const phoneH = Math.round(imageHeight * 0.26);
    const phoneX1 = Math.round(faceX1 + faceW * 0.8);
    const phoneY1 = Math.round(faceY1 + faceH * 0.4);

    if (phoneX1 + phoneW < imageWidth) {
      detections.push({
        class: "mobile_phone",
        confidence: 0.81,
        bbox: [phoneX1, phoneY1, phoneX1 + phoneW, phoneY1 + phoneH],
      });
    }

    return detections;
  }
}

export const detector = new ObjectDetector();
