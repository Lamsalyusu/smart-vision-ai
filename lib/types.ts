export type ObjectClass =
  | "sunglasses"
  | "knife"
  | "water_bottle"
  | "pen"
  | "chair"
  | "human_face"
  | "mobile_phone"
  | "helmet"
  | "fire"
  | "can";

export interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface Spatial3D {
  x: number;
  y: number;
  z: number;
  distance_m: number;
  depth_m: number;
  width_m: number;
  height_m: number;
  elevation_m?: number;
}

export interface Detection {
  class: ObjectClass | string;
  confidence: number;
  bbox: [number, number, number, number]; // [x1, y1, x2, y2]
  track_id?: number;
  velocity?: [number, number];
  spatial_3d?: Spatial3D;
  spatial?: Spatial3D;
}

export interface SceneGraphNode {
  track_id: number;
  class: string;
  confidence: number;
  bbox: [number, number, number, number];
  spatial?: Spatial3D;
  velocity?: [number, number];
}

export interface SceneGraphEdge {
  source_id: number;
  source_class: string;
  target_id: number;
  target_class: string;
  relation: string;
  directional: string;
  distance_m?: number;
}

export interface SceneGraph {
  nodes: SceneGraphNode[];
  edges: SceneGraphEdge[];
  summary_prompt: string;
  graph_metrics: {
    total_nodes: number;
    total_edges: number;
    density: number;
    most_central_node: string | null;
  };
}

/** Measured from recorded frame timings; null until frames have been processed. */
export interface PerformanceStats {
  sample_frames: number;
  window_minutes: number;
  frames_in_window: number;
  frames_per_minute: number;
  inference_ms_p50: number | null;
  inference_ms_p95: number | null;
  roundtrip_ms_p50: number | null;
  roundtrip_ms_p95: number | null;
  detections_per_frame: number | null;
}

export interface TelemetryStats {
  total_detections: number;
  active_sessions: number;
  class_distribution: Record<string, number>;
  performance?: PerformanceStats;
  recent_activity: Array<{
    id: string;
    class: string;
    confidence: number;
    distance_m?: number;
    timestamp: string;
    has_crop: boolean;
  }>;
}
