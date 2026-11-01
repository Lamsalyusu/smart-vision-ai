export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface DetectionRecord {
  id: string;
  className: string;
  confidence: number;
  bbox: BoundingBox;
  depthM?: number;
  spatialCoords?: [number, number, number];
}

export interface SpatialTrack {
  trackId: number;
  className: string;
  state: "tentative" | "confirmed" | "lost";
  confidence: number;
  position3D: [number, number, number];
  velocity3D: [number, number, number];
  ageFrames: number;
}

export interface MeshReconstructionResponse {
  success: boolean;
  objectClass: string;
  format: string;
  threejsGeometry?: {
    metadata: Record<string, unknown>;
    data: {
      attributes: {
        position: { itemSize: number; type: string; array: number[] };
        normal: { itemSize: number; type: string; array: number[] };
      };
      index: { type: string; array: number[] };
    };
  };
  statistics: {
    vertex_count: number;
    face_count: number;
    surface_area_sq_m: number;
  };
}
