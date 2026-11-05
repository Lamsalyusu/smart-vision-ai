import { MeshReconstructionResponse } from "./types.js";

export class SmartVisionClient {
  private baseUrl: string;
  private apiToken?: string;

  constructor(options: { baseUrl?: string; apiToken?: string } = {}) {
    this.baseUrl = (options.baseUrl || "http://localhost:8000").replace(/\/$/, "");
    this.apiToken = options.apiToken;
  }

  async getHealth(): Promise<{ status: string }> {
    const res = await fetch(`${this.baseUrl}/health`);
    if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
    return res.json();
  }

  async reconstruct3DMesh(params: {
    objectClass?: string;
    points?: number[][];
    voxelSize?: number;
    smoothingIterations?: number;
  }): Promise<MeshReconstructionResponse> {
    const res = await fetch(`${this.baseUrl}/mesh/reconstruct`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        // The FastAPI service is internal: apiToken is its INTERNAL_API_TOKEN.
        ...(this.apiToken ? { "X-Internal-Token": this.apiToken } : {}),
      },
      body: JSON.stringify({
        object_class: params.objectClass || "chair",
        points: params.points,
        voxel_size: params.voxelSize || 0.08,
        smoothing_iterations: params.smoothingIterations || 2,
        output_format: "threejs",
      }),
    });

    if (!res.ok) {
      throw new Error(`3D mesh reconstruction failed: ${res.statusText}`);
    }

    return res.json();
  }
}
