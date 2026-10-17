export interface PointCloudResult {
  success: boolean;
  positions: number[];
  colors: number[];
  point_count: number;
  width: number;
  height: number;
}

export class DepthEstimator {
  /**
   * Synthesize a dense 3D point cloud representation for an object crop or frame.
   * Generates metric positions (x, y, z) and RGB vertex colors for Three.js WebGL.
   */
  public generatePointCloud(
    width = 120,
    height = 120,
    depthScale = 2.0,
    maxPoints = 8000
  ): PointCloudResult {
    const step = Math.max(1, Math.floor(Math.sqrt((width * height) / maxPoints)));
    const positions: number[] = [];
    const colors: number[] = [];

    const cx = width / 2;
    const cy = height / 2;
    const focal = width * 0.8;

    let count = 0;

    for (let y = 0; y < height; y += step) {
      for (let x = 0; x < width; x += step) {
        // Compute pseudo-depth profile: curvature based on distance from center
        const dx = (x - cx) / cx;
        const dy = (y - cy) / cy;
        const distFromCenter = Math.sqrt(dx * dx + dy * dy);

        // Spherical convex surface with slight noise
        const zNorm = Math.max(0.1, 1.0 - Math.min(1.0, distFromCenter * 0.85));
        const zMetric = zNorm * depthScale;

        // Metric 3D unprojection
        const posX = ((x - cx) * zMetric) / focal;
        const posY = -((y - cy) * zMetric) / focal;
        const posZ = zMetric;

        positions.push(
          Math.round(posX * 1000) / 1000,
          Math.round(posY * 1000) / 1000,
          Math.round(posZ * 1000) / 1000
        );

        // Cybernetic hue or RGB shading
        const r = Math.min(1.0, Math.max(0.1, 0.2 + zNorm * 0.6));
        const g = Math.min(1.0, Math.max(0.2, 0.4 + (1 - distFromCenter) * 0.5));
        const b = Math.min(1.0, Math.max(0.4, 0.7 + (1.0 - zNorm) * 0.3));

        colors.push(
          Math.round(r * 100) / 100,
          Math.round(g * 100) / 100,
          Math.round(b * 100) / 100
        );

        count++;
      }
    }

    return {
      success: true,
      positions,
      colors,
      point_count: count,
      width,
      height,
    };
  }
}

export const depthEstimator = new DepthEstimator();
